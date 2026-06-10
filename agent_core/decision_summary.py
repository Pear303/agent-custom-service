"""DecisionSummaryExtractor: 从被裁剪的工具调用组中提取决策摘要。

从 agent.decision_summary 迁移，更新 import: from .context_view import PrunedToolCallGroup
"""
from __future__ import annotations

import re
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .context_view import PrunedToolCallGroup


_DECISION_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?:决定|选择|使用|采用|不用|避免|改为|换用)[^\n。！？]{3,80}', re.IGNORECASE),
    re.compile(r'(?:decided|chose|use|avoid|prefer|switch to|opt for)[^\n.!?]{3,80}', re.IGNORECASE),
    re.compile(r'(?:因为|由于|原因是|目的是)[^\n。！？]{3,80}', re.IGNORECASE),
    re.compile(r'(?:because|since|the reason|in order to)[^\n.!?]{3,80}', re.IGNORECASE),
]


def _summarize_tool_call(tc: dict, tm: BaseMessage | None = None) -> str:
    name = tc.get("name", "unknown")
    args = tc.get("args", {})
    content = ""
    if tm is not None:
        c = getattr(tm, "content", "")
        content = c if isinstance(c, str) else str(c)

    if name == "read_file":
        path = args.get("path", "?")
        preview = content[:80].replace("\n", " ").strip()
        return f"read_file({path}) → {preview}..." if len(content) > 80 else f"read_file({path}) → {preview}"

    if name == "write_file":
        path = args.get("path", "?")
        return f"write_file({path}) → 已创建/覆盖"

    if name == "edit_file":
        path = args.get("path", "?")
        replacements = args.get("replacements", [])
        n = len(replacements) if isinstance(replacements, list) else 1
        return f"edit_file({path}) → {n}处修改"

    if name == "run_command":
        cmd = args.get("command", "?")
        cmd_short = cmd[:50] + "..." if len(cmd) > 50 else cmd
        exit_info = ""
        if "exit code" in content.lower() or "exited" in content.lower():
            for line in content.split("\n"):
                if "exit" in line.lower():
                    exit_info = f" → {line.strip()[:40]}"
                    break
        return f"run_command({cmd_short}){exit_info}"

    if name in ("glob_tool", "glob"):
        pattern = args.get("pattern", "?")
        count = content.count("\n") + 1 if content else 0
        return f"glob({pattern}) → {count}个文件"

    if name in ("grep_tool", "grep", "grep_search"):
        pattern = args.get("pattern", "?")
        count = content.count("\n") + 1 if content else 0
        return f"grep({pattern}) → {count}处匹配"

    if name in ("web_fetch", "web_search"):
        url = args.get("url", args.get("query", "?"))
        preview = content[:60].replace("\n", " ").strip()
        return f"{name}({url[:50]}) → {preview}..."

    if name == "update_todos":
        return "update_todos → 已更新待办列表"

    args_str = ", ".join(
        f"{k}={str(v)[:30]}" for k, v in list(args.items())[:3]
    )
    return f"{name}({args_str})"


def _extract_decisions(ai_content: str) -> list[str]:
    if not ai_content:
        return []
    decisions: list[str] = []
    for line in ai_content.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pat in _DECISION_PATTERNS:
            m = pat.search(line)
            if m:
                decisions.append(m.group(0).strip())
                break
    return decisions


_MAX_SUMMARY_CHARS = 4000


class DecisionSummaryExtractor:
    """从被裁剪的工具调用组中提取决策摘要。纯规则，零幻觉。"""

    def extract(self, pruned_groups: list[PrunedToolCallGroup]) -> str:
        if not pruned_groups:
            return ""

        actions: list[str] = []
        decisions: list[str] = []
        file_changes: list[str] = []
        last_todos: str = ""

        for group in pruned_groups:
            for tc in group.tool_calls:
                tc_id = tc.get("id") or tc.get("tool_call_id")
                matching_tm = None
                for tm in group.tool_results:
                    if isinstance(tm, ToolMessage) and tm.tool_call_id == tc_id:
                        matching_tm = tm
                        break
                actions.append(_summarize_tool_call(tc, matching_tm))

            if group.ai_message and isinstance(group.ai_message, AIMessage):
                ai_content = group.ai_message.content
                if isinstance(ai_content, str):
                    decisions.extend(_extract_decisions(ai_content))

            for tc in group.tool_calls:
                name = tc.get("name", "")
                if name in ("write_file", "edit_file"):
                    path = tc.get("args", {}).get("path", "unknown")
                    label = "已创建/覆盖" if name == "write_file" else "已修改"
                    file_changes.append(f"{path}: {label}")

            for tc in group.tool_calls:
                if tc.get("name") == "update_todos":
                    todos_content = tc.get("args", {}).get("todos", "")
                    if todos_content:
                        last_todos = todos_content

        sections: list[str] = []
        sections.append("已完成操作:")
        for a in actions:
            sections.append(f"  - {a}")

        if decisions:
            sections.append("关键决策:")
            seen = set()
            for d in decisions:
                if d not in seen:
                    sections.append(f"  - {d}")
                    seen.add(d)

        if file_changes:
            sections.append("文件变更记录:")
            latest: dict[str, str] = {}
            for fc in file_changes:
                path, label = fc.split(": ", 1)
                latest[path] = label
            for path, label in latest.items():
                sections.append(f"  - {path}: {label}")

        if last_todos:
            todos_preview = last_todos[:200]
            if len(last_todos) > 200:
                todos_preview += "..."
            sections.append(f"当前待办: {todos_preview}")

        max_step = max(g.group_index for g in pruned_groups)
        header = f"[上下文摘要 — 裁剪点: step {max_step}, 共裁剪 {len(pruned_groups)} 组]"

        result = header + "\n" + "\n".join(sections)

        if len(result) > _MAX_SUMMARY_CHARS:
            result = result[:_MAX_SUMMARY_CHARS] + "\n  ... (摘要已截断)"

        return result


def merge_summaries(old_summary: str, new_summary: str) -> str:
    if not old_summary:
        return new_summary
    if not new_summary:
        return old_summary

    def _parse_sections(summary: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current_section = ""
        for line in summary.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[上下文摘要"):
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                current_section = stripped.rstrip(":")
                sections.setdefault(current_section, [])
            elif stripped.startswith("- ") and current_section:
                sections[current_section].append(stripped[2:])
        return sections

    old_sections = _parse_sections(old_summary)
    new_sections = _parse_sections(new_summary)

    merged: dict[str, list[str]] = {}
    merged["已完成操作"] = new_sections.get("已完成操作", [])

    all_decisions = old_sections.get("关键决策", []) + new_sections.get("关键决策", [])
    seen = set()
    merged["关键决策"] = []
    for d in all_decisions:
        if d not in seen:
            merged["关键决策"].append(d)
            seen.add(d)

    file_changes: dict[str, str] = {}
    for fc in old_sections.get("文件变更记录", []):
        if ": " in fc:
            path, label = fc.split(": ", 1)
            file_changes[path] = label
    for fc in new_sections.get("文件变更记录", []):
        if ": " in fc:
            path, label = fc.split(": ", 1)
            file_changes[path] = label
    if file_changes:
        merged["文件变更记录"] = [f"{p}: {l}" for p, l in file_changes.items()]

    if "当前待办" in new_sections:
        merged["当前待办"] = new_sections["当前待办"]
    elif "当前待办" in old_sections:
        merged["当前待办"] = old_sections["当前待办"]

    lines = ["[上下文摘要 — 合并]"]
    for section_name, items in merged.items():
        lines.append(f"{section_name}:")
        for item in items:
            lines.append(f"  - {item}")

    result = "\n".join(lines)

    if len(result) > _MAX_SUMMARY_CHARS:
        if "已完成操作" in merged and len(merged["已完成操作"]) > 3:
            merged["已完成操作"] = merged["已完成操作"][-3:]
            merged["已完成操作"].append("... (更早的操作已省略)")
            lines = ["[上下文摘要 — 合并]"]
            for section_name, items in merged.items():
                lines.append(f"{section_name}:")
                for item in items:
                    lines.append(f"  - {item}")
            result = "\n".join(lines)

    return result[:_MAX_SUMMARY_CHARS]
