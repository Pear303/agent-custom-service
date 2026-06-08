"""DecisionSummaryExtractor: 从被裁剪的工具调用组中提取决策摘要。

设计原则（Verbatim Compaction）：
1. 不调用 LLM — 纯规则提取，零幻觉
2. 不改写原文 — 关键决策从 AIMessage.content 中原样摘取
3. 结构化输出 — 固定格式：已完成操作/关键决策/文件变更/待办
4. 可合并 — 多次压缩的摘要可规则合并，不累积膨胀

与主流方案对比：
- Claude Code: LLM 重写摘要（压缩比高但有幻觉风险）
- Morph Compact: 纯删除（零幻觉但完全丢失决策链）
- 本方案: 规则提取摘要（零幻觉 + 保留决策链）
"""
from __future__ import annotations

import re
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agent.context_view import PrunedToolCallGroup


# ── 关键决策提取正则 ──────────────────────────────────────────
_DECISION_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?:决定|选择|使用|采用|不用|避免|改为|换用)[^\n。！？]{3,80}', re.IGNORECASE),
    re.compile(r'(?:decided|chose|use|avoid|prefer|switch to|opt for)[^\n.!?]{3,80}', re.IGNORECASE),
    re.compile(r'(?:因为|由于|原因是|目的是)[^\n。！？]{3,80}', re.IGNORECASE),
    re.compile(r'(?:because|since|the reason|in order to)[^\n.!?]{3,80}', re.IGNORECASE),
]

# ── 各工具的摘要模板 ──────────────────────────────────────────

def _summarize_tool_call(tc: dict, tm: BaseMessage | None = None) -> str:
    """根据工具类型生成单行摘要。"""
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

    # 通用格式
    args_str = ", ".join(
        f"{k}={str(v)[:30]}" for k, v in list(args.items())[:3]
    )
    return f"{name}({args_str})"


def _extract_decisions(ai_content: str) -> list[str]:
    """从 AIMessage.content 中提取决策性陈述。"""
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
                break  # 每行最多匹配一个
    return decisions


# ── 最大摘要 token 数 ────────────────────────────────────────
_MAX_SUMMARY_CHARS = 4000  # 约 1000 tokens


class DecisionSummaryExtractor:
    """从被裁剪的工具调用组中提取决策摘要。纯规则，零幻觉。"""

    def extract(self, pruned_groups: list[PrunedToolCallGroup]) -> str:
        """提取决策摘要。

        Args:
            pruned_groups: 被裁剪掉的工具调用组列表

        Returns:
            结构化摘要文本，可直接注入为 SystemMessage content
        """
        if not pruned_groups:
            return ""

        actions: list[str] = []
        decisions: list[str] = []
        file_changes: list[str] = []
        last_todos: str = ""

        for group in pruned_groups:
            # 1. 已完成操作
            for tc in group.tool_calls:
                # 查找对应的 ToolMessage
                tc_id = tc.get("id") or tc.get("tool_call_id")
                matching_tm = None
                for tm in group.tool_results:
                    if isinstance(tm, ToolMessage) and tm.tool_call_id == tc_id:
                        matching_tm = tm
                        break
                actions.append(_summarize_tool_call(tc, matching_tm))

            # 2. 关键决策（从 AIMessage.content 提取）
            if group.ai_message and isinstance(group.ai_message, AIMessage):
                ai_content = group.ai_message.content
                if isinstance(ai_content, str):
                    decisions.extend(_extract_decisions(ai_content))

            # 3. 文件变更记录
            for tc in group.tool_calls:
                name = tc.get("name", "")
                if name in ("write_file", "edit_file"):
                    path = tc.get("args", {}).get("path", "unknown")
                    label = "已创建/覆盖" if name == "write_file" else "已修改"
                    file_changes.append(f"{path}: {label}")

            # 4. 最后一次 update_todos
            for tc in group.tool_calls:
                if tc.get("name") == "update_todos":
                    todos_content = tc.get("args", {}).get("todos", "")
                    if todos_content:
                        last_todos = todos_content

        # 构建摘要
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
            # 同一文件只保留最新记录
            latest: dict[str, str] = {}
            for fc in file_changes:
                path, label = fc.split(": ", 1)
                latest[path] = label
            for path, label in latest.items():
                sections.append(f"  - {path}: {label}")

        if last_todos:
            # 截断过长的 todos
            todos_preview = last_todos[:200]
            if len(last_todos) > 200:
                todos_preview += "..."
            sections.append(f"当前待办: {todos_preview}")

        max_step = max(g.group_index for g in pruned_groups)
        header = f"[上下文摘要 — 裁剪点: step {max_step}, 共裁剪 {len(pruned_groups)} 组]"

        result = header + "\n" + "\n".join(sections)

        # 限制摘要大小
        if len(result) > _MAX_SUMMARY_CHARS:
            result = result[:_MAX_SUMMARY_CHARS] + "\n  ... (摘要已截断)"

        return result


def merge_summaries(old_summary: str, new_summary: str) -> str:
    """合并新旧决策摘要。

    策略：
    - 关键决策：累积（不删除旧决策）
    - 文件变更记录：同一文件只保留最新状态
    - 已完成操作：只保留新的（旧的已不需要）
    - 待办：只保留最新的
    """
    if not old_summary:
        return new_summary
    if not new_summary:
        return old_summary

    # 解析各节
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

    # 已完成操作：只保留新的
    merged["已完成操作"] = new_sections.get("已完成操作", [])

    # 关键决策：累积去重
    all_decisions = old_sections.get("关键决策", []) + new_sections.get("关键决策", [])
    seen = set()
    merged["关键决策"] = []
    for d in all_decisions:
        if d not in seen:
            merged["关键决策"].append(d)
            seen.add(d)

    # 文件变更记录：同一文件只保留最新
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

    # 待办：只保留最新的
    if "当前待办" in new_sections:
        merged["当前待办"] = new_sections["当前待办"]
    elif "当前待办" in old_sections:
        merged["当前待办"] = old_sections["当前待办"]

    # 构建合并后的摘要
    lines = ["[上下文摘要 — 合并]"]
    for section_name, items in merged.items():
        lines.append(f"{section_name}:")
        for item in items:
            lines.append(f"  - {item}")

    result = "\n".join(lines)

    if len(result) > _MAX_SUMMARY_CHARS:
        # 优先裁剪"已完成操作"（最不重要）
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
