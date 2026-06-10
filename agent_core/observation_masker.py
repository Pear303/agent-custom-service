"""ObservationMasker: 对只读工具的大体积输出做观察遮蔽。

从 agent.observation_masker 迁移，无 import 变更。
"""
from __future__ import annotations

import re
from langchain_core.messages import BaseMessage, ToolMessage


_READ_ONLY_TOOLS = frozenset({
    "read_file", "glob_tool", "glob", "grep_tool", "grep",
    "grep_search", "web_fetch", "web_search",
})

_MIN_CONTENT_CHARS = 500

_LINE_NUM_PREFIX = re.compile(r'^\s*\d+\|\s')


def _strip_line_number(line: str) -> str:
    return _LINE_NUM_PREFIX.sub('', line)


_PY_KEYWORDS = (
    "import ", "from ", "def ", "class ", "async def ",
    "@", "#",
)
_JS_KEYWORDS = (
    "import ", "export ", "const ", "let ", "var ",
    "function ", "class ", "async function ",
    "interface ", "type ", "enum ",
    "//", "/*", "*/",
)
_GO_KEYWORDS = (
    "import ", "package ", "func ", "type ", "struct ",
    "interface ", "const ", "var ",
    "//", "/*", "*/",
)
_RS_KEYWORDS = (
    "use ", "mod ", "fn ", "struct ", "enum ",
    "impl ", "trait ", "pub fn ", "pub struct ",
    "pub enum ", "const ", "static ",
    "//", "/*", "*/", "///", "//!",
)
_JVM_KEYWORDS = (
    "import ", "package ", "public class ", "private class ",
    "class ", "interface ", "enum ",
    "public ", "private ", "protected ",
    "fun ", "object ", "data class ",
    "//", "/*", "*/",
)

_PREFIX_LOOKUP: dict[str, tuple[str, ...]] = {}
for _kw in set(_PY_KEYWORDS + _JS_KEYWORDS + _GO_KEYWORDS + _RS_KEYWORDS + _JVM_KEYWORDS):
    if _kw:
        _ch = _kw[0]
        if _kw not in _PREFIX_LOOKUP.get(_ch, ()):
            _PREFIX_LOOKUP[_ch] = _PREFIX_LOOKUP.get(_ch, ()) + (_kw,)


def _is_structural_line(code_line: str) -> bool:
    if not code_line:
        return True
    for text in (code_line, code_line.lstrip()):
        if not text:
            continue
        first = text[0]
        candidates = _PREFIX_LOOKUP.get(first)
        if candidates:
            for prefix in candidates:
                if text.startswith(prefix):
                    return True
    if code_line.lstrip().startswith("@"):
        return True
    if code_line.startswith(("```", "=== ", "--- ", "# ")):
        return True
    return False


class ObservationMasker:
    """对只读工具的大体积输出做观察遮蔽。"""

    def __init__(
        self,
        min_content_chars: int = _MIN_CONTENT_CHARS,
        max_file_lines: int = 60,
        max_grep_lines: int = 30,
        max_glob_items: int = 40,
    ):
        self.min_content_chars = min_content_chars
        self.max_file_lines = max_file_lines
        self.max_grep_lines = max_grep_lines
        self.max_glob_items = max_glob_items

    def mask(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        result = list(messages)
        for i, msg in enumerate(result):
            if not isinstance(msg, ToolMessage):
                continue
            if msg.name not in _READ_ONLY_TOOLS:
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) < self.min_content_chars:
                continue
            masked = self._apply_mask(msg.name, content)
            if masked != content:
                result[i] = ToolMessage(
                    content=masked,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    id=msg.id,
                )
        return result

    def _apply_mask(self, tool_name: str, content: str) -> str:
        if tool_name == "read_file":
            return self._mask_file_content(content)
        if tool_name in ("grep_tool", "grep", "grep_search"):
            return self._mask_grep_output(content)
        if tool_name in ("glob_tool", "glob"):
            return self._mask_glob_output(content)
        if tool_name in ("web_fetch", "web_search"):
            return self._mask_web_output(content)
        return content

    def _mask_file_content(self, content: str) -> str:
        lines = content.split("\n")
        kept_lines: list[str] = []
        output_lines = 0
        prev_was_ellipsis = False

        for line_idx, line in enumerate(lines):
            if output_lines >= self.max_file_lines:
                remaining = lines[line_idx:]
                tail_lines = self._extract_tail_errors(remaining)
                if tail_lines:
                    kept_lines.append("  ...")
                    kept_lines.extend(tail_lines)
                else:
                    kept_lines.append(f"  ... (共 {len(lines)} 行，已省略 {len(lines) - output_lines} 行)")
                break

            stripped = line.strip()
            code_line = _strip_line_number(stripped)

            if _is_structural_line(code_line):
                if not code_line and prev_was_ellipsis:
                    continue
                kept_lines.append(line)
                prev_was_ellipsis = False
            else:
                if not prev_was_ellipsis:
                    kept_lines.append("  ...")
                    prev_was_ellipsis = True

            output_lines = len([l for l in kept_lines if l.strip()])

        return "\n".join(kept_lines)

    def _mask_grep_output(self, content: str) -> str:
        lines = content.split("\n")
        if len(lines) <= self.max_grep_lines:
            return content
        head = lines[:self.max_grep_lines // 2]
        tail = lines[-self.max_grep_lines // 4:]
        return "\n".join(head) + f"\n  ... (省略 {len(lines) - len(head) - len(tail)} 行) ...\n" + "\n".join(tail)

    def _mask_glob_output(self, content: str) -> str:
        lines = content.split("\n")
        if len(lines) <= self.max_glob_items:
            return content
        head = lines[:self.max_glob_items]
        remaining = len(lines) - len(head)
        return "\n".join(head) + f"  ... (共 {len(lines)} 个文件，省略 {remaining} 个)"

    def _mask_web_output(self, content: str) -> str:
        if len(content) <= 800:
            return content
        return content[:800] + "\n... (内容已截断)"

    def _extract_tail_errors(self, remaining_lines: list[str]) -> list[str]:
        error_lines: list[str] = []
        error_keywords = ("error", "exception", "traceback", "failed", "错误", "异常")
        for idx, line in enumerate(remaining_lines):
            if any(kw in line.lower() for kw in error_keywords):
                start = max(0, idx - 2)
                end = min(len(remaining_lines), idx + 3)
                error_lines = remaining_lines[start:end]
                break
        return error_lines
