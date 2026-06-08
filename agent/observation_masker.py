"""ObservationMasker: 对只读工具的大体积输出做观察遮蔽。

设计原则（Verbatim Compaction）：
1. 不生成新文本 — 只删除/保留原文片段，零幻觉
2. 保留结构信息 — 函数签名、装饰器、import、类定义原样保留
3. 只处理只读工具 — write_file/edit_file 的输出不遮蔽（Agent 需要确认结果）
4. 小体积不遮蔽 — 低于阈值的内容不做处理，避免过度压缩
5. 多语言支持 — 识别 Python/JS/TS/Go/Rust/Java/Kotlin 等语言的结构行

灵感来源：
- Morph Compact 的 Verbatim Deletion：不重写，只删除低信号 token
- Claude Code 的 microCompact：利用 cache_editing 原地修改

典型效果：
  read_file 返回 800 行 → 遮蔽后 ~30 行（保留签名+import+装饰器）
  压缩比 ~96%，保留完整结构信息，Agent 可随时重新读取
"""
from __future__ import annotations

import re
from langchain_core.messages import BaseMessage, ToolMessage


# 只读工具集合 — 这些工具的输出可以安全遮蔽
_READ_ONLY_TOOLS = frozenset({
    "read_file", "glob_tool", "glob", "grep_tool", "grep",
    "grep_search", "web_fetch", "web_search",
})

# 小体积阈值 — 低于此值不做遮蔽
_MIN_CONTENT_CHARS = 500

# ── 行号前缀剥离 ──────────────────────────────────────────
# read_file 输出格式为 "行号| 内容"，如 "7| def get_users():"
# 遮蔽前需剥离行号前缀，否则 startswith 检查全部失效
_LINE_NUM_PREFIX = re.compile(r'^\s*\d+\|\s')


def _strip_line_number(line: str) -> str:
    """剥离 read_file 输出的行号前缀，返回纯代码行。

    "7| def get_users():" → "def get_users():"
    "  123|   x = 1" → "  x = 1"
    "no prefix here" → "no prefix here"
    """
    return _LINE_NUM_PREFIX.sub('', line)


# ── 多语言结构行识别 ──────────────────────────────────────

# 各语言关键字前缀定义
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

# 合并所有关键字，按首字符分组构建查找表
# O(1) 首字符过滤 + O(k) 前缀匹配，k 为同首字符的关键字数
_PREFIX_LOOKUP: dict[str, tuple[str, ...]] = {}
for _kw in set(_PY_KEYWORDS + _JS_KEYWORDS + _GO_KEYWORDS + _RS_KEYWORDS + _JVM_KEYWORDS):
    if _kw:
        _ch = _kw[0]
        if _kw not in _PREFIX_LOOKUP.get(_ch, ()):
            _PREFIX_LOOKUP[_ch] = _PREFIX_LOOKUP.get(_ch, ()) + (_kw,)


def _is_structural_line(code_line: str) -> bool:
    """判断一行代码是否为结构行（定义/声明/注释），应保留。

    结构行 = import/定义/声明/注释/装饰器/文档标记
    非结构行 = 函数体/方法体/赋值/表达式 → 替换为省略号

    支持语言：Python, JavaScript/TypeScript, Go, Rust, Java, Kotlin
    """
    if not code_line:
        return True  # 空行保留

    # 检查原始行和缩进剥离后的行（支持嵌套定义）
    for text in (code_line, code_line.lstrip()):
        if not text:
            continue
        first = text[0]
        candidates = _PREFIX_LOOKUP.get(first)
        if candidates:
            for prefix in candidates:
                if text.startswith(prefix):
                    return True

    # 装饰器/注解（@decorator, @Override, @Transactional）
    if code_line.lstrip().startswith("@"):
        return True

    # Markdown/文档结构
    if code_line.startswith(("```", "=== ", "--- ", "# ")):
        return True

    return False


class ObservationMasker:
    """对只读工具的大体积输出做观察遮蔽。

    保留关键结构信息，删除冗余的函数体/文件体内容。
    不生成新文本，只删除/保留原文片段。
    """

    def __init__(
        self,
        min_content_chars: int = _MIN_CONTENT_CHARS,
        max_file_lines: int = 60,
        max_grep_lines: int = 30,
        max_glob_items: int = 40,
    ):
        """
        Args:
            min_content_chars: 低于此字符数不做遮蔽
            max_file_lines: read_file 遮蔽后最大保留行数
            max_grep_lines: grep 遮蔽后最大保留行数
            max_glob_items: glob 遮蔽后最大保留项数
        """
        self.min_content_chars = min_content_chars
        self.max_file_lines = max_file_lines
        self.max_grep_lines = max_grep_lines
        self.max_glob_items = max_glob_items

    def mask(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """对消息列表中的只读工具输出做观察遮蔽。

        Args:
            messages: 消息列表

        Returns:
            遮蔽后的消息列表（新列表，不修改原始消息）
        """
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
        """根据工具类型选择遮蔽策略。"""
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
        """文件内容遮蔽：保留结构行，函数体用省略号替代。

        read_file 输出格式为 "行号| 内容"，需先剥离行号前缀再判断。
        保留规则：import/定义/声明/注释/装饰器/文档标记/空行
        删除规则：函数体/方法体/赋值/表达式 → 替换为 "  ..."
        """
        lines = content.split("\n")
        kept_lines: list[str] = []
        output_lines = 0
        prev_was_ellipsis = False

        for line_idx, line in enumerate(lines):
            if output_lines >= self.max_file_lines:
                # 接近上限，只保留最后几行（可能含错误信息）
                remaining = lines[line_idx:]
                # 尝试保留末尾的错误/异常信息
                tail_lines = self._extract_tail_errors(remaining)
                if tail_lines:
                    kept_lines.append("  ...")
                    kept_lines.extend(tail_lines)
                else:
                    kept_lines.append(f"  ... (共 {len(lines)} 行，已省略 {len(lines) - output_lines} 行)")
                break

            stripped = line.strip()
            # 剥离行号前缀（read_file 输出格式："123| code"）
            code_line = _strip_line_number(stripped)

            if _is_structural_line(code_line):
                # 连续空行只保留一个
                if not code_line and prev_was_ellipsis:
                    continue
                kept_lines.append(line)
                prev_was_ellipsis = False
            else:
                if not prev_was_ellipsis:
                    kept_lines.append("  ...")
                    prev_was_ellipsis = True

            output_lines = len([l for l in kept_lines if l.strip()])

        result = "\n".join(kept_lines)
        return result

    def _mask_grep_output(self, content: str) -> str:
        """grep 输出遮蔽：保留前几行和后几行，中间省略。"""
        lines = content.split("\n")
        if len(lines) <= self.max_grep_lines:
            return content

        head = lines[:self.max_grep_lines // 2]
        tail = lines[-self.max_grep_lines // 4:]

        return "\n".join(head) + f"\n  ... (省略 {len(lines) - len(head) - len(tail)} 行) ...\n" + "\n".join(tail)

    def _mask_glob_output(self, content: str) -> str:
        """glob 输出遮蔽：保留前几项，统计总数。"""
        lines = content.split("\n")
        if len(lines) <= self.max_glob_items:
            return content

        head = lines[:self.max_glob_items]
        remaining = len(lines) - len(head)

        return "\n".join(head) + f"  ... (共 {len(lines)} 个文件，省略 {remaining} 个)"

    def _mask_web_output(self, content: str) -> str:
        """web 输出遮蔽：保留前 800 字符。"""
        if len(content) <= 800:
            return content
        return content[:800] + "\n... (内容已截断)"

    def _extract_tail_errors(self, remaining_lines: list[str]) -> list[str]:
        """从剩余行中提取错误/异常信息。"""
        error_lines: list[str] = []
        error_keywords = ("error", "exception", "traceback", "failed", "错误", "异常")
        for idx, line in enumerate(remaining_lines):
            if any(kw in line.lower() for kw in error_keywords):
                # 取错误行及其前后 2 行
                start = max(0, idx - 2)
                end = min(len(remaining_lines), idx + 3)
                error_lines = remaining_lines[start:end]
                break
        return error_lines
