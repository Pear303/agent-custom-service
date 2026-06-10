"""文件操作工具：read_file, write_file, edit_file。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from .context_vars import _ctx_workspace
from .workspace import _resolve


# ── 内部帮助函数 ──────────────────────────────────────────────────

_QUOTE_TABLE = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
})


def _normalize_quotes(s: str) -> str:
    """标准化引号：将弯引号转换为直引号"""
    return s.translate(_QUOTE_TABLE)


def _find_exact(content: str, old: str) -> list[tuple[int, int]]:
    """精确匹配文本，返回所有匹配的 (起始位置, 结束位置)"""
    matches, start = [], 0
    while True:
        idx = content.find(old, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(old)))
        start = idx + max(1, len(old))
    return matches


def _find_trimmed(content: str, old: str, normalize: bool = False) -> list[tuple[int, int]]:
    """按行匹配文本，容忍缩进差异和引号风格"""
    old_lines = old.splitlines()
    if not old_lines:
        return []
    content_lines = content.splitlines(keepends=True)
    if len(content_lines) < len(old_lines):
        return []
    offsets, pos = [], 0
    for line in content_lines:
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)
    prep = (lambda s: _normalize_quotes(s.strip())) if normalize else str.strip
    stripped_old = [prep(l) for l in old_lines]
    w = len(old_lines)
    matches = []
    for i in range(len(content_lines) - w + 1):
        window = [prep(content_lines[i + j].rstrip("\n\r")) for j in range(w)]
        if window != stripped_old:
            continue
        start = offsets[i]
        end = offsets[i + w]
        if content_lines[i + w - 1].endswith("\n"):
            end -= 1
        matches.append((start, end))
    return matches


def _edit_context_preview(content: str, replace_start: int, new_text: str, context_lines: int = 3) -> str:
    """生成 edit_file 修改后的上下文预览，帮助 LLM 确认修改生效。"""
    lines = content.splitlines(keepends=True)
    char_count = 0
    target_line = 0
    for i, line in enumerate(lines):
        if char_count + len(line) > replace_start:
            target_line = i
            break
        char_count += len(line)
    else:
        target_line = len(lines) - 1 if lines else 0

    new_line_count = new_text.count("\n") + (1 if new_text and not new_text.endswith("\n") else 0)
    end_line = min(target_line + max(new_line_count, 1), len(lines))

    start = max(0, target_line - context_lines)
    end = min(len(lines), end_line + context_lines)

    preview_lines = []
    for i in range(start, end):
        marker = ">>>" if target_line <= i < end_line else "   "
        line_content = lines[i].rstrip("\r\n")
        preview_lines.append(f"{marker} {i + 1:4d} | {line_content}")

    return "修改后上下文:\n" + "\n".join(preview_lines)


def _get_actual_snippet(content: str, start_line: int, line_count: int, context_lines: int = 3) -> str:
    """获取文件指定行区域的实际内容，用于 edit_file 失败时帮助 Agent 定位。"""
    lines = content.splitlines(keepends=True)
    s = max(0, start_line - context_lines)
    e = min(len(lines), start_line + line_count + context_lines)
    result = []
    for i in range(s, e):
        marker = ">>>" if start_line <= i < start_line + line_count else "   "
        line_content = lines[i].rstrip("\r\n") if i < len(lines) else ""
        result.append(f"{marker} {i + 1:4d} | {line_content}")
    return "\n".join(result)


def _find_matches(content: str, old: str) -> list[tuple[int, int]]:
    """查找文本匹配：依次尝试精确匹配、修剪匹配、标准化匹配"""
    for finder in (
        lambda: _find_exact(content, old),
        lambda: _find_trimmed(content, old),
        lambda: _find_trimmed(content, old, normalize=True),
    ):
        m = finder()
        if m:
            return m
    return []


def _best_window(old: str, content: str) -> tuple[float, int]:
    """找到与 old_text 最相似的窗口，返回 (相似度, 起始行号)"""
    lines = content.splitlines(keepends=True)
    old_lines = old.splitlines(keepends=True)
    w = max(1, len(old_lines))
    best_ratio, best_start = -1.0, 0
    for i in range(max(1, len(lines) - w + 1)):
        ratio = difflib.SequenceMatcher(None, old_lines, lines[i:i + w]).ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, i
    return best_ratio, best_start


def _display_path(target: Path, root: Path) -> str:
    """显示相对路径：优先相对于工作区"""
    ws = _ctx_workspace.get()
    if ws:
        try:
            return target.relative_to(ws).as_posix()
        except ValueError:
            pass
    return target.relative_to(root).as_posix()


# ═══════════════════════════════════════════════════════════════════
#  read_file
# ═══════════════════════════════════════════════════════════════════

@tool
def read_file(path: str, offset: int = 1, limit: Optional[int] = None) -> str:
    """读取文本文件内容，支持 offset/limit 分页。输出格式：行号|内容。
    Args:
        path: 文件路径（相对于工作区）
        offset: 起始行号，从 1 开始（默认值 1）
        limit: 最多读取行数（默认值 2000）
    """
    _DEFAULT_LIMIT = 2000
    _MAX_CHARS = 128_000
    try:
        fp = _resolve(path)
        if not fp.exists():
            return f"Error: File not found: {path}"
        if not fp.is_file():
            return f"Error: Not a file: {path}"
        try:
            text = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {path}"
        lines = text.splitlines()
        total = len(lines)
        if offset < 1:
            offset = 1
        if offset > total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"
        start = offset - 1
        end = min(start + (limit or _DEFAULT_LIMIT), total)
        numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(lines[start:end])]
        result = "\n".join(numbered)
        if len(result) > _MAX_CHARS:
            trimmed, chars = [], 0
            for line in numbered:
                chars += len(line) + 1
                if chars > _MAX_CHARS:
                    break
                trimmed.append(line)
            end = start + len(trimmed)
            result = "\n".join(trimmed)
        if end < total:
            result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
        else:
            result += f"\n\n(End of file — {total} lines total)"
        return result
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


# ═══════════════════════════════════════════════════════════════════
#  write_file
# ═══════════════════════════════════════════════════════════════════

class WriteFileArgs(BaseModel):
    """write_file 的参数 schema。"""
    path: str = Field(default="", description='文件路径（相对于工作区，如 "index.html" 或 "css/style.css"）。必填，不能省略。')
    content: str = Field(description='要写入的文件内容')


@tool(args_schema=WriteFileArgs)
def write_file(path: str = "", content: str = "") -> str:
    """写入文件（覆盖已有内容）。部分编辑请用 edit_file。
    Args:
        path: 文件路径（相对于工作区，如 \"index.html\" 或 \"css/style.css\"）。必填，不能省略。
        content: 要写入的文件内容
    """
    if not path or not path.strip():
        return (
            "Error: path 参数为空。write_file 必须提供 path 参数，"
            "例如 write_file(path='index.html', content='...')。"
            "请重新调用并指定目标文件路径。"
        )
    _FORBIDDEN_PREFIXES = ("data/", "data\\", "_build", "-p", "node_modules")
    stripped = path.lstrip("/\\")
    if any(stripped.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        return (
            f"Error: 路径 '{path}' 包含系统保留前缀，已被拒绝。"
            f" 请使用相对于项目目录的路径，例如 'index.html'、'css/style.css'。"
        )
    try:
        fp = _resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        display = _display_path(fp, Path(__file__).parent.parent.parent)
        return f"Successfully wrote {len(content)} characters to {display}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


# ═══════════════════════════════════════════════════════════════════
#  edit_file
# ═══════════════════════════════════════════════════════════════════

@tool
def edit_file(
    path: str,
    old_text: str = None,
    new_text: str = None,
    old_string: str = None,
    new_string: str = None,
    replace_all: bool = False,
) -> str:
    """替换文件中的文本。容忍缩进差异和引号风格差异。若 old_text 匹配多处，需提供更多上下文或设 replace_all=true。
    Args:
        path: 文件路径（相对于工作区）
        old_text: 要被替换的原文本（优先使用，也可用 old_string）
        new_text: 替换后的新文本（优先使用，也可用 new_string）
        old_string: old_text 的别名，二者传一个即可
        new_string: new_text 的别名，二者传一个即可
        replace_all: 是否替换所有匹配项（默认 False，只替换第一处）
    """
    old_text = old_text if old_text is not None else old_string
    new_text = new_text if new_text is not None else new_string
    if old_text is None or new_text is None:
        return "Error: 必须提供 old_text 和 new_text（或其别名 old_string/new_string）"
    try:
        fp = _resolve(path)
        if not fp.exists():
            if old_text == "":
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(new_text, encoding="utf-8")
                return f"Successfully created {_display_path(fp, Path(__file__).parent.parent.parent)}"
            return f"Error: File not found: {path}"
        raw = fp.read_bytes()
        uses_crlf = b"\r\n" in raw
        content = raw.decode("utf-8").replace("\r\n", "\n")
        norm_old = old_text.replace("\r\n", "\n")
        if old_text == "":
            if content.strip():
                return f"Error: Cannot create file — {path} already exists and is not empty."
            fp.write_text(new_text, encoding="utf-8")
            return f"Successfully edited {_display_path(fp, Path(__file__).parent.parent.parent)}"
        matches = _find_matches(content, norm_old)
        if not matches:
            ratio, start = _best_window(norm_old, content)
            if ratio > 0.5:
                best_lines = content.splitlines(keepends=True)
                w = max(1, len(norm_old.splitlines()))
                diff = "".join(difflib.unified_diff(
                    norm_old.splitlines(keepends=True),
                    best_lines[start:start + w],
                    fromfile="old_text (provided)",
                    tofile=f"{path} (actual, line {start + 1})",
                ))
                actual_snippet = _get_actual_snippet(content, start, w)
                return f"Error: old_text not found in {path}.\nBest match ({ratio:.0%}) at line {start + 1}:\n{diff}\n\nActual file content around line {start + 1}:\n{actual_snippet}\nTip: Use read_file to get the latest content, then retry with the correct old_text."
            actual_snippet = _get_actual_snippet(content, 0, 20)
            return f"Error: old_text not found in {path}.\nFile beginning:\n{actual_snippet}\nTip: Use read_file to get the latest content, then retry with the correct old_text."
        if len(matches) > 1 and not replace_all:
            lines = [content.count('\n', 0, s) + 1 for s, _ in matches]
            preview = ", ".join(f"line {n}" for n in lines[:3])
            return f"Warning: old_text appears {len(matches)} times at {preview}. Set replace_all=true or add more context."
        norm_new = new_text.replace("\r\n", "\n")
        selected = matches if replace_all else matches[:1]
        new_content = content
        for start, end in reversed(selected):
            actual = new_content[start:end]
            replacement = norm_new
            if replacement == "" and not actual.endswith("\n") and new_content[end:end + 1] == "\n":
                end += 1
            new_content = new_content[:start] + replacement + new_content[end:]
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")
        fp.write_bytes(new_content.encode("utf-8"))
        display = _display_path(fp, Path(__file__).parent.parent.parent)
        context_preview = _edit_context_preview(new_content, selected[0][0], norm_new)
        return f"Successfully edited {display}\n{context_preview}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error editing file: {e}"
