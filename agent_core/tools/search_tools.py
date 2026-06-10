"""文件搜索工具：glob_tool, grep_tool。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Optional

from langchain_core.tools import tool

from .context_vars import _ctx_workspace, _IGNORE_DIRS
from .workspace import _resolve


def _is_binary(raw: bytes) -> bool:
    """检测二进制文件"""
    if b"\x00" in raw:
        return True
    sample = raw[:4096]
    if not sample:
        return False
    non_text = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return (non_text / len(sample)) > 0.2


def _match_glob(rel_path: str, name: str, pattern: str) -> bool:
    """匹配 glob 模式"""
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        return False
    if "/" in normalized or normalized.startswith("**"):
        return PurePosixPath(rel_path).match(normalized)
    return fnmatch.fnmatch(name, normalized)


_TYPE_GLOB_MAP = {
    "py": ("*.py", "*.pyi"), "python": ("*.py", "*.pyi"),
    "js": ("*.js", "*.jsx", "*.mjs", "*.cjs"),
    "ts": ("*.ts", "*.tsx", "*.mts", "*.cts"),
    "tsx": ("*.tsx",), "jsx": ("*.jsx",), "json": ("*.json",),
    "md": ("*.md", "*.mdx"), "markdown": ("*.md", "*.mdx"),
    "go": ("*.go",), "rs": ("*.rs",), "rust": ("*.rs",),
    "java": ("*.java",), "sh": ("*.sh", "*.bash"),
    "yaml": ("*.yaml", "*.yml"), "yml": ("*.yaml", "*.yml"),
    "toml": ("*.toml",), "sql": ("*.sql",),
    "html": ("*.html", "*.htm"), "css": ("*.css", "*.scss", "*.sass"),
}


def _matches_type(name: str, file_type: str | None) -> bool:
    """检查文件名是否匹配指定类型"""
    if not file_type:
        return True
    lowered = file_type.strip().lower()
    if not lowered:
        return True
    patterns = _TYPE_GLOB_MAP.get(lowered, (f"*.{lowered}",))
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


def _display_path(target: Path, root: Path) -> str:
    """显示相对路径：优先相对于工作区"""
    ws = _ctx_workspace.get()
    if ws:
        try:
            return target.relative_to(ws).as_posix()
        except ValueError:
            pass
    return target.relative_to(root).as_posix()


def _iter_files(root: Path):
    """递归遍历文件，跳过忽略目录"""
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        current = Path(dirpath)
        for filename in sorted(filenames):
            yield current / filename


def _iter_entries(root: Path, *, include_files: bool, include_dirs: bool):
    """递归遍历文件或目录条目"""
    if root.is_file():
        if include_files:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        current = Path(dirpath)
        if include_dirs:
            for dirname in dirnames:
                yield current / dirname
        if include_files:
            for filename in sorted(filenames):
                yield current / filename


def _paginate(items: list, limit: int | None, offset: int) -> tuple[list, bool]:
    """分页处理"""
    if limit is None:
        return items[offset:], False
    sliced = items[offset: offset + limit]
    truncated = len(items) > offset + limit
    return sliced, truncated


# ═══════════════════════════════════════════════════════════════════
#  glob_tool
# ═══════════════════════════════════════════════════════════════════

@tool
def glob_tool(
    pattern: str,
    path: str = ".",
    head_limit: Optional[int] = 200,
    offset: int = 0,
    entry_type: str = "files",
) -> str:
    """查找匹配 glob 模式的文件。结果按修改时间排序（最新在前）。跳过 .git 等噪音目录。
    Args:
        pattern: glob 匹配模式（如 "*.py", "**/*.md"）
        path: 搜索根目录（相对于工作区，默认 "."）
        head_limit: 返回结果数量限制（默认 250，设为 0 表示无限制）
        offset: 分页偏移量（默认 0）
        entry_type: 条目类型，"files"（仅文件）、"dirs"（仅目录）、"both"（文件和目录，默认 "files"）
    """
    try:
        root = _resolve(path or ".")
        if not root.exists():
            return f"Error: Path not found: {path}"
        if not root.is_dir():
            return f"Error: Not a directory: {path}"
        limit = None if head_limit and head_limit == 0 else head_limit
        include_files = entry_type in {"files", "both"}
        include_dirs = entry_type in {"dirs", "both"}
        matches: list[tuple[str, float]] = []
        for entry in _iter_entries(root, include_files=include_files, include_dirs=include_dirs):
            rel_path = entry.relative_to(root).as_posix()
            if _match_glob(rel_path, entry.name, pattern):
                display = _display_path(entry, root)
                if entry.is_dir():
                    display += "/"
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                matches.append((display, mtime))
        if not matches:
            return f"No paths matched pattern '{pattern}' in {path}"
        matches.sort(key=lambda item: (-item[1], item[0]))
        ordered = [name for name, _ in matches]
        paged, truncated = _paginate(ordered, limit, offset)
        result = "\n".join(paged)
        if truncated:
            result += f"\n\n(pagination: limit={limit}, offset={offset})"
        elif offset > 0:
            result += f"\n\n(pagination: offset={offset})"
        return result
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error finding files: {e}"


# ═══════════════════════════════════════════════════════════════════
#  grep_tool
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_HEAD_LIMIT = 100
_MAX_RESULT_CHARS = 16_000
_MAX_FILE_BYTES = 2_000_000


@tool
def grep_tool(
    pattern: str,
    path: str = ".",
    glob: Optional[str] = None,
    type: Optional[str] = None,
    case_insensitive: bool = False,
    fixed_strings: bool = False,
    output_mode: str = "files_with_matches",
    context_before: int = 0,
    context_after: int = 0,
    head_limit: Optional[int] = 100,
    offset: int = 0,
) -> str:
    """搜索文件内容。output_mode: content/files_with_matches/count。跳过二进制和大文件（>2MB）。
    Args:
        pattern: 搜索模式（正则表达式，除非 fixed_strings=True）
        path: 搜索根目录（相对于工作区，默认 "."）
        glob: 文件名 glob 过滤模式（可选，如 "*.py"）
        type: 文件类型过滤（可选，如 "py", "js", "md" 等）
        case_insensitive: 是否大小写不敏感（默认 False）
        fixed_strings: 是否将 pattern 视为固定字符串而非正则（默认 False）
        output_mode: 输出模式，"content"（显示匹配内容和上下文）、"files_with_matches"（仅显示匹配文件列表）、"count"（显示每个文件的匹配次数，默认 "files_with_matches"）
        context_before: 每个匹配项前显示的行数（默认 0）
        context_after: 每个匹配项后显示的行数（默认 0）
        head_limit: 返回结果数量限制（默认 250，设为 0 表示无限制）
        offset: 分页偏移量（默认 0）
    """
    try:
        target = _resolve(path or ".")
        if not target.exists():
            return f"Error: Path not found: {path}"
        if not (target.is_dir() or target.is_file()):
            return f"Error: Unsupported path: {path}"
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            needle = re.escape(pattern) if fixed_strings else pattern
            regex = re.compile(needle, flags)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        limit = None if head_limit and head_limit == 0 else head_limit
        blocks: list[str] = []
        result_chars = 0
        seen_content_matches = 0
        truncated = False
        size_truncated = False
        skipped_binary = 0
        skipped_large = 0
        matching_files: list[str] = []
        counts: dict[str, int] = {}
        file_mtimes: dict[str, float] = {}
        root = target if target.is_dir() else target.parent

        for file_path in _iter_files(target):
            rel_path = file_path.relative_to(root).as_posix()
            if glob and not _match_glob(rel_path, file_path.name, glob):
                continue
            if not _matches_type(file_path.name, type):
                continue
            raw = file_path.read_bytes()
            if len(raw) > _MAX_FILE_BYTES:
                skipped_large += 1
                continue
            if _is_binary(raw):
                skipped_binary += 1
                continue
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped_binary += 1
                continue
            lines = content.splitlines()
            display_path = _display_path(file_path, root)
            file_had_match = False
            for idx, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                file_had_match = True
                if output_mode == "count":
                    counts[display_path] = counts.get(display_path, 0) + 1
                    continue
                if output_mode == "files_with_matches":
                    if display_path not in matching_files:
                        matching_files.append(display_path)
                        file_mtimes[display_path] = mtime
                    break
                seen_content_matches += 1
                if seen_content_matches <= offset:
                    continue
                if limit is not None and len(blocks) >= limit:
                    truncated = True
                    break
                start_line = max(1, idx - context_before)
                end_line = min(len(lines), idx + context_after)
                block_lines = [f"{display_path}:{idx}"]
                for line_no in range(start_line, end_line + 1):
                    marker = ">" if line_no == idx else " "
                    block_lines.append(f"{marker} {line_no}| {lines[line_no - 1]}")
                block = "\n".join(block_lines)
                extra_sep = 2 if blocks else 0
                if result_chars + extra_sep + len(block) > _MAX_RESULT_CHARS:
                    size_truncated = True
                    break
                blocks.append(block)
                result_chars += extra_sep + len(block)
            if output_mode == "count" and file_had_match:
                if display_path not in matching_files:
                    matching_files.append(display_path)
                    file_mtimes[display_path] = mtime
            if output_mode in {"count", "files_with_matches"} and file_had_match:
                continue
            if truncated or size_truncated:
                break

        if output_mode == "files_with_matches":
            if not matching_files:
                result = f"No matches found for pattern '{pattern}' in {path}"
            else:
                ordered_files = sorted(
                    matching_files,
                    key=lambda name: (-file_mtimes.get(name, 0.0), name),
                )
                paged, truncated = _paginate(ordered_files, limit, offset)
                result = "\n".join(paged)
        elif output_mode == "count":
            if not counts:
                result = f"No matches found for pattern '{pattern}' in {path}"
            else:
                ordered_files = sorted(
                    matching_files,
                    key=lambda name: (-file_mtimes.get(name, 0.0), name),
                )
                ordered, truncated = _paginate(ordered_files, limit, offset)
                lines = [f"{name}: {counts[name]}" for name in ordered]
                result = "\n".join(lines)
        else:
            if not blocks:
                result = f"No matches found for pattern '{pattern}' in {path}"
            else:
                result = "\n\n".join(blocks)

        notes: list[str] = []
        if output_mode == "content" and truncated:
            notes.append(f"(pagination: limit={limit}, offset={offset})")
        elif output_mode == "content" and size_truncated:
            notes.append("(output truncated due to size)")
        elif truncated and output_mode in {"count", "files_with_matches"}:
            notes.append(f"(pagination: limit={limit}, offset={offset})")
        elif output_mode in {"count", "files_with_matches"} and offset > 0:
            notes.append(f"(pagination: offset={offset})")
        elif output_mode == "content" and offset > 0 and blocks:
            notes.append(f"(pagination: offset={offset})")
        if skipped_binary:
            notes.append(f"(skipped {skipped_binary} binary/unreadable files)")
        if skipped_large:
            notes.append(f"(skipped {skipped_large} large files)")
        if output_mode == "count" and counts:
            notes.append(f"(total matches: {sum(counts.values())} in {len(counts)} files)")
        if notes:
            result += "\n\n" + "\n".join(notes)
        return result
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error searching files: {e}"
