"""LangChain 工具集合 —— @tool 函数包装原有的执行逻辑"""
from __future__ import annotations

import difflib
import gzip
import io
import json
import os
import re
import subprocess
import urllib.request
from contextvars import ContextVar
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from .subagent_parallel import ParallelAgentExecutor

# 异步安全的上下文存储，每个协程独立
_ctx_workspace: ContextVar[Path | None] = ContextVar("workspace", default=None)
_ctx_skills_loader: ContextVar[Any | None] = ContextVar("skills_loader", default=None)
_ctx_todo_store: ContextVar[Any | None] = ContextVar("todo_store", default=None)
_ctx_subagent_registry: ContextVar[Any | None] = ContextVar("subagent_registry", default=None)
_ctx_llm_ref: ContextVar[Any | None] = ContextVar("llm_ref", default=None)
_ctx_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_ctx_ticket_id: ContextVar[str | None] = ContextVar("ticket_id", default=None)

_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".env",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
}


def set_workspace(path: Path) -> None:
    """设置当前协程的上下文的工作区根目录"""
    _ctx_workspace.set(path)


def set_skills_loader(loader: Any) -> None:
    """设置当前上下文的技能加载器"""
    _ctx_skills_loader.set(loader)


def set_todo_store(store: Any) -> None:
    """设置当前上下文的 Todo 存储实例"""
    _ctx_todo_store.set(store)


def set_subagent_deps(llm, registry) -> None:
    """设置当前上下文的子 Agent 依赖"""
    _ctx_subagent_registry.set(registry)
    _ctx_llm_ref.set(llm)


def set_user_id(user_id: str) -> None:
    """设置当前上下文的用户 ID"""
    _ctx_user_id.set(user_id)


def set_ticket_id(ticket_id: str) -> None:
    """设置当前上下文的工单 ID"""
    _ctx_ticket_id.set(ticket_id)


def clear_context() -> None:
    """清理当前上下文的所有 ContextVar 值。
    
    在 agent 使用完毕后调用，防止 ContextVar 泄漏到后续请求。
    """
    _ctx_workspace.set(None)
    _ctx_skills_loader.set(None)
    _ctx_todo_store.set(None)
    _ctx_subagent_registry.set(None)
    _ctx_llm_ref.set(None)
    _ctx_user_id.set(None)
    _ctx_ticket_id.set(None)


def _build_workspace(root: Path, user_id: str | None, ticket_id: str | None) -> Path:
    """构建完整的工作目录路径：{root}/data/users/{user_id}/{ticket_id}"""
    if user_id and ticket_id:
        ws = root / "data" / "users" / user_id / ticket_id
    elif user_id:
        ws = root / "data" / "users" / user_id
    else:
        ws = root / "data" / "users" / "_anonymous"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _resolve(path: str) -> Path:
    """解析路径，所有路径均约束在 workspace 内，防止文件逃逸到根目录。
    
    LLM 可能传入各种格式的路径：
    - 相对路径：src/app.js → workspace/src/app.js
    - 绝对路径：f:/XiangMu/.../projects/myapp/app.js → 先剥离项目根，再拼到 workspace
    - 带项目目录的路径：projects/myapp/app.js → 拼到 workspace/projects/myapp/app.js
    
    关键约束：最终路径必须严格在 workspace 目录内。
    """
    p = Path(path).expanduser()
    workspace = _ctx_workspace.get()

    if workspace is not None:
        if p.is_absolute():
            _project_root = Path(__file__).parent.parent
            try:
                p = p.relative_to(_project_root)
            except ValueError:
                p = Path(p.name)

        resolved = (workspace / p).resolve()

        # 安全检查：确保解析后的路径仍在 workspace 内
        # 防止 LLM 通过 ../../ 等方式逃逸到 workspace 之外
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            # 路径逃逸，回退到只用文件名
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "路径逃逸检测: %s 解析到 workspace 外 (%s)，回退为文件名 %s",
                path, resolved, p.name
            )
            resolved = workspace / p.name

        return resolved

    import logging as _logging
    _logging.getLogger(__name__).critical(
        "_resolve: _ctx_workspace is None — 文件将解析到 CWD！"
        " path=%s, thread=%s, user_id=%s, ticket_id=%s",
        str(p), __import__("threading").get_ident(),
        _ctx_user_id.get(), _ctx_ticket_id.get(),
    )
    return p.resolve()


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


def _is_binary(raw: bytes) -> bool:
    """检测二进制文件：检查空字节和非文本字符比例"""
    if b"\x00" in raw:
        return True
    sample = raw[:4096]
    if not sample:
        return False
    non_text = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return (non_text / len(sample)) > 0.2


def _match_glob(rel_path: str, name: str, pattern: str) -> bool:
    """匹配 glob 模式：支持通配符和路径匹配"""
    import fnmatch
    normalized = pattern.strip().replace("\\", "/")
    if not normalized:
        return False
    if "/" in normalized or normalized.startswith("**"):
        from pathlib import PurePosixPath
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
    import fnmatch
    if not file_type:
        return True
    lowered = file_type.strip().lower()
    if not lowered:
        return True
    patterns = _TYPE_GLOB_MAP.get(lowered, (f"*.{lowered}",))
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


class _TextExtractor(HTMLParser):
    """HTML 文本提取器：去除脚本和样式标签"""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _display_path(target: Path, root: Path) -> str:
    """显示相对路径：优先相对于工作区"""
    if _workspace:
        try:
            return target.relative_to(_workspace).as_posix()
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

@tool
def write_file(path: str, content: str) -> str:
    """写入文件（覆盖已有内容）。部分编辑请用 edit_file。
    Args:
        path: 文件路径（相对于工作区）
        content: 要写入的文件内容
    """
    try:
        fp = _resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        fp.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {fp}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


# ═══════════════════════════════════════════════════════════════════
#  edit_file
# ═══════════════════════════════════════════════════════════════════

@tool
def edit_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """替换文件中的文本。容忍缩进差异和引号风格差异。若 old_text 匹配多处，需提供更多上下文或设 replace_all=true。
    Args:
        path: 文件路径（相对于工作区）
        old_text: 要被替换的原文本
        new_text: 替换后的新文本
        replace_all: 是否替换所有匹配项（默认 False，只替换第一处）
    """
    try:
        fp = _resolve(path)
        if not fp.exists():
            if old_text == "":
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(new_text, encoding="utf-8")
                return f"Successfully created {fp}"
            return f"Error: File not found: {path}"
        raw = fp.read_bytes()
        uses_crlf = b"\r\n" in raw  # 检测换行符风格
        content = raw.decode("utf-8").replace("\r\n", "\n")
        norm_old = old_text.replace("\r\n", "\n")
        if old_text == "":
            if content.strip():
                return f"Error: Cannot create file — {path} already exists and is not empty."
            fp.write_text(new_text, encoding="utf-8")
            return f"Successfully edited {fp}"
        matches = _find_matches(content, norm_old)  # 查找匹配位置
        if not matches:
            # 未找到匹配，返回最相似位置的差异对比
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
                return f"Error: old_text not found in {path}.\nBest match ({ratio:.0%}) at line {start + 1}:\n{diff}"
            return f"Error: old_text not found in {path}."
        if len(matches) > 1 and not replace_all:
            lines = [content.count('\n', 0, s) + 1 for s, _ in matches]
            preview = ", ".join(f"line {n}" for n in lines[:3])
            return f"Warning: old_text appears {len(matches)} times at {preview}. Set replace_all=true or add more context."
        norm_new = new_text.replace("\r\n", "\n")
        selected = matches if replace_all else matches[:1]  # 选择要替换的匹配项
        new_content = content
        for start, end in reversed(selected):  # 从后往前替换，避免位置偏移
            actual = new_content[start:end]
            replacement = norm_new
            if replacement == "" and not actual.endswith("\n") and new_content[end:end + 1] == "\n":
                end += 1
            new_content = new_content[:start] + replacement + new_content[end:]
        if uses_crlf:
            new_content = new_content.replace("\n", "\r\n")  # 恢复原始换行符风格
        fp.write_bytes(new_content.encode("utf-8"))
        return f"Successfully edited {fp}"
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error editing file: {e}"


# ═══════════════════════════════════════════════════════════════════
#  run_command
# ═══════════════════════════════════════════════════════════════════

@tool
def run_command(command: str) -> str:
    """在终端执行一条 shell 命令并返回输出。工作目录自动设为用户项目目录。
    Args:
        command: 要执行的 shell 命令字符串
    """
    workspace = _ctx_workspace.get()
    if workspace is None:
        import logging as _logging
        _logging.getLogger(__name__).critical(
            "run_command: _ctx_workspace is None — 命令将在 CWD 执行！command=%s",
            command[:200],
        )
    cwd = str(workspace) if workspace else None

    result = subprocess.run(
        command, shell=True, capture_output=True,
        encoding="utf-8", errors="replace",
        cwd=cwd,
    )
    return result.stdout or result.stderr


# ═══════════════════════════════════════════════════════════════════
#  web_fetch
# ═══════════════════════════════════════════════════════════════════

def _fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    """抓取网页内容并提取文本"""
    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # 解压 gzip/deflate
    content_encoding = resp.headers.get("Content-Encoding", "")
    if content_encoding == "gzip":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception:
            pass
    elif content_encoding == "deflate":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception:
            pass

    # 检测字符编码：优先从 Content-Type 头获取
    charset = "utf-8"
    content_type = resp.headers.get("Content-Type", "")
    m = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    if m:
        charset = m.group(1).lower()
    # 部分网站声明 GBK/GB2312 但实际是 GB18030
    if charset in ("gbk", "gb2312", "gb18030"):
        charset = "gb18030"

    try:
        text = raw_bytes.decode(charset, errors="replace")
    except (LookupError, ValueError):
        # 未知编码时回退 UTF-8
        text = raw_bytes.decode("utf-8", errors="replace")

    if extract_mode == "text":
        parser = _TextExtractor()
        parser.feed(text)
        text = parser.get_text()
    return text[:max_chars]


@tool
def web_fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    """获取指定 URL 的网页内容。extract_mode: text（纯文本，默认）或 raw（原始 HTML）
    Args:
        url: 要抓取的网页 URL
        extract_mode: 提取模式，"text" 提取纯文本（去除脚本和样式），"raw" 返回原始 HTML（默认 "text"）
        max_chars: 最大返回字符数（默认 8000）
    """
    return _fetch(url, extract_mode, max_chars)


# ═══════════════════════════════════════════════════════════════════
#  glob
# ═══════════════════════════════════════════════════════════════════

def _paginate(items: list, limit: int | None, offset: int) -> tuple[list, bool]:
    """分页处理：返回切片和是否截断标志"""
    if limit is None:
        return items[offset:], False
    sliced = items[offset: offset + limit]
    truncated = len(items) > offset + limit
    return sliced, truncated


@tool
def glob_tool(
    pattern: str,
    path: str = ".",
    head_limit: Optional[int] = 250,
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
        matches.sort(key=lambda item: (-item[1], item[0]))  # 按修改时间降序排序
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
#  grep
# ═══════════════════════════════════════════════════════════════════

_DEFAULT_HEAD_LIMIT = 250
_MAX_RESULT_CHARS = 128_000
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
    head_limit: Optional[int] = 250,
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


# ═══════════════════════════════════════════════════════════════════
#  load_skill
# ═══════════════════════════════════════════════════════════════════

@tool
def load_skill(name: str) -> str:
    """加载指定名称的技能（skill）。技能提供专题知识和操作指南。
    Args:
        name: 技能名称（如 "pdf", "github", "weather" 等）
    """
    skills_loader = _ctx_skills_loader.get()
    if skills_loader is None:
        return "Error: Skills loader not initialized"
    return skills_loader.get_content(name)


# ═══════════════════════════════════════════════════════════════════
#  update_todos
# ═══════════════════════════════════════════════════════════════════

@tool
def update_todos(todos: str) -> str:
    """更新 Todo 列表。传入 JSON 格式的 todo 数组，或传 'list' 列出当前 todo。
    Args:
        todos: JSON 格式的 todo 数组字符串，或字符串 "list" 用于查看当前列表
    
    每个 todo 项支持以下字段：
    - content / task / title / name / description: 任务描述（必填）
    - status: "pending" | "in_progress" | "completed"（可选，默认 pending）
    
    也可直接传入字符串数组（如 ["步骤1", "步骤2"]），会自动转为待办项。
    """
    todo_store = _ctx_todo_store.get()
    if todo_store is None:
        return "Error: Todo store not initialized"
    if todos.strip() == "list":
        return todo_store.render()
    try:
        data = json.loads(todos)
        return todo_store.update(data)
    except json.JSONDecodeError as e:
        return f"Error: invalid JSON: {e}"


# ═══════════════════════════════════════════════════════════════════
#  dispatch_subagent
# ═══════════════════════════════════════════════════════════════════

@tool
def dispatch_subagent(agent_type: str, task: str) -> str:
    """派遣子代理独立处理任务。子代理有自己独立的上下文，办完只回传文字总结。
    agent_type 可用: quick_helper, web_researcher, doc_analyzer, engine_executor, validator, skill_manager, document_processor, system_maintainer
    Args:
        agent_type: 子代理类型（quick_helper/web_researcher/doc_analyzer/engine_executor/validator/skill_manager/document_processor/system_maintainer）
        task: 要委派给子代理的具体任务描述
    """
    subagent_registry = _ctx_subagent_registry.get()
    llm_ref = _ctx_llm_ref.get()
    
    if subagent_registry is None:
        return "Error: Subagent registry not initialized"
    if llm_ref is None:
        return "Error: LLM not initialized"

    spec = subagent_registry.get(agent_type)  # 查询子代理规格
    if spec is None:
        available = ", ".join(subagent_registry.names())
        return f"Error: unknown subagent '{agent_type}'. Available: {available}"

    # 子代理的工具从白名单中筛选
    tools = [
        _SUBAGENT_TOOL_MAP[name]
        for name in spec.tool_names
        if name in _SUBAGENT_TOOL_MAP
    ]
    if not tools:
        return f"Error: no tools available for subagent '{agent_type}'"

    # 子代理有自己独立的 prompt 和 executor
    prompt = ChatPromptTemplate.from_messages([
        ("system", spec.system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm_ref, tools, prompt)
    executor = ParallelAgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=spec.max_turns,
        handle_parsing_errors=True,
        verbose=False,
    )

    print(f"\n[派遣子代理 · {agent_type}]: {task[:80]}")

    try:
        result = executor.invoke({
            "input": task,
            "chat_history": [],
        })
        final = result["output"]  # ← 只回传总结，子代理内部历史不暴露
    except Exception as exc:
        return f"Error: subagent '{agent_type}' raised: {exc}"

    print(f"[子代理汇报]: {final[:200]}")
    return final


# 子代理工具映射表。不包含 dispatch_subagent（防递归）和 update_todos（防改主 todolist）
_SUBAGENT_TOOL_MAP = {
    "run_command": run_command,
    "web_fetch": web_fetch,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "glob": glob_tool,
    "grep": grep_tool,
    "load_skill": load_skill,
    # dispatch_subagent 不在其中，防止子代理递归派遣
}
