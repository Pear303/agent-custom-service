"""Shell 命令工具：run_command。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

from langchain_core.tools import tool

from .context_vars import _ctx_workspace

_logger = logging.getLogger(__name__)


@tool
def run_command(command: str) -> str:
    """在终端执行一条 shell 命令并返回输出。工作目录自动设为用户项目目录。
    Args:
        command: 要执行的 shell 命令字符串
    """
    workspace = _ctx_workspace.get()
    if workspace is None:
        _logger.critical(
            "run_command: _ctx_workspace is None — 命令将在 CWD 执行！command=%s",
            command[:200],
        )
    cwd = str(workspace) if workspace else None

    # D17: 检测命令中的绝对路径，防止 LLM 使用错误的工作区路径
    if workspace and "data" in command and "users" in command:
        import re as _re
        _abs_pattern = _re.compile(r'[A-Za-z]:\\[^\s"]*data[\\/]users[\\/]')
        if _abs_pattern.search(command):
            return (
                f"Error: 命令中包含绝对路径，这违反了工作区约束规则。\n"
                f"你的工作目录（CWD）是: {cwd}\n"
                f"请使用相对路径重新执行命令。例如：\n"
                f"  - 错误: `cd /d {cwd} && python test.py`\n"
                f"  - 正确: `python test.py`（run_command 自动在 CWD 下执行）\n"
                f"  - 错误: `copy {cwd}\\src.py {cwd}\\dst.py`\n"
                f"  - 正确: 使用 write_file/edit_file 等专用工具操作文件\n"
            )

    # D11: 拦截无意义的路径探测命令
    _wasteful_commands = {"pwd", "cd", "dir", "ls", "get-location", "echo %cd%"}
    cmd_stripped = command.strip().lower()
    if cmd_stripped in _wasteful_commands:
        return f"[提示] 你的工作目录已设定为: {cwd}\n请直接使用相对路径操作文件，无需探测目录。"

    # D16: 拦截文件操作类 shell 命令
    _file_op_commands = {
        "copy", "xcopy", "move", "rename", "del", "rm", "rmdir",
        "mkdir", "md", "cp", "mv", "touch", "new-item",
    }
    first_word = command.strip().split()[0].strip('"').lower() if command.strip() else ""
    if first_word in _file_op_commands:
        return (
            f"Error: 禁止使用 '{first_word}' 命令操作文件。"
            f"请使用专用工具：write_file（创建/覆盖文件）、edit_file（编辑文件）、"
            f"read_file（读取文件）、glob_tool（查找文件）。"
            f"这些工具会自动处理路径解析和工作区约束。"
        )

    # Windows 下设置子进程使用 UTF-8 输出
    env = os.environ.copy()
    if sys.platform == "win32":
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        command, shell=True, capture_output=True,
        encoding="utf-8", errors="replace",
        cwd=cwd,
        env=env,
    )
    output = result.stdout or result.stderr

    # D12: 截断过长输出
    _MAX_OUTPUT_CHARS = 4000
    if len(output) > _MAX_OUTPUT_CHARS:
        head = output[:_MAX_OUTPUT_CHARS // 2]
        tail = output[-_MAX_OUTPUT_CHARS // 2:]
        output = f"{head}\n\n... [输出已截断，共 {len(output)} 字符] ...\n\n{tail}"

    return output
