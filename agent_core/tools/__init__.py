"""agent_core.tools: 工具层统一 re-export。

内部按职责拆分为独立模块
"""

# ContextVar + set/clear
from .context_vars import (
    _ctx_workspace,
    _ctx_skills_loader,
    _ctx_todo_store,
    _ctx_sub_reg,
    _ctx_llm_ref,
    _ctx_user_id,
    _ctx_ticket_id,
    _IGNORE_DIRS,
    set_workspace,
    set_skills_loader,
    set_todo_store,
    set_subagent_deps,
    set_user_id,
    set_ticket_id,
    clear_context,
)

# Workspace 路径构建
from .workspace import _build_workspace, _resolve, get_workspace_path

# 工具函数
from .file_tools import read_file, write_file, edit_file
from .shell_tools import run_command
from .web_tools import web_fetch
from .skill_tools import load_skill
from .todo_tools import update_todos
from .search_tools import glob_tool, grep_tool

__all__ = [
    # ContextVar
    "_ctx_workspace", "_ctx_skills_loader", "_ctx_todo_store",
    "_ctx_sub_reg", "_ctx_llm_ref", "_ctx_user_id", "_ctx_ticket_id",
    "_IGNORE_DIRS",
    # set/clear
    "set_workspace", "set_skills_loader", "set_todo_store",
    "set_subagent_deps", "set_user_id", "set_ticket_id", "clear_context",
    # Workspace
    "_build_workspace", "_resolve", "get_workspace_path",
    # 工具
    "read_file", "write_file", "edit_file",
    "run_command", "web_fetch",
    "load_skill", "update_todos",
    "glob_tool", "grep_tool",
]
