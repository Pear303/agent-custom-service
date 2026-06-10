"""ContextVar 定义 + set/clear 函数。

从 agent.lc_tools 提取，供 agent_core.tools 和 agent_by_langgraph 共用。
"""
from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any


# ── 异步安全的上下文存储，每个协程独立 ──────────────────────────

_ctx_workspace: ContextVar[Path | None] = ContextVar("workspace", default=None)
_ctx_skills_loader: ContextVar[Any | None] = ContextVar("skills_loader", default=None)
_ctx_todo_store: ContextVar[Any | None] = ContextVar("todo_store", default=None)
_ctx_sub_reg: ContextVar[Any | None] = ContextVar("sub_reg", default=None)
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
    _ctx_sub_reg.set(registry)
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
    _ctx_sub_reg.set(None)
    _ctx_llm_ref.set(None)
    _ctx_user_id.set(None)
    _ctx_ticket_id.set(None)
