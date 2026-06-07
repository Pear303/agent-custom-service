"""ContextVar 统一管理 —— 快照/恢复的单一真相源。

新增 ContextVar 时只需在此文件的 _CONTEXT_VARS 注册表中添加一行，
所有快照/恢复逻辑自动同步，避免散落在多个文件中遗漏。
"""
from __future__ import annotations

from typing import Any

# ── ContextVar 注册表 ──────────────────────────────────────────
# 新增 ContextVar 时只需在此表追加一行，无需修改其他文件。
# 格式: (key_name, context_var)
# key_name 用于快照 dict 的键名，必须唯一。
_CONTEXT_VARS: list[tuple[str, Any]] = []


def _init_registry():
    """延迟初始化注册表，避免循环导入。"""
    if _CONTEXT_VARS:
        return
    from agent.lc_tools import (
        _ctx_workspace,
        _ctx_skills_loader,
        _ctx_todo_store,
        _ctx_sub_reg,
        _ctx_llm_ref,
        _ctx_user_id,
        _ctx_ticket_id,
    )
    _CONTEXT_VARS.clear()
    _CONTEXT_VARS.extend([
        ("workspace", _ctx_workspace),
        ("skills_loader", _ctx_skills_loader),
        ("todo_store", _ctx_todo_store),
        ("sub_reg", _ctx_sub_reg),
        ("llm_ref", _ctx_llm_ref),
        ("user_id", _ctx_user_id),
        ("ticket_id", _ctx_ticket_id),
    ])


def snapshot() -> dict[str, Any]:
    """捕获当前线程所有已注册 ContextVar 的值快照。"""
    _init_registry()
    return {key: var.get() for key, var in _CONTEXT_VARS}


def restore(snap: dict[str, Any]) -> None:
    """在工作线程内恢复 ContextVar 值。"""
    _init_registry()
    for key, var in _CONTEXT_VARS:
        if key in snap:
            var.set(snap[key])
