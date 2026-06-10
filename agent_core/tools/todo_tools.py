"""Todo 更新工具：update_todos。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .context_vars import _ctx_todo_store


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
