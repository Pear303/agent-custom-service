from __future__ import annotations

import json


def test_update_todos_sets_and_lists(monkeypatch):
    from agent.lc_tools import update_todos
    from agent.todo import TodoStore

    store = TodoStore()
    monkeypatch.setattr("agent.lc_tools._todo_store", store)

    todos = [
        {"id": 1, "content": "Task A", "status": "pending"},
        {"id": 2, "content": "Task B", "status": "in_progress"},
    ]
    result = update_todos.invoke({"todos": str(json.dumps(todos))})
    assert "todos updated" in result
    assert "Task A" in result

    result2 = update_todos.invoke({"todos": "list"})
    assert "Task A" in result2
    assert "Task B" in result2


def test_update_todos_list_empty(monkeypatch):
    from agent.lc_tools import update_todos
    from agent.todo import TodoStore

    store = TodoStore()
    monkeypatch.setattr("agent.lc_tools._todo_store", store)

    result = update_todos.invoke({"todos": "list"})
    assert "待办" in result
