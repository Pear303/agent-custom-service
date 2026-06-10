"""待办事项存储：TodoStore。

从 agent.todo 迁移，无 import 变更。
"""
from __future__ import annotations


_VALID_STATUS = ("pending", "in_progress", "completed")
_STATUS_ICON = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_STATUS_LABEL = {"pending": "待办", "in_progress": "进行中", "completed": "已完成"}
_CONTENT_FIELDS = ("content", "task", "title", "name", "description", "事项", "任务")


def _extract_content(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for field in _CONTENT_FIELDS:
            value = item.get(field)
            if value and isinstance(value, str):
                return value.strip()
        for v in item.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _normalize_items(items) -> list:
    if isinstance(items, dict):
        for key in ("todos", "items", "tasks"):
            if key in items and isinstance(items[key], list):
                return items[key]
        content = _extract_content(items)
        if content:
            return [items]
        return []
    if isinstance(items, list):
        normalized = []
        for item in items:
            if isinstance(item, str):
                normalized.append({"content": item.strip()})
            else:
                normalized.append(item)
        return normalized
    return []


def _render(todos: list[dict]) -> str:
    if not todos:
        return "(当前无待办事项)"
    lines = []
    for t in todos:
        icon = _STATUS_ICON.get(t.get("status", "pending"), "[?]")
        lines.append(f"  {icon} {t.get('id')}. {t.get('content', '')}")
    return "\n".join(lines)


class TodoStore:
    """待办事项存储管理器。"""

    def __init__(self, user_id: str | None = None):
        self.user_id = user_id
        self.todos: list[dict] = []
        self._file = None
        if user_id:
            from pathlib import Path
            self._file = Path(__file__).parent.parent / "data" / "users" / user_id / "todos.json"
            self._load()

    def _load(self):
        if self._file and self._file.exists():
            import json
            try:
                self.todos = json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.todos = []

    def _save(self):
        if self._file:
            import json
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(json.dumps(self.todos, ensure_ascii=False, indent=2), encoding="utf-8")

    def update(self, items) -> str:
        items = _normalize_items(items)
        cleaned: list[dict] = []
        for i, t in enumerate(items, start=1):
            content = _extract_content(t)
            if not content:
                continue
            status = t.get("status", "pending")
            if status not in _VALID_STATUS:
                status = "pending"
            cleaned.append({
                "id": t.get("id", i),
                "content": content,
                "status": status,
            })

        dropped = len(items) - len(cleaned)
        in_progress_count = sum(1 for t in cleaned if t["status"] == "in_progress")
        if in_progress_count > 1:
            return "Error: 同一时间只能有一个 in_progress 任务，请重新规划。"

        self.todos = cleaned
        self._save()

        n_completed = sum(1 for t in cleaned if t["status"] == "completed")
        n_pending = sum(1 for t in cleaned if t["status"] == "pending")
        n_in_progress = in_progress_count

        current = ""
        for t in cleaned:
            if t["status"] == "in_progress":
                current = t.get("content", "")
                break

        parts = []
        if n_completed:
            parts.append(f"{n_completed}项已完成")
        if n_in_progress:
            parts.append(f"{n_in_progress}项进行中")
        if n_pending:
            parts.append(f"{n_pending}项待办")
        summary_line = " · ".join(parts) if parts else "0项"

        print(f"\n{'='*40}")
        print(f"  计划进度  |  {summary_line}")
        if current:
            print(f"  当前执行  |  {current}")
        print(f"{'='*40}")
        if cleaned:
            print(_render(self.todos))
        if dropped:
            print(f"  ({dropped} 项因缺少内容被跳过)")
        print()

        summary = (
            f"todos updated: total={len(self.todos)}, completed={n_completed}, "
            f"in_progress={n_in_progress}, pending={n_pending}"
        )
        return summary + "\n\n当前列表：\n" + _render(self.todos)

    def render(self) -> str:
        return _render(self.todos)
