"""三层记忆存储系统：原始历史 / 每日情景记忆 / 长期记忆。

从 agent.memory 迁移，无 import 变更。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage


_UTC8 = timezone(timedelta(hours=8))

_TYPE_TO_JSONL_ROLE: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}

_JSONL_ROLE_TO_MESSAGE_CLS: dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}


class MemoryStore(BaseChatMessageHistory):
    """三层记忆存储管理器。"""

    def __init__(self, memory_dir: Path | None = None, user_file: Path | None = None, user_id: str | None = None):
        if user_id:
            _base = Path(__file__).parent.parent / "data" / "users" / user_id
            self.memory_dir = _base / "memory"
            self.user_file = _base / "USER.md"
        else:
            self.memory_dir = memory_dir
            self.user_file = user_file

        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self._ensure()

    def _ensure(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n此文件常驻上下文，记录核心目标、当前任务与关键事实。\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("")

    def append_history(self, role: str, content: Any, additional_kwargs: dict | None = None) -> None:
        row = {
            "ts": datetime.now(_UTC8).isoformat(timespec="seconds"),
            "role": role,
            "content": content if isinstance(content, str) else _json_safe(content),
        }
        if additional_kwargs:
            row["additional_kwargs"] = additional_kwargs
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            role = _TYPE_TO_JSONL_ROLE.get(msg.type, "unknown")
            extra = getattr(msg, "additional_kwargs", None) or None
            self.append_history(role, msg.content, additional_kwargs=extra)

    def today_episode_path(self) -> Path:
        date = datetime.now(_UTC8).strftime("%Y-%m-%d")
        return self.memory_dir / f"{date}.md"

    def read_today_episode(self) -> str:
        p = self.today_episode_path()
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def append_episode(self, content: str) -> None:
        p = self.today_episode_path()
        existing = p.read_text(encoding="utf-8") if p.exists() else f"# {p.stem} 情景记忆\n"
        new_text = existing.rstrip() + "\n\n" + content.strip() + "\n"
        p.write_text(new_text, encoding="utf-8")

    def read_memory(self) -> str:
        if not self.memory_file.exists():
            return ""
        try:
            return self.memory_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return self.memory_file.read_text(encoding="gbk", errors="ignore")

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content.strip() + "\n", encoding="utf-8")

    def append_compact_marker(self) -> None:
        row = {"ts": datetime.now(_UTC8).isoformat(timespec="seconds"), "type": "compact_event"}
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        self.append_compact_marker()

    def load_unarchived_history(self) -> list:
        if not self.history_file.exists():
            return []
        rows = []
        with self.history_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        last_marker = -1
        for i, row in enumerate(rows):
            if row.get("type") == "compact_event":
                last_marker = i
        return [
            {"role": r["role"], "content": r["content"]}
            for r in rows[last_marker + 1:]
            if "role" in r and "content" in r
        ]

    @property
    def messages(self) -> list[BaseMessage]:
        raw = self.load_unarchived_history()
        result: list[BaseMessage] = []
        for entry in raw:
            role = entry["role"]
            content = entry["content"]
            extra_kwargs = entry.get("additional_kwargs", None)
            message_cls = _JSONL_ROLE_TO_MESSAGE_CLS.get(role)
            if message_cls is not None:
                if extra_kwargs:
                    result.append(message_cls(content=content, additional_kwargs=extra_kwargs))
                else:
                    result.append(message_cls(content=content))
        return result

    def read_user(self) -> str:
        return self.user_file.read_text(encoding="utf-8") if self.user_file.exists() else ""

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content.strip() + "\n", encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: _json_safe(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)
