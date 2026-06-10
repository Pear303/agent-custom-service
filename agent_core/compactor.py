"""Compactor: 历史对话压缩器，将旧的历史对话总结为今日片段并更新 MEMORY.md / USER.md。

从 agent.compactor 迁移，更新 import: from .memory import MemoryStore
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .memory import MemoryStore


_UTC8 = timezone(timedelta(hours=8))

_PROMPT_FILE = Path(__file__).parent.parent / "templates" / "agent" / "compact_prompt.md"
_PROMPT_TEMPLATE = _PROMPT_FILE.read_text(encoding="utf-8")


def _extract(tag: str, text: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _messages_to_text(messages: list) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}] {content}")
        elif isinstance(content, list):
            for block in content:
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if btype == "text":
                    text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                    parts.append(f"[{role}] {text}")
                elif btype == "tool_use":
                    name = getattr(block, "name", None) or block.get("name", "")
                    parts.append(f"[{role}:tool_call] {name}")
                elif btype == "tool_result":
                    c = getattr(block, "content", None) or block.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(
                            (getattr(x, "text", None) or (x.get("text") if isinstance(x, dict) else str(x)) or "")
                            for x in c
                        )
                    snippet = str(c)[:300]
                    parts.append(f"[{role}:tool_result] {snippet}")
    return "\n".join(parts)


class Compactor:
    """历史对话压缩器类。"""

    K = 10

    def __init__(self, client, model: str, memory_store: MemoryStore, max_tokens: int = 4000):
        self.client = client
        self.model = model
        self.memory = memory_store
        self.max_tokens = max_tokens

    def compact(self, history: list) -> list:
        if len(history) <= self.K:
            return history
        old = history[: -self.K]
        self._run_compaction(old)
        print(f"[Compacted: {len(old)} turns → today episode + MEMORY updated]")
        return history[-self.K:]

    def compact_startup(self, history: list) -> None:
        if len(history) < 2:
            return
        self._run_compaction(history)
        print(f"[Startup compacted: {len(history)} unarchived turns → MEMORY updated]")

    def compact_store(self, keep: int | None = None) -> None:
        history = self.memory.load_unarchived_history()
        effective_keep = keep if keep is not None else self.K
        if len(history) <= effective_keep:
            return
        old = history[:-effective_keep]
        self._run_compaction(old)
        print(f"[Compacted: {len(old)} turns → today episode + MEMORY updated]")

    def _run_compaction(self, old_messages: list) -> None:
        prompt = _PROMPT_TEMPLATE.format(
            old_conversation=_messages_to_text(old_messages),
            current_memory=self.memory.read_memory() or "(空)",
            current_user=self.memory.read_user() or "(空)",
            today_episode=self.memory.read_today_episode() or "(空)",
            now_hhmm=datetime.now(_UTC8).strftime("%H:%M"),
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        if episode := _extract("episode", text):
            self.memory.append_episode(episode)
        if new_memory := _extract("updated_memory", text):
            self.memory.write_memory(new_memory)
        if new_user := _extract("updated_user", text):
            self.memory.write_user(new_user)
        self.memory.append_compact_marker()
