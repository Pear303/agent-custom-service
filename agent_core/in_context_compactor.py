"""InContextCompactor: 会话内 verbatim 上下文压缩。

从 agent.in_context_compactor 迁移，更新 import: from .context_view import _ensure_tool_call_integrity
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .context_view import _ensure_tool_call_integrity


_KEEP_RECENT_STEPS = 5
_TRUNCATED_CHARS = 200


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)


def _msg_tokens(msg: BaseMessage) -> int:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return _estimate_tokens(content)
    return sum(
        _estimate_tokens(b.get("text", ""))
        for b in content
        if isinstance(b, dict)
    )


def _compute_step_map(messages: list[BaseMessage]) -> dict[int, int]:
    step_map: dict[int, int] = {}
    current_step = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            current_step += 1
        step_map[i] = current_step
    return step_map


class InContextCompactor:
    """会话内 verbatim 上下文压缩器。"""

    def __init__(
        self,
        max_context_tokens: int = 200_000,
        target_ratio: float = 0.6,
        keep_recent_steps: int = _KEEP_RECENT_STEPS,
        truncated_chars: int = _TRUNCATED_CHARS,
    ):
        self.max_context_tokens = max_context_tokens
        self.target_ratio = target_ratio
        self.keep_recent_steps = keep_recent_steps
        self.truncated_chars = truncated_chars

    def compact(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        total_tokens = sum(_msg_tokens(m) for m in messages)
        threshold = self.max_context_tokens * self.target_ratio
        if total_tokens <= threshold:
            return messages
        result = self._truncate_old_tool_messages(messages, total_tokens, threshold)
        return _ensure_tool_call_integrity(result)

    def _truncate_old_tool_messages(
        self,
        messages: list[BaseMessage],
        current_tokens: int,
        target_tokens: int,
    ) -> list[BaseMessage]:
        step_map = _compute_step_map(messages)
        max_step = max(step_map.values()) if step_map else 0
        recent_step_cutoff = max_step - self.keep_recent_steps

        truncatable: list[tuple[int, int]] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, ToolMessage):
                continue
            step = step_map.get(i, 0)
            if step > recent_step_cutoff:
                continue
            tokens = _msg_tokens(msg)
            if tokens < 100:
                continue
            truncatable.append((i, tokens))

        truncatable.sort(key=lambda x: x[1], reverse=True)

        result = list(messages)
        saved_tokens = 0
        need_to_save = current_tokens - target_tokens

        for idx, original_tokens in truncatable:
            if saved_tokens >= need_to_save:
                break
            msg = result[idx]
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            truncated_content = content[:self.truncated_chars]
            omitted_chars = len(content) - self.truncated_chars
            new_content = (
                f"{truncated_content}\n\n"
                f"... [{omitted_chars} chars omitted, use read_file to see full content] ..."
            )
            result[idx] = ToolMessage(
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                id=msg.id,
            )
            new_tokens = _msg_tokens(result[idx])
            saved_tokens += original_tokens - new_tokens

        return result
