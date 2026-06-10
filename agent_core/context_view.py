"""ContextView: 会话内上下文视图裁剪。

从 agent.context_view 迁移，无 import 变更。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass
class PrunedToolCallGroup:
    """被裁剪掉的工具调用组，供决策摘要提取。"""
    group_index: int
    ai_message: BaseMessage | None = None
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[BaseMessage] = field(default_factory=list)


def _has_reasoning(msg: BaseMessage) -> bool:
    return bool(
        isinstance(msg, AIMessage)
        and getattr(msg, "additional_kwargs", {}).get("reasoning_content")
    )


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


def _build_tool_call_groups(messages: list[BaseMessage]) -> list[set[int]]:
    tool_call_to_tool_msg: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_call_to_tool_msg[msg.tool_call_id] = i

    groups: list[set[int]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            group: set[int] = {i}
            for tc in msg.tool_calls:
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if tc_id and tc_id in tool_call_to_tool_msg:
                    group.add(tool_call_to_tool_msg[tc_id])
            groups.append(group)
    return groups


def _build_index_to_group(messages: list[BaseMessage]) -> dict[int, int]:
    groups = _build_tool_call_groups(messages)
    idx_to_group: dict[int, int] = {}
    for gi, group in enumerate(groups):
        for idx in group:
            idx_to_group[idx] = gi
    return idx_to_group


def _ensure_tool_call_integrity(messages: list[BaseMessage]) -> list[BaseMessage]:
    """确保 AIMessage(tool_calls) 后紧跟所有对应的 ToolMessage。"""
    existing_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            existing_tool_call_ids.add(msg.tool_call_id)

    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            complete_calls = [
                tc for tc in msg.tool_calls
                if tc.get("id") in existing_tool_call_ids
            ]
            if not complete_calls:
                continue
            if len(complete_calls) < len(msg.tool_calls):
                complete_ids = {tc.get("id") for tc in complete_calls}
                new_additional_kwargs = dict(msg.additional_kwargs)
                if "tool_calls" in new_additional_kwargs:
                    new_additional_kwargs["tool_calls"] = [
                        tc for tc in new_additional_kwargs["tool_calls"]
                        if tc.get("id") in complete_ids
                    ]
                new_msg = AIMessage(
                    content=msg.content,
                    tool_calls=complete_calls,
                    id=msg.id,
                    additional_kwargs=new_additional_kwargs,
                )
                result.append(new_msg)
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


class ContextView:
    """会话内上下文视图裁剪器。"""

    def __init__(
        self,
        max_context_tokens: int = 200_000,
        target_ratio: float = 0.6,
        min_window: int = 6,
        keep_recent_groups: int = 5,
    ):
        self.max_context_tokens = max_context_tokens
        self.target_ratio = target_ratio
        self.min_window = min_window
        self.keep_recent_groups = keep_recent_groups

    def build_view(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[PrunedToolCallGroup]]:
        if not messages:
            return messages, []

        total_tokens = sum(_msg_tokens(m) for m in messages)
        threshold = self.max_context_tokens * self.target_ratio
        if total_tokens <= threshold:
            return messages, []

        must_keep = self._compute_must_keep(messages)
        view, kept_indices = self._build_with_budget(messages, must_keep, threshold)
        pruned_groups = self._collect_pruned_groups(messages, kept_indices)
        return view, pruned_groups

    def _compute_must_keep(self, messages: list[BaseMessage]) -> set[int]:
        must_keep: set[int] = set()
        groups = _build_tool_call_groups(messages)
        idx_to_group = _build_index_to_group(messages)

        recent_group_indices: set[int] = set()
        if groups:
            start = max(0, len(groups) - self.keep_recent_groups)
            recent_group_indices = set(range(start, len(groups)))

        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                must_keep.add(i)
            elif _has_reasoning(msg):
                must_keep.add(i)
            elif getattr(msg, "metadata", {}).get("milestone", False):
                must_keep.add(i)
            elif i in idx_to_group and idx_to_group[i] in recent_group_indices:
                must_keep.add(i)

        for i in list(must_keep):
            if i in idx_to_group:
                gi = idx_to_group[i]
                group = groups[gi]
                must_keep.update(group)

        return must_keep

    def _build_with_budget(
        self,
        messages: list[BaseMessage],
        must_keep: set[int],
        token_budget: int,
    ) -> tuple[list[BaseMessage], set[int]]:
        must_keep_tokens = sum(_msg_tokens(messages[i]) for i in must_keep)
        if must_keep_tokens >= token_budget:
            kept = must_keep
            return [messages[i] for i in sorted(kept)], kept

        remaining_budget = token_budget - must_keep_tokens
        idx_to_group = _build_index_to_group(messages)

        additional: set[int] = set()
        for i in range(len(messages) - 1, -1, -1):
            if i in must_keep or i in additional:
                continue
            msg = messages[i]
            msg_tokens = _msg_tokens(msg)

            if i in idx_to_group:
                gi = idx_to_group[i]
                group = _build_tool_call_groups(messages)[gi]
                group_members = group - must_keep - additional
                group_tokens = sum(_msg_tokens(messages[j]) for j in group_members)
                if group_tokens <= remaining_budget:
                    additional.update(group_members)
                    remaining_budget -= group_tokens
                else:
                    break
            else:
                if msg_tokens <= remaining_budget:
                    additional.add(i)
                    remaining_budget -= msg_tokens
                else:
                    break

        kept = must_keep | additional
        result = [messages[i] for i in sorted(kept)]

        if len(result) < self.min_window:
            idx_to_group_2 = _build_index_to_group(messages)
            groups_2 = _build_tool_call_groups(messages)
            for i in range(len(messages) - 1, -1, -1):
                if i not in kept:
                    if i in idx_to_group_2:
                        gi = idx_to_group_2[i]
                        group = groups_2[gi]
                        for j in group:
                            if j not in kept:
                                result.append(messages[j])
                                kept.add(j)
                    else:
                        result.append(messages[i])
                        kept.add(i)
                    if i not in idx_to_group_2 and len(result) >= self.min_window:
                        break
            result.sort(key=lambda m: messages.index(m))

        result = _ensure_tool_call_integrity(result)
        return result, kept

    def _collect_pruned_groups(
        self,
        messages: list[BaseMessage],
        kept_indices: set[int],
    ) -> list[PrunedToolCallGroup]:
        groups = _build_tool_call_groups(messages)
        pruned: list[PrunedToolCallGroup] = []

        for gi, group in enumerate(groups):
            if group & kept_indices:
                continue
            ai_msg = None
            tool_calls = []
            tool_results = []
            for idx in sorted(group):
                msg = messages[idx]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    ai_msg = msg
                    tool_calls = list(msg.tool_calls)
                elif isinstance(msg, ToolMessage):
                    tool_results.append(msg)
            pruned.append(PrunedToolCallGroup(
                group_index=gi,
                ai_message=ai_msg,
                tool_calls=tool_calls,
                tool_results=tool_results,
            ))
        return pruned
