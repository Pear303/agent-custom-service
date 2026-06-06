"""ContextView: 会话内上下文视图裁剪。

在 call_agent 节点注入 LLM 前构建裁剪后的消息视图。
State 和 Checkpointer 始终保持完整消息序列，视图只是临时过滤。

设计原则：
1. State 不可变，视图可裁剪 — 不在 reducer 层面删除消息
2. 工具调用组原子性 — AIMessage(tool_calls) + ToolMessage 作为整体
3. reasoning_content 不可触碰 — 含 DeepSeek reasoning_content 的消息不可裁剪
4. 动态窗口 — 保留窗口大小基于 token 估算动态调整
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def _has_reasoning(msg: BaseMessage) -> bool:
    """检查消息是否包含 DeepSeek reasoning_content，不可裁剪。"""
    return bool(
        isinstance(msg, AIMessage)
        and getattr(msg, "additional_kwargs", {}).get("reasoning_content")
    )


def _estimate_tokens(text: str) -> int:
    """粗估 token 数：中文约 1.5 字符/token，英文约 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)


def _msg_tokens(msg: BaseMessage) -> int:
    """估算单条消息的 token 数。"""
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return _estimate_tokens(content)
    # 多模态内容块
    return sum(
        _estimate_tokens(b.get("text", ""))
        for b in content
        if isinstance(b, dict)
    )


def _build_tool_call_groups(messages: list[BaseMessage]) -> list[set[int]]:
    """识别所有工具调用组。

    工具调用组 = AIMessage(tool_calls=...) + 紧随其后的所有对应 ToolMessage。
    这些消息必须作为原子单位保留或裁剪，否则 LangGraph 会报错。

    Returns:
        工具调用组列表，每个组是一组消息索引的集合
    """
    # 建立 tool_call_id → ToolMessage 索引的映射
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
    """构建消息索引 → 工具调用组索引的映射。

    Returns:
        dict: 消息索引 → 组索引（不属于任何组的消息不在 dict 中）
    """
    groups = _build_tool_call_groups(messages)
    idx_to_group: dict[int, int] = {}
    for gi, group in enumerate(groups):
        for idx in group:
            idx_to_group[idx] = gi
    return idx_to_group


def _ensure_tool_call_integrity(messages: list[BaseMessage]) -> list[BaseMessage]:
    """确保 AIMessage(tool_calls) 后紧跟所有对应的 ToolMessage。

    如果 ToolMessage 被裁剪掉，必须连带裁剪对应的 AIMessage(tool_calls)，
    否则 LLM API 会报 "insufficient tool messages following tool_calls" 错误。

    策略：
    1. 收集所有存在的 tool_call_id
    2. 对于每条 AIMessage(tool_calls)，检查其所有 tool_call_id 是否都有对应的 ToolMessage
    3. 如果缺少 ToolMessage，从该 AIMessage 中移除对应的 tool_call
    4. 如果移除后 AIMessage 不再有任何 tool_calls，则移除整条 AIMessage
    """
    # 收集所有存在的 tool_call_id
    existing_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            existing_tool_call_ids.add(msg.tool_call_id)

    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # 检查每个 tool_call 是否有对应的 ToolMessage
            complete_calls = [
                tc for tc in msg.tool_calls
                if tc.get("id") in existing_tool_call_ids
            ]
            if not complete_calls:
                # 所有 tool_calls 都缺少 ToolMessage，移除整条 AIMessage
                continue
            if len(complete_calls) < len(msg.tool_calls):
                # 部分 tool_calls 缺少 ToolMessage，创建新的 AIMessage 只保留完整的
                new_msg = AIMessage(
                    content=msg.content,
                    tool_calls=complete_calls,
                    id=msg.id,
                    additional_kwargs=msg.additional_kwargs,
                )
                result.append(new_msg)
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


class ContextView:
    """会话内上下文视图裁剪器。

    从完整的 state["messages"] 构建一个裁剪后的消息列表，
    用于注入 LLM。不修改原始 state。

    裁剪策略（按优先级）：
    1. SystemMessage — 始终保留
    2. 含 reasoning_content 的 AIMessage — 始终保留（DeepSeek 要求）
    3. milestone 消息 — 保留（用户原始请求、关键决策点）
    4. 最近的工具调用组 — 作为原子单位保留（keep_recent_groups 个）
    5. 旧的工具调用组 — 允许整体裁剪（组内原子性不变）
    6. 最近的消息 — 保留（滑动窗口）
    7. 旧的非关键消息 — 裁剪掉
    """

    def __init__(
        self,
        max_context_tokens: int = 200_000,
        target_ratio: float = 0.6,
        min_window: int = 6,
        keep_recent_groups: int = 5,
    ):
        """
        Args:
            max_context_tokens: 上下文窗口最大 token 数
            target_ratio: 目标使用比例（裁剪后不超过此比例）
            min_window: 最少保留的最近消息条数
            keep_recent_groups: 保留最近的工具调用组数量（旧组允许整体裁剪）
        """
        self.max_context_tokens = max_context_tokens
        self.target_ratio = target_ratio
        self.min_window = min_window
        self.keep_recent_groups = keep_recent_groups

    def build_view(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """构建裁剪后的消息视图。

        Args:
            messages: state["messages"] 的完整消息序列

        Returns:
            裁剪后的消息列表，可直接传给 LLM
        """
        if not messages:
            return messages

        # 快速检查：如果总 token 未超阈值，直接返回
        total_tokens = sum(_msg_tokens(m) for m in messages)
        threshold = self.max_context_tokens * self.target_ratio
        if total_tokens <= threshold:
            return messages

        # 需要裁剪 — 构建保留计划
        must_keep = self._compute_must_keep(messages)
        view = self._build_with_budget(messages, must_keep, threshold)
        return view

    def _compute_must_keep(self, messages: list[BaseMessage]) -> set[int]:
        """计算必须保留的消息索引集合。"""
        must_keep: set[int] = set()
        groups = _build_tool_call_groups(messages)
        idx_to_group = _build_index_to_group(messages)

        # 确定哪些工具调用组是"最近的"（必须保留）
        # 组按出现顺序编号，保留最后 keep_recent_groups 个
        recent_group_indices: set[int] = set()
        if groups:
            start = max(0, len(groups) - self.keep_recent_groups)
            recent_group_indices = set(range(start, len(groups)))

        for i, msg in enumerate(messages):
            # 规则1: SystemMessage 始终保留
            if isinstance(msg, SystemMessage):
                must_keep.add(i)
            # 规则2: 含 reasoning_content 的 AIMessage 始终保留
            elif _has_reasoning(msg):
                must_keep.add(i)
            # 规则3: milestone 消息保留
            elif getattr(msg, "metadata", {}).get("milestone", False):
                must_keep.add(i)
            # 规则4: 最近的工具调用组作为原子单位保留
            # 旧的工具调用组不标记为 must_keep，允许整体裁剪
            elif i in idx_to_group and idx_to_group[i] in recent_group_indices:
                must_keep.add(i)

        # 确保工具调用组原子性：如果 must_keep 中有组内某个成员，
        # 整组都必须保留（例如 reasoning_content AIMessage 属于旧组）
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
    ) -> list[BaseMessage]:
        """在 token 预算内构建消息视图。

        策略：
        1. 先加入所有 must_keep 消息
        2. 从尾部向前填充最近的消息，直到预算用完
        3. 确保工具调用组的原子性（不拆组）
        """
        # 计算必须保留消息的 token 消耗
        must_keep_tokens = sum(_msg_tokens(messages[i]) for i in must_keep)

        # 如果必须保留的消息已超预算，全部返回（无法裁剪）
        if must_keep_tokens >= token_budget:
            return [messages[i] for i in sorted(must_keep)]

        remaining_budget = token_budget - must_keep_tokens
        idx_to_group = _build_index_to_group(messages)

        # 从尾部向前，添加非 must_keep 的最近消息
        additional: set[int] = set()

        for i in range(len(messages) - 1, -1, -1):
            if i in must_keep or i in additional:
                continue

            msg = messages[i]
            msg_tokens = _msg_tokens(msg)

            # 如果这条消息属于某个工具调用组，整组加入
            if i in idx_to_group:
                gi = idx_to_group[i]
                group = _build_tool_call_groups(messages)[gi]
                group_members = group - must_keep - additional
                group_tokens = sum(_msg_tokens(messages[j]) for j in group_members)

                if group_tokens <= remaining_budget:
                    additional.update(group_members)
                    remaining_budget -= group_tokens
                else:
                    break  # 预算不够，停止
            else:
                if msg_tokens <= remaining_budget:
                    additional.add(i)
                    remaining_budget -= msg_tokens
                else:
                    break  # 预算用完，停止

        # 合并并排序
        kept = must_keep | additional
        result = [messages[i] for i in sorted(kept)]

        # 确保满足最小窗口（同时保证工具调用组原子性）
        if len(result) < self.min_window:
            idx_to_group_2 = _build_index_to_group(messages)
            groups_2 = _build_tool_call_groups(messages)
            for i in range(len(messages) - 1, -1, -1):
                if i not in kept:
                    # 如果这条消息属于工具调用组，整组添加
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
                    if len(result) >= self.min_window:
                        break
            # 按原始顺序排序
            result.sort(key=lambda m: messages.index(m))

        # 后验证：确保 AIMessage(tool_calls) 后紧跟所有对应的 ToolMessage
        # 如果 ToolMessage 被裁剪掉，必须连带裁剪 AIMessage(tool_calls)，
        # 否则 LLM API 会报 "insufficient tool messages" 错误
        result = _ensure_tool_call_integrity(result)

        return result
