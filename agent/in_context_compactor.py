"""InContextCompactor: 会话内 verbatim 上下文压缩。

与跨会话归档的 Compactor 完全独立：
- InContextCompactor: 会话内，纯规则，裁剪注入 LLM 的消息视图
- Compactor: 跨会话，调 LLM，归档到 MEMORY.md

核心策略：只裁剪 ToolMessage 的体积，不删除任何 AIMessage。
原因：AIMessage 是 Agent 的决策链，删除会导致推理断裂。
工具输出才是上下文膨胀的主要来源（占 60-80%）。

裁剪规则：
1. SystemMessage — 不动
2. HumanMessage — 不动
3. AIMessage — 不动（含 reasoning_content 的更不动）
4. ToolMessage — 按体积优先级截断：
   a. 最近的 KEEP_RECENT_STEPS 步工具结果完整保留
   b. 旧的大体积 ToolMessage 截断为摘要（保留 name + 前 N 字符）
   c. 截断顺序：体积最大的优先
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# 防御性导入：确保工具调用完整性后验证可用
from agent.context_view import _ensure_tool_call_integrity


# 保留最近 N 步的工具结果完整
_KEEP_RECENT_STEPS = 5

# 截断后的摘要字符数
_TRUNCATED_CHARS = 200


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
    return sum(
        _estimate_tokens(b.get("text", ""))
        for b in content
        if isinstance(b, dict)
    )


def _compute_step_map(messages: list[BaseMessage]) -> dict[int, int]:
    """计算每条消息属于第几"步"。

    步的定义：以 AIMessage(tool_calls=...) 为步界，
    AIMessage 及其对应的 ToolMessage 属于同一步。
    """
    step_map: dict[int, int] = {}
    current_step = 0

    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            current_step += 1
        step_map[i] = current_step

    return step_map


class InContextCompactor:
    """会话内 verbatim 上下文压缩器。

    只裁剪 ToolMessage 的体积，不删除任何消息。
    在 call_agent 节点中、注入 LLM 前调用。
    """

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
        """对消息列表做 verbatim 压缩，返回裁剪后的列表。

        不修改原始消息对象，创建新的 ToolMessage 替换旧的大体积输出。

        Args:
            messages: 完整的消息序列

        Returns:
            裁剪后的消息序列（原始消息对象不变，ToolMessage 可能被替换为截断版本）
        """
        total_tokens = sum(_msg_tokens(m) for m in messages)
        threshold = self.max_context_tokens * self.target_ratio

        if total_tokens <= threshold:
            return messages  # 不需要压缩

        result = self._truncate_old_tool_messages(messages, total_tokens, threshold)
        # 防御性后验证：确保截断未破坏工具调用完整性
        return _ensure_tool_call_integrity(result)

    def _truncate_old_tool_messages(
        self,
        messages: list[BaseMessage],
        current_tokens: int,
        target_tokens: int,
    ) -> list[BaseMessage]:
        """截断旧的 ToolMessage，释放 token 空间。

        步骤：
        1. 找到所有 ToolMessage，按"步数距离"和"体积"排序
        2. 最近的 keep_recent_steps 步内的 ToolMessage 不截断
        3. 旧的大体积 ToolMessage 优先截断
        4. 截断到目标 token 数为止
        """
        step_map = _compute_step_map(messages)
        max_step = max(step_map.values()) if step_map else 0
        # keep_recent_steps 步内的 ToolMessage 不截断
        # 例如 max_step=5, keep_recent_steps=2 → cutoff=3, step 4和5不截断
        recent_step_cutoff = max_step - self.keep_recent_steps

        # 收集可截断的 ToolMessage（旧的、大体积的优先）
        truncatable: list[tuple[int, int]] = []  # (index, original_tokens)
        for i, msg in enumerate(messages):
            if not isinstance(msg, ToolMessage):
                continue
            step = step_map.get(i, 0)
            if step > recent_step_cutoff:
                continue  # 最近的步，不截断
            tokens = _msg_tokens(msg)
            if tokens < 100:
                continue  # 已经很短，不需要截断
            truncatable.append((i, tokens))

        # 按体积降序排列（大体积优先截断，收益最大）
        truncatable.sort(key=lambda x: x[1], reverse=True)

        # 逐个截断，直到 token 降到目标
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
