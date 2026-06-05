"""并发工具节点 —— 替换 LangGraph 预构建的 ToolNode，支持只读工具并发执行。

当 LLM 在同一帧回复中发出多个 tool_calls 时：
- 只读工具（web_fetch、read_file、glob、grep、load_skill）并发执行
- 非只读工具（write_file、edit_file、run_command 等）顺序执行

ContextVar 传播方案：
  Python 的 ThreadPoolExecutor 不会自动传播 ContextVar，因此
  在提交任务前捕获当前线程的 ContextVar 快照，在工作线程内恢复。

兼容性：
  LangGraph >= 1.2 的 add_node 要求传入 Runnable 或 callable，
  因此 ParallelToolNode 实现 __call__ 使其可作为 callable 节点使用。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from agent.lc_tools import (
    _ctx_workspace,
    _ctx_skills_loader,
    _ctx_todo_store,
    _ctx_subagent_registry,
    _ctx_llm_ref,
    _ctx_user_id,
    _ctx_ticket_id,
)

# 只读工具集合，可安全并发
_READ_ONLY_TOOLS = frozenset({
    "web_fetch",
    "read_file",
    "glob_tool",
    "grep_tool",
    "load_skill",
})


def _snapshot_context_vars() -> dict:
    """捕获当前线程的所有 ContextVar 值快照。"""
    return {
        "workspace": _ctx_workspace.get(),
        "skills_loader": _ctx_skills_loader.get(),
        "todo_store": _ctx_todo_store.get(),
        "subagent_registry": _ctx_subagent_registry.get(),
        "llm_ref": _ctx_llm_ref.get(),
        "user_id": _ctx_user_id.get(),
        "ticket_id": _ctx_ticket_id.get(),
    }


def _restore_context_vars(snapshot: dict) -> None:
    """在工作线程内恢复 ContextVar 值。"""
    _ctx_workspace.set(snapshot["workspace"])
    _ctx_skills_loader.set(snapshot["skills_loader"])
    _ctx_todo_store.set(snapshot["todo_store"])
    _ctx_subagent_registry.set(snapshot["subagent_registry"])
    _ctx_llm_ref.set(snapshot["llm_ref"])
    _ctx_user_id.set(snapshot["user_id"])
    _ctx_ticket_id.set(snapshot["ticket_id"])


class ParallelToolNode:
    """支持只读工具并发执行的工具节点，替代 LangGraph 的 ToolNode。

    用法与 ToolNode 完全兼容，可直接替换：
        tool_node = ParallelToolNode(tools)   # 替代 ToolNode(tools)
        builder.add_node("tools", tool_node)

    行为差异：
    - 当 AIMessage 包含多个只读 tool_calls 时，用 ThreadPoolExecutor 并发执行
    - 非只读工具始终顺序执行
    - 单个 tool_call 时直接执行，无并发开销

    兼容性：
    - 实现 __call__ 使其可作为 LangGraph add_node 的 callable 参数
    - 实现 invoke 保持与 Runnable 接口兼容
    """

    def __init__(self, tools: Sequence[BaseTool], *, verbose: bool = False):
        self._tool_node = ToolNode(tools)
        self._name_to_tool: dict[str, BaseTool] = {t.name: t for t in tools}
        self._verbose = verbose

    def __call__(self, state: dict, config: RunnableConfig) -> dict:
        """LangGraph add_node 要求 callable 签名：fn(state, config) -> dict。"""
        return self.invoke(state, config)

    def invoke(self, state: dict, config: RunnableConfig | None = None) -> dict:
        """执行工具调用，只读工具并发，非只读工具顺序。"""
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return {"messages": []}

        tool_calls = last_msg.tool_calls

        # 单个 tool_call → 直接走 ToolNode，无并发开销
        if len(tool_calls) <= 1:
            return self._tool_node.invoke(state, config)

        # 分类：只读 vs 非只读
        read_only_calls = [tc for tc in tool_calls if tc["name"] in _READ_ONLY_TOOLS]
        write_calls = [tc for tc in tool_calls if tc["name"] not in _READ_ONLY_TOOLS]

        # 如果没有多个只读工具需要并发，直接走 ToolNode
        if len(read_only_calls) <= 1:
            return self._tool_node.invoke(state, config)

        # 有多个只读工具 → 并发执行
        results: list[ToolMessage] = []

        # 先顺序执行非只读工具（通过 ToolNode，使用完整 state 上下文）
        if write_calls:
            # 构造仅包含非只读 tool_calls 的 AIMessage，保留原始消息上下文
            write_ai_msg = AIMessage(
                content=last_msg.content or "",
                tool_calls=write_calls,
                additional_kwargs=last_msg.additional_kwargs,
                id=last_msg.id,
            )
            write_state = {"messages": list(messages[:-1]) + [write_ai_msg]}
            write_result = self._tool_node.invoke(write_state, config)
            results.extend(write_result.get("messages", []))

        # 并发执行只读工具（直接调用工具实例，避免构造不完整的 state）
        if read_only_calls and len(read_only_calls) >= 2:
            if self._verbose:
                names = ", ".join(tc["name"] for tc in read_only_calls)
                print(f"\n[并发执行 {len(read_only_calls)} 个只读工具]: {names}\n")

            ctx_snapshot = _snapshot_context_vars()

            def _run_single(tc: dict) -> ToolMessage:
                """在当前线程执行单个工具调用（ContextVar 已恢复）。"""
                _restore_context_vars(ctx_snapshot)
                tool = self._name_to_tool.get(tc["name"])
                if tool is None:
                    return ToolMessage(
                        content=f"Error: unknown tool '{tc['name']}'",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status="error",
                    )
                try:
                    observation = tool.invoke(tc["args"])
                    return ToolMessage(
                        content=str(observation),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                except Exception as exc:
                    return ToolMessage(
                        content=f"Error: {exc}",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status="error",
                    )

            with ThreadPoolExecutor(max_workers=len(read_only_calls)) as pool:
                parallel_results = list(pool.map(_run_single, read_only_calls))
            results.extend(parallel_results)

        elif read_only_calls:
            # 只有一个只读工具，走 ToolNode
            single_ai_msg = AIMessage(
                content=last_msg.content or "",
                tool_calls=read_only_calls,
                additional_kwargs=last_msg.additional_kwargs,
                id=last_msg.id,
            )
            single_state = {"messages": list(messages[:-1]) + [single_ai_msg]}
            single_result = self._tool_node.invoke(single_state, config)
            results.extend(single_result.get("messages", []))

        return {"messages": results}
