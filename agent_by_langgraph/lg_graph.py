"""LangGraph StateGraph 定义 — 主 Agent 的图结构。

用 StateGraph 替代 LangChain 的 AgentExecutor，显式控制 Agent 循环。

消息拼接策略：
  初始调用时，调用方将 chat_history + input 作为 messages 的一部分传入，
  call_agent 只需在 messages 前插入 SystemMessage，不再重复拼接
  chat_history 和 input，避免多轮循环时 token 浪费。

兼容性：
  LangGraph >= 1.2 的 compile() 不再接受 callbacks 参数，
  图级回调改为在 invoke/astream 时通过 config.callbacks 传入。
"""
from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import tools_condition

from agent_by_langgraph.lg_parallel_tools import ParallelToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_agent_graph(
    llm, tools, system_prompt: str,
    llm_callbacks: list | None = None,
    checkpointer=None,
):
    """构造并编译主 Agent 的 StateGraph。

    图结构：
        START → agent → tools_condition
                            ├──→ tools → agent (循环)
                            └──→ END

    Args:
        llm: 绑定了 DeepSeek API 的 ChatOpenAI 实例（支持 reasoning_content）
        tools: 工具列表
        system_prompt: 构建好的系统提示词
        llm_callbacks: 图级回调（始终生效，如 token 追踪）。
            LangGraph >= 1.2 不再支持 compile(callbacks=...)，
            改为在 invoke 时通过 config.callbacks 传入。
            此参数保留用于向后兼容，实际回调在 invoke 时合并。
        checkpointer: 可选的 LangGraph Checkpointer（如 SqliteSaver），
            启用后支持状态持久化、时间旅行调试和断点续跑

    Returns:
        CompiledGraph

    Note:
        - 回调通过 invoke(input, config={"callbacks": [...]}) 传入，
          在节点内透传给 LLM，使 ReasoningCollector 等能捕获 reasoning_content
        - 调用方应传入 {"messages": [SystemMessage, ...chat_history, HumanMessage(input)]}
          而非分开传 chat_history 和 input，避免多轮循环时重复拼接
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ParallelToolNode(tools)

    def call_agent(state: AgentState, config: RunnableConfig) -> dict:
        # state["messages"] 已包含完整上下文：
        #   [SystemMessage, ...chat_history, HumanMessage(input), AIMessage, ToolMessage, ...]
        # 多轮 agent→tools 循环时，messages 通过 add_messages reducer 自动累积，
        # 无需重复拼接 chat_history 和 input
        response = llm_with_tools.invoke(state["messages"], config=config)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    compiled = builder.compile(checkpointer=checkpointer)
    # 保存 llm_callbacks 供调用方在 invoke 时合并到 config.callbacks
    compiled._lg_llm_callbacks = list(llm_callbacks) if llm_callbacks else []
    return compiled
