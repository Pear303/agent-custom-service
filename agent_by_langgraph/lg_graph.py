"""LangGraph StateGraph 定义 — 主 Agent 的图结构。

用 StateGraph 替代 LangChain 的 AgentExecutor，显式控制 Agent 循环。
"""
from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input: str
    chat_history: Sequence[BaseMessage]


def create_agent_graph(
    llm, tools, system_prompt: str,
    llm_callbacks: list | None = None,
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
        llm_callbacks: 图级回调（始终生效，如 token 追踪）

    Returns:
        CompiledGraph

    Note:
        - 图级回调通过 `compile(callbacks=...)` 注册，作用于所有节点
        - per-invoke 回调通过 `invoke(input, config={"callbacks": [...]})` 传入
          并在节点内透传给 LLM，使 ReasoningCollector 等能捕获 reasoning_content
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    graph_callbacks = list(llm_callbacks) if llm_callbacks else []

    def call_agent(state: AgentState, config: RunnableConfig) -> dict:
        msgs = [SystemMessage(content=system_prompt)]
        msgs.extend(state["chat_history"])
        msgs.append(HumanMessage(content=state["input"]))
        response = llm_with_tools.invoke(msgs, config=config)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(callbacks=graph_callbacks)
