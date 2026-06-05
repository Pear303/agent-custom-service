"""子代理子图 —— 将子代理定义为 LangGraph StateGraph 子图。

每个子代理拥有独立的 system prompt、工具白名单和最大轮数限制。

并发执行：
  子图使用 ParallelToolNode 替代标准 ToolNode，支持同一帧内多个只读工具
  并发执行。ContextVar 通过快照-恢复机制在工作线程间传播。

异步节点：
  call_subagent 使用 async def + ainvoke，I/O 等待期间释放事件循环，
  允许同一 superstep 中无依赖的节点并发执行。
"""
from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages

from agent.subagents.spec import SubagentSpec
from agent_by_langgraph.lg_parallel_tools import ParallelToolNode


class SubagentState(TypedDict):
    """子代理子图的状态定义。

    Attributes:
        messages: 当前轮次的消息序列（含 tool_calls / tool_responses）
        input: 子代理要执行的原始任务
        turns_remaining: 剩余 agent→tools 循环轮数；每轮减 1，归零则强制结束
        max_turns: 起始上限（保留字段，便于诊断）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input: str
    turns_remaining: int
    max_turns: int


def _route_after_agent(state: SubagentState) -> str:
    """agent 节点后路由：有 tool_calls 走 tools，否则直接 END。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _post_tools(state: SubagentState) -> dict:
    """tools 节点之后：递减剩余轮数；归零时注入终止消息以强制 END。"""
    remaining = state["turns_remaining"] - 1
    if remaining <= 0:
        return {
            "turns_remaining": 0,
            "messages": [
                AIMessage(
                    content="[子代理已达最大轮数限制，任务可能未完成]"
                )
            ],
        }
    return {"turns_remaining": remaining}


def _route_after_tools(state: SubagentState) -> str:
    """post_tools 之后路由：还有轮数则回 agent 继续循环，否则终止。"""
    if state["turns_remaining"] <= 0:
        return END
    return "agent"


def create_subagent_graph(llm, spec: SubagentSpec):
    """创建子代理子图。

    图结构：
        START → agent → _route_after_agent
                            ├── 有 tool_calls → tools → post_tools → _route_after_tools
                            │                                       ├── 还有轮数 → agent (循环)
                            │                                       └── 归零 → END
                            └── 无 tool_calls → END

    Args:
        llm: DeepSeekChatOpenAI 实例
        spec: SubagentSpec 子代理规格

    Returns:
        CompiledGraph: 子图，调用方应传入
            `{"input": task, "turns_remaining": spec.max_turns, "max_turns": spec.max_turns, "messages": []}`
    """
    from agent_by_langgraph.lg_tools import make_subagent_tools

    tools = make_subagent_tools(spec.tool_names)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ParallelToolNode(tools)

    async def call_subagent(state: SubagentState, config: RunnableConfig) -> dict:
        # 首轮：构建 [SystemMessage, HumanMessage(input)] 作为起始
        # 后续轮次：state["messages"] 已包含前一轮的完整消息序列
        # （含 SystemMessage），通过 add_messages reducer 自动累积，
        # 无需重复注入 SystemMessage
        existing = state.get("messages", [])
        if not existing:
            # 首轮：只有 input，无历史
            msgs = [
                SystemMessage(content=spec.system_prompt),
                HumanMessage(content=state["input"]),
            ]
        else:
            # 后续轮次：existing 已含 SystemMessage + HumanMessage + AIMessage + ToolMessage
            # 直接使用，不重复注入 SystemMessage
            msgs = list(existing)
        response = await llm_with_tools.ainvoke(msgs, config=config)
        return {"messages": [response]}

    builder = StateGraph(SubagentState)
    builder.add_node("agent", call_subagent)
    builder.add_node("tools", tool_node)
    builder.add_node("post_tools", _post_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _route_after_agent,
        {"tools": "tools", END: END},
    )
    builder.add_edge("tools", "post_tools")
    builder.add_conditional_edges(
        "post_tools", _route_after_tools,
        {"agent": "agent", END: END},
    )

    return builder.compile()


# 按 (model_name, agent_type) 区分的子图缓存
# 注意：缓存的是 CompiledGraph，持有对 LLM 与工具的强引用；
# 长期运行的服务应在切换 LLM / 模型时调用 clear_subgraph_cache() 释放旧实例
_subgraph_cache: dict[tuple[str, str], object] = {}


def get_subagent_graph(llm, registry, agent_type: str):
    """获取或创建缓存的子代理子图。

    缓存 key: (model_name, agent_type) —— 相同模型 + 相同子代理类型复用子图，
    不同模型则创建新子图（因为工具绑定与 LLM 实例相关）。

    Args:
        llm: ChatModel 实例
        registry: SubagentRegistry
        agent_type: 子代理名称

    Returns:
        CompiledGraph

    Raises:
        ValueError: 未知 agent_type
    """
    model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "") or "unknown"
    key = (model_name, agent_type)
    if key not in _subgraph_cache:
        spec = registry.get(agent_type)
        if spec is None:
            available = ", ".join(registry.names())
            raise ValueError(
                f"Unknown subagent '{agent_type}'. Available: {available}"
            )
        _subgraph_cache[key] = create_subagent_graph(llm, spec)
    return _subgraph_cache[key]


def clear_subgraph_cache() -> None:
    """清空子图缓存。

    适用场景：
    - 切换 LLM 模型后强制重建（id 改变本就会重建，但清理可释放旧 CompiledGraph）
    - 进程退出前的清理（防止 CompiledGraph 持有 LLM / 工具阻止 GC）
    - 单元测试间隔离
    """
    _subgraph_cache.clear()
