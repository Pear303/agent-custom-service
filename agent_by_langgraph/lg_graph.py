"""LangGraph StateGraph 定义 — 主 Agent 的图结构。

用 StateGraph 替代 LangChain 的 AgentExecutor，显式控制 Agent 循环。

消息拼接策略：
  初始调用时，调用方将 chat_history + input 作为 messages 的一部分传入，
  call_agent 只需在 messages 前插入 SystemMessage，不再重复拼接
  chat_history 和 input，避免多轮循环时 token 浪费。

并行子代理派遣（Send API）：
  当 LLM 在同一帧发出多个 dispatch_subagent_lg 调用时，
  route_after_agent 将它们路由到 subagent_dispatcher 节点，
  由该节点通过 Send 原语生成多个并行 subagent_worker 实例。
  所有 worker 在同一 superstep 内并行执行，结果通过
  Annotated[list[str], add] reducer 安全合并到 subagent_results，
  最后由 aggregate_results 节点汇总为一条 AIMessage 注入主对话流。

安全审批门（interrupt）：
  危险工具（write_file、edit_file、run_command）调用前，
  route_after_agent 将其路由到 interrupt_approval 节点，
  该节点调用 interrupt() 暂停执行，等待人工确认后继续。

兼容性：
  LangGraph >= 1.2 的 compile() 不再接受 callbacks 参数，
  图级回调改为在 invoke/astream 时通过 config.callbacks 传入。
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Send, interrupt

from agent_by_langgraph.lg_parallel_tools import ParallelToolNode


# 需要人工审批的危险工具
_DANGEROUS_TOOLS = frozenset({
    "write_file", "edit_file", "run_command",
})


class AgentState(TypedDict):
    """主 Agent 状态。

    Attributes:
        messages: 消息序列，通过 add_messages reducer 自动累积
        subagent_results: 并行子代理结果，通过 add reducer 安全合并
            多个 subagent_worker 在同一 superstep 并行写入，
            aggregate_results 消费后清空
        _approval_next: 审批路由指示，interrupt_approval 设置，
            _route_after_approval 消费。值为 "tools" 或 "agent"
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    subagent_results: Annotated[list[str], add]
    _approval_next: str


class SubagentWorkerState(TypedDict):
    """单个子代理 worker 的独立状态。

    每个 Send("subagent_worker", ...) 创建一个独立的 worker 实例，
    拥有自己的 state，不与主 AgentState 共享。
    """
    agent_type: str
    task: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _route_after_agent(state: AgentState) -> str | list[Send]:
    """agent 节点后的路由：区分普通工具调用、子代理派遣、危险工具审批、直接结束。

    路由优先级：
    1. 无 tool_calls → END
    2. 有 dispatch_subagent_lg 调用 → subagent_dispatcher（并行派遣）
    3. 有危险工具调用 → interrupt_approval（安全审批）
    4. 普通工具调用 → tools

    注意：当子代理调用和普通工具调用混合出现时，
    优先走 subagent_dispatcher（子代理调用通常更重要且耗时更长）。
    普通工具调用会在下一轮 agent→tools 循环中被执行。
    """
    last = state["messages"][-1]

    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END

    # 分类 tool_calls
    subagent_calls = [tc for tc in last.tool_calls if tc["name"] == "dispatch_subagent_lg"]
    dangerous_calls = [tc for tc in last.tool_calls if tc["name"] in _DANGEROUS_TOOLS]
    normal_calls = [tc for tc in last.tool_calls if tc["name"] not in _DANGEROUS_TOOLS and tc["name"] != "dispatch_subagent_lg"]

    # 子代理调用优先：走 subagent_dispatcher 并行派遣
    if subagent_calls:
        return "subagent_dispatcher"

    # 危险工具：走 interrupt_approval 安全审批
    if dangerous_calls and not normal_calls:
        return "interrupt_approval"

    # 混合场景（危险 + 普通）：走 interrupt_approval，
    # 审批通过后由 tools 节点统一执行
    if dangerous_calls and normal_calls:
        return "interrupt_approval"

    # 普通工具调用
    return "tools"


def _subagent_dispatcher(state: AgentState) -> list[Send]:
    """将 LLM 发出的多个 dispatch_subagent_lg 调用转为并行 Send。

    每个 Send 触发一个独立的 subagent_worker 节点实例，
    在同一 superstep 内并行执行。
    """
    last = state["messages"][-1]
    sends = []
    for tc in last.tool_calls:
        if tc["name"] == "dispatch_subagent_lg":
            sends.append(Send(
                "subagent_worker",
                {
                    "agent_type": tc["args"]["agent_type"],
                    "task": tc["args"]["task"],
                    "messages": [],
                },
            ))
    return sends


def _subagent_worker(state: SubagentWorkerState, config: RunnableConfig) -> dict:
    """执行单个子代理任务，返回结果写入 subagent_results。

    通过 ContextVar 获取 LLM 和子代理注册表，
    调用 get_subagent_graph 创建/获取缓存的子图并执行。
    """
    from agent.lc_tools import _ctx_llm_ref, _ctx_subagent_registry
    from agent_by_langgraph.lg_subagent import get_subagent_graph

    registry = _ctx_subagent_registry.get()
    llm = _ctx_llm_ref.get()
    agent_type = state["agent_type"]
    task = state["task"]

    if registry is None:
        return {"subagent_results": [f"Error: Subagent registry not initialized"]}
    if llm is None:
        return {"subagent_results": [f"Error: LLM not initialized"]}

    spec = registry.get(agent_type)
    if spec is None:
        available = ", ".join(registry.names())
        return {"subagent_results": [f"Error: unknown subagent '{agent_type}'. Available: {available}"]}

    print(f"\n[LG 并行子代理 · {agent_type}]: {task[:80]}")

    try:
        subgraph = get_subagent_graph(llm, registry, agent_type)
        result = subgraph.invoke({
            "input": task,
            "turns_remaining": spec.max_turns,
            "max_turns": spec.max_turns,
            "messages": [],
        })
    except Exception as exc:
        return {"subagent_results": [f"Error: subagent '{agent_type}' raised: {exc}"]}

    # 提取最后一条 AIMessage 的文本
    last_text = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            content = msg.content
            last_text = content if isinstance(content, str) else str(content)
            break
    print(f"[LG 子代理汇报 · {agent_type}]: {last_text[:200]}")
    if last_text:
        return {"subagent_results": [f"[{agent_type}] {last_text}"]}
    return {"subagent_results": [f"[{agent_type}] [子代理未产出任何回复]"]}


def _aggregate_results(state: AgentState) -> dict:
    """将并行子代理的结果合并为一条 AIMessage，注入主对话流。

    消费 subagent_results 后清空，避免下一轮循环重复消费。
    """
    results = state.get("subagent_results", [])
    if not results:
        return {"messages": [], "subagent_results": []}

    # 合并所有子代理汇报为一条消息
    combined = "\n\n---\n\n".join(results)
    summary_msg = AIMessage(content=f"[子代理并行汇报]\n\n{combined}")

    # 清空 subagent_results（已消费），追加汇总消息
    return {
        "messages": [summary_msg],
        "subagent_results": [],
    }


def _interrupt_approval(state: AgentState) -> dict:
    """危险操作审批门：暂停执行，等待人工确认。

    当 LLM 发出危险工具调用（write_file、edit_file、run_command）时，
    interrupt() 暂停图执行，返回审批请求给调用方。
    调用方通过 Command(resume=...) 恢复执行：
    - "approve": 放行，交给 tools 节点执行
    - "reject": 拒绝，注入拒绝消息

    返回值包含 __next__ 字段，指示下一个节点：
    - "tools": 审批通过，执行工具
    - "agent": 审批拒绝，回到 agent
    """
    last = state["messages"][-1]
    dangerous_calls = [
        tc for tc in last.tool_calls if tc["name"] in _DANGEROUS_TOOLS
    ]

    # 构建审批请求
    approval_request = {
        "type": "dangerous_tool_approval",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
            for tc in dangerous_calls
        ],
    }

    # 暂停执行，返回审批请求给调用方
    decision = interrupt(approval_request)

    if decision == "approve":
        # 放行：tools 节点会正常执行这些 tool_calls
        return {"messages": [], "_approval_next": "tools"}
    else:
        # 拒绝：注入拒绝消息
        tool_names = ", ".join(tc["name"] for tc in dangerous_calls)
        reject_msg = AIMessage(
            content=f"[人工审批] 以下操作已被拒绝: {tool_names}"
        )
        return {"messages": [reject_msg], "_approval_next": "agent"}


def _route_after_approval(state: AgentState) -> str:
    """interrupt_approval 后路由：根据审批结果决定下一步。"""
    return state.get("_approval_next", "agent")


def create_agent_graph(
    llm, tools, system_prompt: str,
    llm_callbacks: list | None = None,
    checkpointer=None,
):
    """构造并编译主 Agent 的 StateGraph。

    图结构：
        START → agent → route_after_agent
                            ├── 无 tool_calls → END
                            ├── dispatch_subagent_lg → subagent_dispatcher
                            │     └── [Send × N] → subagent_worker → aggregate_results → agent
                            ├── 危险工具 → interrupt_approval
                            │     ├── approve → tools → agent
                            │     └── reject → agent
                            └── 普通工具 → tools → agent

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
        - interrupt() 需要 checkpointer 才能工作（状态需持久化才能恢复）
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ParallelToolNode(tools)

    async def call_agent(state: AgentState, config: RunnableConfig) -> dict:
        # state["messages"] 已包含完整上下文：
        #   [SystemMessage, ...chat_history, HumanMessage(input), AIMessage, ToolMessage, ...]
        # 多轮 agent→tools 循环时，messages 通过 add_messages reducer 自动累积，
        # 无需重复拼接 chat_history 和 input
        response = await llm_with_tools.ainvoke(state["messages"], config=config)
        return {"messages": [response]}

    builder = StateGraph(AgentState)

    # 节点
    builder.add_node("agent", call_agent)
    builder.add_node("tools", tool_node)
    builder.add_node("subagent_dispatcher", _subagent_dispatcher)
    builder.add_node("subagent_worker", _subagent_worker)
    builder.add_node("aggregate_results", _aggregate_results)
    builder.add_node("interrupt_approval", _interrupt_approval)

    # 边
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route_after_agent)
    builder.add_edge("tools", "agent")
    builder.add_edge("subagent_worker", "aggregate_results")
    builder.add_edge("aggregate_results", "agent")
    builder.add_conditional_edges("interrupt_approval", _route_after_approval)
    builder.add_edge("interrupt_approval", "agent")

    compiled = builder.compile(checkpointer=checkpointer)
    # 保存 llm_callbacks 供调用方在 invoke 时合并到 config.callbacks
    compiled._lg_llm_callbacks = list(llm_callbacks) if llm_callbacks else []
    return compiled
