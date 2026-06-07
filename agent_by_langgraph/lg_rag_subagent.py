"""requirement_analyst CRAG 子图 — 带查询改写 + 文档评估 + 自纠正的 RAG。

图结构：
    START → _should_retrieve
                ├── 首轮（无 rag_context）→ query_rewriter → retrieve → grade_documents
                │                                                        ├── 有相关文档 → agent
                │                                                        └── 无相关文档 → web_fallback → agent
                └── 后续轮次（有 rag_context）→ agent
                    agent → _route_after_agent
                        ├── 有 tool_calls → tools → post_tools → _route_after_tools
                        │                                       ├── 还有轮数 → agent
                        │                                       └── 归零 → END
                        └── 无 tool_calls → END

与普通子图（lg_subagent.py）的区别：
- 首轮自动触发 RAG 检索链：查询改写 → 多查询检索 → Rerank → 文档评估
- 文档评估后注入 rag_context 到 agent 的 SystemMessage
- 文档不相关时回退到 web_fallback（LLM 知识 + web_search）
- 后续轮次走普通 agent→tools 循环（保留 rag_search 工具供主动检索）
"""
from __future__ import annotations

import logging
from typing import Annotated, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages

from agent.subagents.spec import SubagentSpec
from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

logger = logging.getLogger(__name__)


class RAGSubagentState(TypedDict):
    """CRAG 子图状态。

    继承普通子图字段，新增 RAG 专用字段。

    Attributes:
        messages: 消息序列
        input: 子代理要执行的原始任务
        turns_remaining: 剩余 agent→tools 循环轮数
        max_turns: 起始上限
        rewritten_queries: 查询改写后的查询列表
        retrieved_docs: 检索到的文档（序列化为 dict）
        rag_context: 注入 agent 的 RAG 上下文
        needs_web_fallback: 是否需要 web 回退
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    input: str
    turns_remaining: int
    max_turns: int
    rewritten_queries: list[str]
    retrieved_docs: list[dict]
    rag_context: str
    needs_web_fallback: bool


# ── 路由函数 ──────────────────────────────────────────────────

def _should_retrieve(state: RAGSubagentState) -> str:
    """首轮走 RAG 检索，后续轮次走 agent 循环。

    判断依据：rag_context 是否已有值。
    - 首轮：rag_context 为空 → 走 query_rewriter
    - 后续：rag_context 已注入 → 走 agent
    """
    if state.get("rag_context"):
        return "agent"
    return "query_rewriter"


def _route_after_grade(state: RAGSubagentState) -> str:
    """文档评估后路由：有相关文档 → agent，无相关文档 → web_fallback。"""
    if state.get("needs_web_fallback"):
        return "web_fallback"
    return "agent"


def _route_after_agent(state: RAGSubagentState) -> str:
    """agent 节点后路由：有 tool_calls 走 tools，否则 END。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _route_after_tools(state: RAGSubagentState) -> str:
    """post_tools 后路由：还有轮数回 agent，否则 END。"""
    if state["turns_remaining"] <= 0:
        return END
    return "agent"


# ── CRAG 节点（无状态，不依赖 LLM 的放在模块级）───────────────

async def _web_fallback(state: RAGSubagentState, config: RunnableConfig) -> dict:
    """Web 回退节点：知识库无相关文档时，依赖 LLM 知识 + web_search。"""
    fallback_context = (
        "[知识库未找到相关文档，请基于你的专业知识进行分析。"
        "如果需要补充信息，可以使用 web_fetch 工具搜索相关资料。]"
    )
    logger.info("[CRAG web_fallback] 知识库无相关文档，回退到 LLM 知识 + web_search")
    return {"rag_context": fallback_context, "needs_web_fallback": False}


def _post_tools(state: RAGSubagentState) -> dict:
    """tools 节点之后：递减剩余轮数；归零时注入终止消息。

    轮数耗尽时补全未响应的 tool_calls，防止消息序列不合法导致 API 400。
    """
    remaining = state["turns_remaining"] - 1
    if remaining <= 0:
        new_messages = []
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                new_messages.append(ToolMessage(
                    content="[轮数耗尽，工具调用被截断]",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    status="error",
                ))
        new_messages.append(AIMessage(content="[子代理已达最大轮数限制，任务可能未完成]"))
        return {
            "turns_remaining": 0,
            "messages": new_messages,
        }
    return {"turns_remaining": remaining}


def _ensure_rag_message_integrity(msgs: list[BaseMessage]) -> list[BaseMessage]:
    """确保消息序列完整性：AIMessage(tool_calls) 后必须紧跟所有对应的 ToolMessage。"""
    if not msgs:
        return msgs

    result = list(msgs)
    i = 0
    while i < len(result):
        msg = result[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            expected_ids = {tc["id"] for tc in msg.tool_calls}
            responded_ids = set()
            j = i + 1
            while j < len(result) and isinstance(result[j], ToolMessage):
                responded_ids.add(result[j].tool_call_id)
                j += 1
            missing_ids = expected_ids - responded_ids
            if missing_ids:
                id_to_name = {tc["id"]: tc["name"] for tc in msg.tool_calls}
                patch = []
                for mid in missing_ids:
                    patch.append(ToolMessage(
                        content="[工具调用被截断，响应缺失]",
                        tool_call_id=mid,
                        name=id_to_name.get(mid, "unknown"),
                        status="error",
                    ))
                    logger.warning(
                        "[CRAG] 补全缺失 ToolMessage: tool_call_id=%s, name=%s",
                        mid, id_to_name.get(mid, "unknown"),
                    )
                for k, pm in enumerate(patch):
                    result.insert(j + k, pm)
                i += len(patch)
        i += 1
    return result


# ── 子图构建 ──────────────────────────────────────────────────

def create_rag_subagent_graph(
    llm,
    spec: SubagentSpec,
    checkpointer=None,
):
    """创建 requirement_analyst 的 CRAG 子图。

    Args:
        llm: DeepSeekChatOpenAI 实例
        spec: SubagentSpec 子代理规格
        checkpointer: 可选的 LangGraph Checkpointer

    Returns:
        CompiledGraph
    """
    from agent_by_langgraph.lg_tools import make_subagent_tools
    from agent.rag.chains import rewrite_query, multi_query_retrieve, grade_documents, format_rag_context

    # 构建工具列表（make_subagent_tools 已包含 rag_search）
    tools = make_subagent_tools(spec.tool_names)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ParallelToolNode(tools)

    # ── CRAG 闭包节点（捕获 llm，不依赖 ContextVar）───────────

    async def query_rewriter(state: RAGSubagentState, config: RunnableConfig) -> dict:
        """查询改写节点：将模糊需求改写为精确检索查询。"""
        queries = await rewrite_query(llm, state["input"])
        return {"rewritten_queries": queries}

    async def retrieve(state: RAGSubagentState, config: RunnableConfig) -> dict:
        """多查询检索 + Rerank 节点。"""
        queries = state.get("rewritten_queries", [state["input"]])
        original_query = state["input"]

        try:
            docs = await multi_query_retrieve(
                queries=queries,
                original_query=original_query,
                top_k_per_query=10,
                final_top_k=10,
            )
        except Exception as exc:
            logger.error("[CRAG retrieve] 检索失败: %s", exc)
            docs = []

        # 序列化为 dict 存储（LangGraph state 要求可序列化）
        serialized = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

        logger.info("[CRAG retrieve] 检索到 %d 条文档", len(docs))
        return {"retrieved_docs": serialized}

    async def grade_docs(state: RAGSubagentState, config: RunnableConfig) -> dict:
        """文档评估节点：用 LLM 判断每个检索文档与查询的相关性。"""
        query = state["input"]

        # 反序列化文档
        docs = [
            Document(page_content=d["content"], metadata=d["metadata"])
            for d in state.get("retrieved_docs", [])
        ]

        if not docs:
            logger.warning("[CRAG grade] 无文档可评估，标记需 web 回退")
            return {"rag_context": "", "needs_web_fallback": True}

        # 评估文档相关性
        graded = await grade_documents(llm, query, docs)

        if graded:
            context = format_rag_context(graded)
            logger.info("[CRAG grade] 保留 %d 条相关文档", len(graded))
            return {"rag_context": context, "needs_web_fallback": False}
        else:
            logger.warning("[CRAG grade] 所有文档不相关，标记需 web 回退")
            return {"rag_context": "", "needs_web_fallback": True}

    async def call_rag_agent(state: RAGSubagentState, config: RunnableConfig) -> dict:
        """带 RAG 上下文的 agent 节点。

        首轮：注入 SystemMessage + RAG 上下文 + HumanMessage
        后续：使用已有消息序列
        """
        existing = list(state.get("messages", []))
        if not existing:
            msgs = [SystemMessage(content=spec.system_prompt)]
            # 注入 RAG 上下文
            rag_ctx = state.get("rag_context", "")
            if rag_ctx:
                msgs.append(SystemMessage(
                    content=f"## 知识库检索结果\n\n以下是与你当前任务相关的知识库内容，请参考：\n\n{rag_ctx}"
                ))
            msgs.append(HumanMessage(content=state["input"]))
        else:
            msgs = existing

        # 消息完整性校验：确保 AIMessage(tool_calls) 后紧跟所有对应的 ToolMessage
        msgs = _ensure_rag_message_integrity(msgs)

        response = await llm_with_tools.ainvoke(msgs, config=config)
        return {"messages": [response]}

    builder = StateGraph(RAGSubagentState)

    # 添加节点
    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_docs)
    builder.add_node("web_fallback", _web_fallback)
    builder.add_node("agent", call_rag_agent)
    builder.add_node("tools", tool_node)
    builder.add_node("post_tools", _post_tools)

    # 边
    builder.add_conditional_edges(START, _should_retrieve)
    builder.add_edge("query_rewriter", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges("grade_documents", _route_after_grade)
    builder.add_edge("web_fallback", "agent")

    # agent → tools 循环
    builder.add_conditional_edges("agent", _route_after_agent)
    builder.add_edge("tools", "post_tools")
    builder.add_conditional_edges("post_tools", _route_after_tools)

    return builder.compile(checkpointer=checkpointer)
