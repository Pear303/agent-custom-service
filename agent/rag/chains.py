"""CRAG 链 — 查询改写 + 文档评估 + 自纠正。

Corrective RAG (CRAG) 核心逻辑：
1. 查询改写：将客户模糊需求改写为精确检索查询
2. 文档评估：用 LLM 判断检索文档与查询的相关性
3. 自纠正：文档不相关时回退到 LLM 知识 + web_search

这些函数供 lg_rag_subagent.py 的 LangGraph 节点调用。
"""
from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ── 1. 查询改写 ──────────────────────────────────────────────

_REWRITE_PROMPT = """\
你是一个查询改写器。将客户的模糊需求描述改写为精确的检索查询。

改写规则：
1. 提取核心概念和关键词
2. 补充同义词和相关术语
3. 移除口语化表达，保留专业术语
4. 输出 2-3 个不同角度的检索查询，每行一个
5. 不要输出任何解释，只输出改写后的查询

示例：
输入："我想做一个电商小程序"
电商小程序 功能模块 需求分析
在线商城 技术架构 用户流程
购物车 支付集成 订单管理"""


async def rewrite_query(llm, query: str) -> list[str]:
    """将模糊查询改写为多个精确检索查询。

    Args:
        llm: LLM 实例（用于改写推理）
        query: 原始查询文本

    Returns:
        改写后的查询列表（2-3 个）
    """
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=f"输入：{query}"),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        # 最多 3 个查询
        queries = lines[:3]
        if not queries:
            queries = [query]
        logger.info("[CRAG 查询改写] '%s' → %s", query[:50], queries)
        return queries
    except Exception as exc:
        logger.warning("[CRAG 查询改写] 失败，使用原始查询: %s", exc)
        return [query]


# ── 2. 文档评估 ──────────────────────────────────────────────

_GRADE_PROMPT = """\
你是一个文档相关性评估器。判断检索到的文档是否与用户查询相关。

评估标准：
- "relevant"：文档直接回答了查询，或提供了关键参考信息
- "irrelevant"：文档与查询无关，或信息过于泛化无法使用

请只回答 "relevant" 或 "irrelevant"，不要输出其他内容。"""


async def grade_document(llm, query: str, doc: Document) -> bool:
    """评估单个文档与查询的相关性。

    Args:
        llm: LLM 实例（用于评估推理）
        query: 用户查询
        doc: 待评估的文档

    Returns:
        True 表示相关，False 表示不相关
    """
    try:
        # 截断过长文档，避免 token 浪费
        content = doc.page_content[:1000]
        user_msg = f"用户查询：{query}\n\n文档内容：{content}\n\n请评估。"
        response = await llm.ainvoke([
            SystemMessage(content=_GRADE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        answer = response.content.strip().lower() if isinstance(response.content, str) else ""
        # 使用正则匹配，避免 "irrelevant" 中的 "relevant" 子串误判
        import re
        is_relevant = bool(re.match(r"^relevant\b", answer))
        return is_relevant
    except Exception as exc:
        logger.warning("[CRAG 文档评估] 评估失败，默认保留: %s", exc)
        return True  # 评估失败时默认保留，避免误删


async def grade_documents(
    llm,
    query: str,
    docs: list[Document],
) -> list[Document]:
    """批量评估文档相关性，过滤不相关文档。

    使用 asyncio.gather 并行评估所有文档，避免串行等待。

    Args:
        llm: LLM 实例
        query: 用户查询
        docs: 待评估的文档列表

    Returns:
        评估后保留的相关文档列表
    """
    if not docs:
        return []

    import asyncio
    results = await asyncio.gather(
        *[grade_document(llm, query, doc) for doc in docs]
    )

    graded = [doc for doc, is_relevant in zip(docs, results) if is_relevant]

    logger.info(
        "[CRAG 文档评估] %d → %d (过滤 %d 不相关)",
        len(docs), len(graded), len(docs) - len(graded),
    )
    return graded


# ── 3. 多查询检索 + Rerank ──────────────────────────────────

async def multi_query_retrieve(
    queries: list[str],
    original_query: str,
    top_k_per_query: int = 10,
    final_top_k: int = 10,
) -> list[Document]:
    """多查询检索 + 去重 + Rerank。

    对每个改写查询分别检索，合并去重后 Rerank 精排。

    Args:
        queries: 改写后的查询列表
        original_query: 原始查询（用于 Rerank）
        top_k_per_query: 每个查询的粗检索数量
        final_top_k: 最终返回的文档数量

    Returns:
        Rerank 后的文档列表
    """
    import asyncio
    from .vectorstore import get_vectorstore
    from .reranker import rerank_documents

    vectorstore = get_vectorstore()
    all_docs: list[Document] = []
    seen_contents: set[str] = set()

    # 并发检索所有查询
    async def _search_one(q: str) -> list[Document]:
        return await asyncio.to_thread(vectorstore.similarity_search, q, k=top_k_per_query)

    per_query_results = await asyncio.gather(*[_search_one(q) for q in queries])

    for docs in per_query_results:
        for doc in docs:
            content_key = doc.page_content[:200]
            if content_key not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(content_key)

    if not all_docs:
        return []

    logger.info("[CRAG 多查询检索] 合并去重: %d 条", len(all_docs))

    # Rerank 精排（CPU 密集，包装为 to_thread 避免阻塞事件循环）
    ranked = await asyncio.to_thread(rerank_documents, original_query, all_docs, top_k=final_top_k)
    return ranked


# ── 4. 格式化上下文 ──────────────────────────────────────────

def format_rag_context(docs: list[Document]) -> str:
    """将检索文档格式化为注入 agent 的上下文字符串。

    Args:
        docs: 检索到的文档列表

    Returns:
        格式化的上下文字符串
    """
    if not docs:
        return ""

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{i}] (来源: {source})\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
