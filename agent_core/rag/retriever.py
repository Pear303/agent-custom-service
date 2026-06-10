"""RAG 检索工具 — 封装为 LangChain Tool，供子代理调用。

从 agent.rag.retriever 迁移，更新 import: from .vectorstore / .reranker
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def rag_search(query: str, top_k: Optional[int] = 5) -> str:
    """从需求分析知识库中检索相关文档片段。

    适用于：查找历史需求模板、行业需求框架、技术方案参考、需求分析方法论。
    输入：查询文本 → 输出：Top-K 相关文档片段（经 Rerank 排序）

    Args:
        query: 查询文本，描述你想查找的需求分析知识
        top_k: 返回的文档数量，默认 5

    Returns:
        格式化的检索结果，每段包含来源和内容
    """
    from .vectorstore import get_vectorstore
    from .reranker import rerank_documents

    try:
        vectorstore = get_vectorstore()

        try:
            count = await asyncio.to_thread(vectorstore._collection.count)
        except AttributeError:
            count = -1
        if count == 0:
            return "[知识库为空，请先运行索引构建。提示：from agent_core.rag.indexer import index_knowledge_base; index_knowledge_base()]"

        raw_docs = await asyncio.to_thread(vectorstore.similarity_search, query, k=20)
        if not raw_docs:
            return "[知识库无相关文档]"

        logger.info("[rag_search] 粗检索: %d 条结果", len(raw_docs))

        ranked_docs = await asyncio.to_thread(rerank_documents, query, raw_docs, top_k=top_k or 5)

        results = []
        for i, doc in enumerate(ranked_docs, 1):
            source = doc.metadata.get("source", "未知来源")
            results.append(f"[{i}] (来源: {source})\n{doc.page_content}")

        output = "\n\n---\n\n".join(results)
        logger.info("[rag_search] 返回 %d 条精排结果", len(ranked_docs))
        return output

    except Exception as exc:
        logger.error("[rag_search] 检索失败: %s", exc)
        return f"[检索失败: {exc}]"
