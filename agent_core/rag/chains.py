"""CRAG 链 — 查询改写 + 文档评估 + 自纠正。

从 agent.rag.chains 迁移，更新 import: from .vectorstore / .reranker
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


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
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_REWRITE_PROMPT),
            HumanMessage(content=f"输入：{query}"),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        queries = lines[:3]
        if not queries:
            queries = [query]
        logger.info("[CRAG 查询改写] '%s' → %s", query[:50], queries)
        return queries
    except Exception as exc:
        logger.warning("[CRAG 查询改写] 失败，使用原始查询: %s", exc)
        return [query]


_GRADE_PROMPT = """\
你是一个文档相关性评估器。判断检索到的文档是否与用户查询相关。

评估标准：
- "relevant"：文档直接回答了查询，或提供了关键参考信息
- "irrelevant"：文档与查询无关，或信息过于泛化无法使用

对每个文档，只回答 "relevant" 或 "irrelevant"。"""


_BATCH_GRADE_PROMPT = """\
你是一个文档相关性评估器。判断以下每个文档是否与用户查询相关。

评估标准：
- "relevant"：文档直接回答了查询，或提供了关键参考信息
- "irrelevant"：文档与查询无关，或信息过于泛化无法使用

请按以下 JSON 格式输出，不要输出其他内容：
[{"index": 0, "relevance": "relevant"}, {"index": 1, "relevance": "irrelevant"}, ...]"""


async def grade_documents(
    llm,
    query: str,
    docs: list[Document],
) -> list[Document]:
    if not docs:
        return []

    import json

    doc_sections = []
    for i, doc in enumerate(docs):
        content = doc.page_content[:1000]
        doc_sections.append(f"[文档 {i}]\n{content}")

    user_msg = (
        f"用户查询：{query}\n\n"
        + "\n\n".join(doc_sections)
        + "\n\n请评估每个文档与查询的相关性。"
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=_BATCH_GRADE_PROMPT),
            HumanMessage(content=user_msg),
        ])
        content = response.content if isinstance(response.content, str) else str(response.content)

        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            results = json.loads(json_match.group())
            relevant_indices = set()
            for item in results:
                idx = item.get("index")
                rel = item.get("relevance", "").strip().lower()
                if idx is not None and re.match(r"^relevant\b", rel):
                    relevant_indices.add(idx)
            graded = [doc for i, doc in enumerate(docs) if i in relevant_indices]
        else:
            logger.warning("[CRAG 文档评估] 批量评估 JSON 解析失败，回退逐文档评估")
            graded = await _grade_documents_fallback(llm, query, docs)

    except Exception as exc:
        logger.warning("[CRAG 文档评估] 批量评估失败，回退逐文档评估: %s", exc)
        graded = await _grade_documents_fallback(llm, query, docs)

    logger.info(
        "[CRAG 文档评估] %d → %d (过滤 %d 不相关)",
        len(docs), len(graded), len(docs) - len(graded),
    )
    return graded


async def _grade_documents_fallback(
    llm,
    query: str,
    docs: list[Document],
) -> list[Document]:
    import asyncio

    async def _grade_one(doc: Document) -> bool:
        try:
            content = doc.page_content[:1000]
            user_msg = f"用户查询：{query}\n\n文档内容：{content}\n\n请评估。"
            response = await llm.ainvoke([
                SystemMessage(content=_GRADE_PROMPT),
                HumanMessage(content=user_msg),
            ])
            answer = response.content.strip().lower() if isinstance(response.content, str) else ""
            return bool(re.match(r"^relevant\b", answer))
        except Exception as exc:
            logger.warning("[CRAG 文档评估] 单文档评估失败，默认保留: %s", exc)
            return True

    results = await asyncio.gather(*[_grade_one(doc) for doc in docs])
    return [doc for doc, is_relevant in zip(docs, results) if is_relevant]


async def multi_query_retrieve(
    queries: list[str],
    original_query: str,
    top_k_per_query: int = 10,
    final_top_k: int = 10,
) -> list[Document]:
    import asyncio
    from .vectorstore import get_vectorstore
    from .reranker import rerank_documents

    vectorstore = get_vectorstore()
    all_docs: list[Document] = []
    seen_contents: set[str] = set()

    async def _search_one(q: str) -> list[Document]:
        return await asyncio.to_thread(vectorstore.similarity_search, q, k=top_k_per_query)

    per_query_results = await asyncio.gather(*[_search_one(q) for q in queries])

    for docs in per_query_results:
        for doc in docs:
            content_key = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_key not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(content_key)

    if not all_docs:
        return []

    logger.info("[CRAG 多查询检索] 合并去重: %d 条", len(all_docs))

    ranked = await asyncio.to_thread(rerank_documents, original_query, all_docs, top_k=final_top_k)
    return ranked


def format_rag_context(docs: list[Document]) -> str:
    if not docs:
        return ""

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{i}] (来源: {source})\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)
