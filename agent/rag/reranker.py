"""Reranker — 使用 HuggingFaceCrossEncoder 对检索结果精排。

模型：BAAI/bge-reranker-large（与 bge embedding 配套）
流程：粗检索 Top-20 → rerank → 取 Top-K（默认 5）

通过 langchain_huggingface 的 HuggingFaceCrossEncoder 集成，
模型从本地缓存加载，避免重复下载。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

# Reranker 模型名称（与 bge embedding 配套）
_RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

# 复用已有的模型缓存目录
_MODELS_DIR = Path(__file__).parent.parent / "embeddings" / "models"

_reranker_instance: HuggingFaceCrossEncoder | None = None
_reranker_lock = threading.Lock()


def _get_reranker() -> HuggingFaceCrossEncoder:
    """获取 Reranker 模型单例（线程安全）。"""
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance

        import os
        os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_MODELS_DIR))

        logger.info("[RAG Reranker] 加载模型: %s", _RERANKER_MODEL_NAME)

        _reranker_instance = HuggingFaceCrossEncoder(
            model_name=_RERANKER_MODEL_NAME,
        )

        logger.info("[RAG Reranker] 模型加载完成")

    return _reranker_instance


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """对检索结果进行 Rerank 精排。

    Args:
        query: 用户查询文本
        docs: 粗检索返回的文档列表
        top_k: 返回的文档数量

    Returns:
        按 Rerank 分数降序排列的 Top-K 文档
    """
    if not docs:
        return []

    # 文档数量不足 top_k 时直接返回
    if len(docs) <= top_k:
        return docs

    reranker = _get_reranker()

    # 构造 (query, document) 对
    pairs = [(query, doc.page_content) for doc in docs]

    # 计算相关性分数
    scores = reranker.score(pairs)

    # 按分数降序排序
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    logger.info(
        "[RAG Reranker] 精排: %d → %d (最高分: %.4f, 最低分: %.4f)",
        len(docs), top_k, scored[0][1], scored[-1][1],
    )

    return [doc for doc, _ in scored[:top_k]]
