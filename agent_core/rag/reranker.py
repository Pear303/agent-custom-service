"""Reranker — 使用 HuggingFaceCrossEncoder 对检索结果精排。

从 agent.rag.reranker 迁移，更新路径: _MODELS_DIR 指向 agent/embeddings/models。
修复了 git merge 冲突标记。
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

_RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"

_MODELS_DIR = Path(__file__).parent.parent.parent / "agent" / "embeddings" / "models"

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

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
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
    """对检索结果进行 Rerank 精排。"""
    if not docs:
        return []

    if len(docs) <= top_k:
        return docs

    reranker = _get_reranker()

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.score(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    logger.info(
        "[RAG Reranker] 精排: %d → %d (最高分: %.4f, 最低分: %.4f)",
        len(docs), top_k, scored[0][1], scored[-1][1],
    )

    return [doc for doc, _ in scored[:top_k]]
