"""ChromaDB 向量库管理 — 需求分析知识库的持久化存储。

从 agent.rag.vectorstore 迁移，更新 import: from .embeddings import get_embeddings
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from langchain_chroma import Chroma

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "requirement_knowledge"
_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "chroma" / _COLLECTION_NAME

_vectorstore_instance: Chroma | None = None
_vectorstore_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """获取 ChromaDB 向量库单例。"""
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    with _vectorstore_lock:
        if _vectorstore_instance is not None:
            return _vectorstore_instance

        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = get_embeddings()

        logger.info("[RAG VectorStore] 初始化 ChromaDB: %s", _PERSIST_DIR)

        _vectorstore_instance = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(_PERSIST_DIR),
        )

        try:
            count = _vectorstore_instance._collection.count()
        except AttributeError:
            count = -1
        logger.info("[RAG VectorStore] 已有文档块数: %d", count)

    return _vectorstore_instance


def reset_vectorstore(*, delete_data: bool = False) -> None:
    """重置向量库单例。"""
    global _vectorstore_instance
    with _vectorstore_lock:
        if delete_data and _PERSIST_DIR.exists():
            import shutil
            shutil.rmtree(_PERSIST_DIR)
            logger.info("[RAG VectorStore] 已删除持久化数据: %s", _PERSIST_DIR)
        _vectorstore_instance = None
