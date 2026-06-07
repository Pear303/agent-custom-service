"""ChromaDB 向量库管理 — 需求分析知识库的持久化存储。

集合命名约定：requirement_knowledge（需求分析专用）
持久化路径：data/chroma/requirement_knowledge/
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
    """获取 ChromaDB 向量库单例。

    首次调用时初始化，后续复用。
    如果持久化目录已有数据，自动加载已有索引。
    线程安全：使用 Lock 防止并发初始化。
    """
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    with _vectorstore_lock:
        # double-check：获取锁后再次检查
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

        # 检查已有文档数量
        try:
            count = _vectorstore_instance._collection.count()
        except AttributeError:
            count = -1
        logger.info("[RAG VectorStore] 已有文档块数: %d", count)

    return _vectorstore_instance


def reset_vectorstore(*, delete_data: bool = False) -> None:
    """重置向量库单例。

    Args:
        delete_data: 是否同时删除磁盘上的持久化数据。
            True 时删除 _PERSIST_DIR 目录，下次 get_vectorstore() 将创建空集合；
            False 时仅重置内存实例，磁盘数据保留，下次加载仍会读到旧数据。
    """
    global _vectorstore_instance
    with _vectorstore_lock:
        if delete_data and _PERSIST_DIR.exists():
            import shutil
            shutil.rmtree(_PERSIST_DIR)
            logger.info("[RAG VectorStore] 已删除持久化数据: %s", _PERSIST_DIR)
        _vectorstore_instance = None
