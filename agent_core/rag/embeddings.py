"""Embedding 模型初始化 — 加载本地 bge-small-zh-v1.5。

从 agent.rag.embeddings 迁移，更新路径: _MODELS_DIR 指向 agent/embeddings/models。
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# 复用已有的模型缓存目录（仍在 agent/embeddings/models 下）
_MODELS_DIR = Path(__file__).parent.parent.parent / "agent" / "embeddings" / "models"

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_MODELS_DIR))

_embeddings_instance: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型单例（线程安全）。"""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    with _embeddings_lock:
        if _embeddings_instance is not None:
            return _embeddings_instance

        logger.info("[RAG Embeddings] 加载模型: %s (缓存: %s)", _MODEL_NAME, _MODELS_DIR)

        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("[RAG Embeddings] 模型加载完成")

    return _embeddings_instance
