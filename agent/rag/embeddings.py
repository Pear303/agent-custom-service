"""Embedding 模型初始化 — 加载本地 bge-small-zh-v1.5。

复用 agent/embeddings/config.py 的模型路径配置，
通过 langchain_huggingface 的 HuggingFaceEmbeddings 加载。

bge 系列要求：
- normalize_embeddings=True（官方推荐，影响检索质量）
- bge-small-zh-v1.5 最大输入 512 tokens
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# 复用已有的模型缓存目录
_MODELS_DIR = Path(__file__).parent.parent / "embeddings" / "models"

# bge-small-zh-v1.5 的 HuggingFace Hub 快照路径
_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 确保 HF 环境变量指向本地缓存（与 agent/embeddings/config.py 一致）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(_MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_MODELS_DIR))

_embeddings_instance: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型单例（线程安全）。

    优先从本地缓存加载，避免重复下载。
    bge 系列要求 normalize_embeddings=True。
    """
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
