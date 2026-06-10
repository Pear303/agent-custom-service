"""文档摄入 — 加载需求文档 → 分块 → 入库 ChromaDB。

从 agent.rag.indexer 迁移，更新 import: from .vectorstore import get_vectorstore
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
)


def _chunk_id(doc: Document, chunk_index: int) -> str:
    source = doc.metadata.get("source", "")
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
    return f"{source}#{chunk_index}#{content_hash}"


def index_documents(docs: list[Document]) -> list[str]:
    chunks = _SPLITTER.split_documents(docs)
    if not chunks:
        logger.warning("[RAG Indexer] 分块结果为空，跳过入库")
        return []

    logger.info("[RAG Indexer] 分块: %d 文档 → %d 块", len(docs), len(chunks))

    chunk_ids = [_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

    vectorstore = get_vectorstore()
    ids = vectorstore.add_documents(chunks, ids=chunk_ids)

    logger.info("[RAG Indexer] 入库完成: %d 块", len(ids))
    return ids


def index_from_directory(dir_path: str | Path) -> list[str]:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        logger.error("[RAG Indexer] 目录不存在: %s", dir_path)
        return []

    all_docs: list[Document] = []

    for ext in ("*.txt", "*.md"):
        for file_path in sorted(dir_path.rglob(ext)):
            try:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(file_path.relative_to(dir_path))
                all_docs.extend(docs)
                logger.info("[RAG Indexer] 加载: %s (%d 段)", file_path.name, len(docs))
            except Exception as exc:
                logger.warning("[RAG Indexer] 加载失败 %s: %s", file_path, exc)

    for file_path in sorted(dir_path.rglob("*.pdf")):
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(dir_path))
            all_docs.extend(docs)
            logger.info("[RAG Indexer] 加载: %s (%d 页)", file_path.name, len(docs))
        except Exception as exc:
            logger.warning("[RAG Indexer] 加载失败 %s: %s", file_path, exc)

    for file_path in sorted(dir_path.rglob("*.docx")):
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(dir_path))
            all_docs.extend(docs)
            logger.info("[RAG Indexer] 加载: %s (%d 段)", file_path.name, len(docs))
        except Exception as exc:
            logger.warning("[RAG Indexer] 加载失败 %s: %s", file_path, exc)

    if not all_docs:
        logger.warning("[RAG Indexer] 未找到任何文档: %s", dir_path)
        return []

    return index_documents(all_docs)


def index_knowledge_base() -> list[str]:
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    all_ids: list[str] = []

    req_dir = raw_dir / "需求分析"
    if req_dir.exists():
        ids = index_from_directory(req_dir)
        all_ids.extend(ids)

    for txt_file in sorted(raw_dir.glob("*.txt")):
        try:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file.name
            ids = index_documents(docs)
            all_ids.extend(ids)
        except Exception as exc:
            logger.warning("[RAG Indexer] 加载失败 %s: %s", txt_file, exc)

    logger.info("[RAG Indexer] 知识库索引完成: %d 块", len(all_ids))
    return all_ids
