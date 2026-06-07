"""文档摄入 — 加载需求文档 → 分块 → 入库 ChromaDB。

分块策略：
- RecursiveCharacterTextSplitter，chunk_size=512, overlap=50
- 分隔符按中文语义层级递减：标题 → 段落 → 句号 → 分号 → 逗号
- chunk_size=512 适配 bge-small-zh-v1.5 的 512 token 上限

支持的文档格式：
- .txt / .md：TextLoader
- .pdf：PyPDFLoader
- .docx：Docx2txtLoader
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# 分块器：chunk_size=512 适配 bge-small-zh-v1.5，overlap=50 保证语义连续
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", ""],
)


def _chunk_id(doc: Document, chunk_index: int) -> str:
    """基于 source + 内容哈希 + chunk 索引生成唯一 ID，防止重复入库。"""
    source = doc.metadata.get("source", "")
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
    return f"{source}#{chunk_index}#{content_hash}"


def index_documents(docs: list[Document]) -> list[str]:
    """将文档分块后入库 ChromaDB。

    使用 source + 内容哈希 + chunk 索引作为唯一 ID，
    重复调用时相同 chunk 会覆盖而非重复插入。

    Args:
        docs: 已加载的 Document 列表

    Returns:
        入库的 chunk ID 列表
    """
    chunks = _SPLITTER.split_documents(docs)
    if not chunks:
        logger.warning("[RAG Indexer] 分块结果为空，跳过入库")
        return []

    logger.info("[RAG Indexer] 分块: %d 文档 → %d 块", len(docs), len(chunks))

    # 为每个 chunk 生成唯一 ID，防止重复入库
    chunk_ids = [_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]

    vectorstore = get_vectorstore()
    ids = vectorstore.add_documents(chunks, ids=chunk_ids)

    logger.info("[RAG Indexer] 入库完成: %d 块", len(ids))
    return ids


def index_from_directory(dir_path: str | Path) -> list[str]:
    """从目录批量加载文档并入库。

    递归扫描目录下的 .txt / .md / .pdf 文件。

    Args:
        dir_path: 文档目录路径

    Returns:
        入库的 chunk ID 列表
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        logger.error("[RAG Indexer] 目录不存在: %s", dir_path)
        return []

    all_docs: list[Document] = []

    # 加载 .txt 和 .md 文件
    for ext in ("*.txt", "*.md"):
        for file_path in sorted(dir_path.rglob(ext)):
            try:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                # 为每个文档添加来源元数据
                for doc in docs:
                    doc.metadata["source"] = str(file_path.relative_to(dir_path))
                all_docs.extend(docs)
                logger.info("[RAG Indexer] 加载: %s (%d 段)", file_path.name, len(docs))
            except Exception as exc:
                logger.warning("[RAG Indexer] 加载失败 %s: %s", file_path, exc)

    # 加载 .pdf 文件
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

    # 加载 .docx 文件
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
    """索引默认的需求分析知识库（data/raw/需求分析/）。

    Returns:
        入库的 chunk ID 列表
    """
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    all_ids: list[str] = []

    # 索引需求分析知识库
    req_dir = raw_dir / "需求分析"
    if req_dir.exists():
        ids = index_from_directory(req_dir)
        all_ids.extend(ids)

    # 索引根目录下的知识库文件
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
