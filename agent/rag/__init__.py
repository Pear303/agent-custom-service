"""RAG 模块 — 为 requirement_analyst 提供知识库检索能力。

核心组件：
- embeddings: Embedding 模型初始化（bge-small-zh-v1.5）
- vectorstore: ChromaDB 向量库管理
- indexer: 文档摄入 + 分块入库
- retriever: rag_search 工具封装
- reranker: HuggingFaceCrossEncoder 精排
- chains: CRAG 链（查询改写 + 文档评估 + 自纠正）
"""
from .retriever import rag_search

__all__ = ["rag_search"]
