"""RAG 链路集成测试 — 验证 embedding→index→retrieve→rerank→answer 全链路。

需要本地模型文件（bge-small-zh-v1.5, bge-reranker-large），首次运行会下载。
标记为 @pytest.mark.rag，可单独运行或跳过。

缺少可选依赖（langchain_chroma, langchain_huggingface）时自动跳过。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── 依赖检查 ──────────────────────────────────────────────────

def _has_langchain_chroma():
    try:
        import langchain_chroma  # noqa: F401
        return True
    except ImportError:
        return False


def _has_huggingface_cross_encoder():
    try:
        from langchain_huggingface import HuggingFaceCrossEncoder  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


requires_chroma = pytest.mark.skipif(
    not _has_langchain_chroma(),
    reason="langchain_chroma not installed",
)

requires_reranker = pytest.mark.skipif(
    not _has_huggingface_cross_encoder(),
    reason="HuggingFaceCrossEncoder not available",
)


# ── 纯逻辑测试（无需模型） ──────────────────────────────────────


class TestIndexerLogic:
    """文档摄入逻辑测试。"""

    @requires_chroma
    def test_chunk_id_deterministic(self):
        """相同内容产生相同 chunk_id。"""
        from agent_core.rag.indexer import _chunk_id

        doc = Document(page_content="测试内容", metadata={"source": "test.md"})
        id1 = _chunk_id(doc, 0)
        id2 = _chunk_id(doc, 0)
        assert id1 == id2

    @requires_chroma
    def test_chunk_id_different_for_different_content(self):
        """不同内容产生不同 chunk_id。"""
        from agent_core.rag.indexer import _chunk_id

        doc1 = Document(page_content="内容A", metadata={"source": "test.md"})
        doc2 = Document(page_content="内容B", metadata={"source": "test.md"})
        id1 = _chunk_id(doc1, 0)
        id2 = _chunk_id(doc2, 0)
        assert id1 != id2

    @requires_chroma
    def test_chunk_id_includes_index(self):
        """不同 chunk_index 产生不同 chunk_id。"""
        from agent_core.rag.indexer import _chunk_id

        doc = Document(page_content="测试内容", metadata={"source": "test.md"})
        id1 = _chunk_id(doc, 0)
        id2 = _chunk_id(doc, 1)
        assert id1 != id2

    @requires_chroma
    def test_index_from_nonexistent_directory(self):
        """索引不存在的目录返回空列表。"""
        from agent_core.rag.indexer import index_from_directory

        result = index_from_directory("/nonexistent/path/12345")
        assert result == []


class TestFormatRagContext:
    """RAG 上下文格式化测试。"""

    def test_empty_docs(self):
        from agent_core.rag.chains import format_rag_context

        assert format_rag_context([]) == ""

    def test_single_doc(self):
        from agent_core.rag.chains import format_rag_context

        docs = [Document(page_content="内容A", metadata={"source": "file1.md"})]
        result = format_rag_context(docs)
        assert "[1]" in result
        assert "内容A" in result
        assert "file1.md" in result

    def test_multiple_docs(self):
        from agent_core.rag.chains import format_rag_context

        docs = [
            Document(page_content="内容A", metadata={"source": "f1.md"}),
            Document(page_content="内容B", metadata={"source": "f2.md"}),
        ]
        result = format_rag_context(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "内容A" in result
        assert "内容B" in result


class TestRerankerLogic:
    """Reranker 逻辑测试（mock 模型）。"""

    @requires_reranker
    def test_rerank_empty_docs(self):
        from agent_core.rag.reranker import rerank_documents

        result = rerank_documents("query", [], top_k=5)
        assert result == []

    @requires_reranker
    def test_rerank_fewer_than_topk(self):
        """文档数少于 top_k 时全部返回。"""
        from agent_core.rag.reranker import rerank_documents

        docs = [Document(page_content="doc1")]
        with patch("agent_core.rag.reranker._get_reranker") as mock_get:
            result = rerank_documents("query", docs, top_k=5)
            assert result == docs
            mock_get.assert_not_called()


class TestVectorstoreSingleton:
    """向量库单例管理测试。"""

    @requires_chroma
    def test_reset_vectorstore(self):
        """reset_vectorstore 应将实例置为 None。"""
        from agent_core.rag.vectorstore import reset_vectorstore

        reset_vectorstore()
        from agent_core.rag import vectorstore
        assert vectorstore._vectorstore_instance is None


# ── 需要模型的集成测试 ──────────────────────────────────────────


@pytest.mark.rag
class TestRAGIntegration:
    """RAG 全链路集成测试（需要本地模型）。"""

    @pytest.mark.asyncio
    async def test_rewrite_query(self):
        """查询改写应返回查询列表。"""
        from agent_core.rag.chains import rewrite_query

        mock_llm = AsyncMock()
        from langchain_core.messages import AIMessage
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content="电商小程序 功能模块\n在线商城 技术架构\n购物车 支付集成"
        ))

        queries = await rewrite_query(mock_llm, "我想做一个电商小程序")
        assert len(queries) >= 1
        assert isinstance(queries[0], str)

    @pytest.mark.asyncio
    async def test_rewrite_query_fallback_on_error(self):
        """查询改写失败时回退到原始查询。"""
        from agent_core.rag.chains import rewrite_query

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        queries = await rewrite_query(mock_llm, "原始查询")
        assert queries == ["原始查询"]

    @pytest.mark.asyncio
    async def test_grade_documents_all_relevant(self):
        """文档评估：所有文档相关。"""
        from agent_core.rag.chains import grade_documents

        mock_llm = AsyncMock()
        from langchain_core.messages import AIMessage
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content='[{"index": 0, "relevance": "relevant"}, {"index": 1, "relevance": "relevant"}]'
        ))

        docs = [
            Document(page_content="相关文档1"),
            Document(page_content="相关文档2"),
        ]
        result = await grade_documents(mock_llm, "查询", docs)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_grade_documents_partial_relevant(self):
        """文档评估：部分文档相关。"""
        from agent_core.rag.chains import grade_documents

        mock_llm = AsyncMock()
        from langchain_core.messages import AIMessage
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content='[{"index": 0, "relevance": "relevant"}, {"index": 1, "relevance": "irrelevant"}]'
        ))

        docs = [
            Document(page_content="相关文档"),
            Document(page_content="不相关文档"),
        ]
        result = await grade_documents(mock_llm, "查询", docs)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_grade_documents_empty(self):
        """文档评估：空列表。"""
        from agent_core.rag.chains import grade_documents

        result = await grade_documents(AsyncMock(), "查询", [])
        assert result == []
