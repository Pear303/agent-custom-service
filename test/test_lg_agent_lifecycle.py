"""LGAgent 生命周期集成测试 — 验证 Agent 创建/缓存/重置/Checkpointer/资源清理。

需要 DEEPSEEK_API_KEY 环境变量。标记为 @pytest.mark.integration。
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Factory 缓存逻辑测试（无需 LLM） ──────────────────────────


class TestFactoryCache:
    """Agent 工厂缓存逻辑测试。"""

    def setup_method(self):
        from agent_by_langgraph import factory
        factory._agent_cache.clear()
        factory._agent_cache_timestamps.clear()

    def teardown_method(self):
        from agent_by_langgraph import factory
        factory._agent_cache.clear()
        factory._agent_cache_timestamps.clear()

    def _mock_create_agent(self, factory_module):
        """配置 mock 使 create_lg_agent 可运行。"""
        factory_module.LGAgent = MagicMock()
        mock_instance = MagicMock()
        mock_instance._first_turn = True
        mock_instance.checkpointer_ready = False
        mock_instance.will_have_checkpointer = False
        mock_instance.close = MagicMock()
        factory_module.LGAgent.return_value = mock_instance

        factory_module._get_llm = MagicMock()
        factory_module._get_skills_loader = MagicMock()
        factory_module._get_subagent_registry = MagicMock()

        os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
        return mock_instance

    def test_cache_key_format(self):
        """缓存 key 格式为 user_id:ticket_id。"""
        from agent_by_langgraph import factory
        mock = self._mock_create_agent(factory)

        from agent_by_langgraph.factory import create_lg_agent
        create_lg_agent("user1", "ticket1")
        assert "user1:ticket1" in factory._agent_cache

    def test_cache_eviction_on_max_size(self):
        """超过 _MAX_CACHE_SIZE 时 LRU 淘汰。"""
        from agent_by_langgraph import factory
        from agent_by_langgraph.factory import _MAX_CACHE_SIZE, create_lg_agent
        mock = self._mock_create_agent(factory)

        for i in range(_MAX_CACHE_SIZE + 5):
            create_lg_agent(f"user_{i}", f"ticket_{i}")

        assert len(factory._agent_cache) <= _MAX_CACHE_SIZE

    def test_reset_lg_agent(self):
        """重置指定用户的 Agent 缓存。"""
        from agent_by_langgraph import factory
        mock = self._mock_create_agent(factory)

        from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
        create_lg_agent("user_reset_test", "t1")
        assert "user_reset_test:t1" in factory._agent_cache

        reset_lg_agent("user_reset_test", "t1")
        assert "user_reset_test:t1" not in factory._agent_cache

    def test_cache_hit_returns_same_instance(self):
        """同一 user_id 第二次调用返回缓存实例。"""
        from agent_by_langgraph import factory
        mock = self._mock_create_agent(factory)

        from agent_by_langgraph.factory import create_lg_agent
        a1 = create_lg_agent("user_cache_hit", "t1")
        a2 = create_lg_agent("user_cache_hit", "t1")
        assert a1 is a2


# ── LGAgent 属性测试 ──────────────────────────────────────────


class TestLGAgentProperties:
    """LGAgent 属性和方法测试（mock 初始化）。"""

    def _make_agent(self):
        """创建一个 mock 的 LGAgent 实例用于属性测试。

        不真正初始化 LGAgent（太重），而是直接构造一个简化版本。
        """
        from agent_by_langgraph.lg_agent import LGAgent

        agent = MagicMock(spec=LGAgent)
        agent._checkpointer_initialized = False
        agent._checkpointer_db_path = None
        agent.user_id = "test_user"
        agent.ticket_id = "test_ticket"
        agent._first_turn = True

        # graph mock — will_have_checkpointer 会访问 graph.checkpointer
        mock_graph = MagicMock()
        mock_graph.checkpointer = None
        agent.graph = mock_graph

        # 绑定真实 property
        agent.will_have_checkpointer = LGAgent.will_have_checkpointer.fget(agent)
        agent.checkpointer_ready = LGAgent.checkpointer_ready.fget(agent)

        return agent

    def test_will_have_checkpointer_false_when_no_db_path(self):
        """无 checkpointer db_path 时 will_have_checkpointer 为 False。"""
        agent = self._make_agent()
        assert agent.will_have_checkpointer is False

    def test_will_have_checkpointer_true_with_db_path(self):
        """有 checkpointer db_path 时 will_have_checkpointer 为 True。"""
        from agent_by_langgraph.lg_agent import LGAgent
        agent = self._make_agent()
        agent._checkpointer_db_path = "/tmp/test.db"
        # 重新绑定 property 以读取更新后的属性
        agent.will_have_checkpointer = LGAgent.will_have_checkpointer.fget(agent)
        assert agent.will_have_checkpointer is True

    def test_checkpointer_ready_default_false(self):
        """checkpointer_ready 默认为 False。"""
        agent = self._make_agent()
        assert agent.checkpointer_ready is False

    def test_checkpointer_ready_true_when_initialized(self):
        """checkpointer_initialized 为 True 时 checkpointer_ready 为 True。"""
        from agent_by_langgraph.lg_agent import LGAgent
        agent = self._make_agent()
        agent._checkpointer_initialized = True
        agent.checkpointer_ready = LGAgent.checkpointer_ready.fget(agent)
        assert agent.checkpointer_ready is True


# ── ReasoningCollector 测试 ──────────────────────────────────────


class TestReasoningCollector:
    """ReasoningCollector 回调测试。"""

    def test_collects_ai_message(self):
        from agent_by_langgraph.lg_agent import ReasoningCollector
        from langchain_core.messages import AIMessage

        collector = ReasoningCollector()

        mock_response = MagicMock()
        mock_gen = MagicMock()
        mock_gen.message = AIMessage(content="test reply", additional_kwargs={"reasoning_content": "thinking..."})
        mock_response.generations = [[mock_gen]]

        collector.on_llm_end(mock_response)

        assert len(collector.ai_messages) == 1
        assert collector.last.content == "test reply"

    def test_last_returns_none_when_empty(self):
        from agent_by_langgraph.lg_agent import ReasoningCollector

        collector = ReasoningCollector()
        assert collector.last is None

    def test_multiple_messages(self):
        from agent_by_langgraph.lg_agent import ReasoningCollector
        from langchain_core.messages import AIMessage

        collector = ReasoningCollector()

        for i in range(3):
            mock_response = MagicMock()
            mock_gen = MagicMock()
            mock_gen.message = AIMessage(content=f"reply {i}")
            mock_response.generations = [[mock_gen]]
            collector.on_llm_end(mock_response)

        assert len(collector.ai_messages) == 3
        assert collector.last.content == "reply 2"

    def test_on_llm_end_with_exception_silenced(self):
        """on_llm_end 内部异常应被静默处理。"""
        from agent_by_langgraph.lg_agent import ReasoningCollector

        collector = ReasoningCollector()
        mock_response = MagicMock()
        mock_response.generations = [[]]  # 空生成列表

        # 不应抛异常
        collector.on_llm_end(mock_response)
        assert len(collector.ai_messages) == 0


# ── TokenTracker 测试 ──────────────────────────────────────────


class TestTokenTracker:
    """TokenTracker 测试。"""

    def test_record_raw(self):
        """record_raw 应记录 token 用量。"""
        from agent_core.telemetry import TokenTracker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = TokenTracker(log_file=Path(f.name))

        tracker.record_raw("test-model", 100, 50, 150)
        assert tracker._last_input_tokens == 100

        # 清理
        Path(f.name).unlink(missing_ok=True)

    def test_should_compact(self):
        """should_compact 在 token 使用超过阈值时返回 True。"""
        from agent_core.telemetry import TokenTracker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = TokenTracker(log_file=Path(f.name))

        # 默认应该不触发压缩
        assert tracker.should_compact(max_context=200_000, threshold=0.5) is False

        # 模拟大量 token 使用
        tracker.record_raw("test-model", 100_001, 50_000, 150_001)
        assert tracker.should_compact(max_context=200_000, threshold=0.5) is True

        Path(f.name).unlink(missing_ok=True)


class TestTokenTrackerCallback:
    """TokenTracker 回调测试。"""

    def test_records_token_usage_from_llm_output(self):
        """从 llm_output 记录 token 用量。"""
        from agent_by_langgraph.lg_agent import TokenTrackerCallback
        from agent_core.telemetry import TokenTracker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tracker = TokenTracker(log_file=Path(f.name))

        callback = TokenTrackerCallback(tracker, "test-model")

        mock_response = MagicMock()
        mock_response.llm_output = {
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        # generations 为空，走 llm_output 分支
        mock_response.generations = [[]]

        callback.on_llm_end(mock_response)
        assert tracker._last_input_tokens == 100

        Path(f.name).unlink(missing_ok=True)
