"""API 路由层单元测试 — mock 底层服务，验证路由参数校验、响应格式、错误处理。

使用 FastAPI TestClient，不启动真实服务，秒级完成。
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── 辅助：创建测试用 FastAPI 应用 ─────────────────────────────


def _create_test_app():
    """创建一个最小化的 FastAPI 应用用于路由测试。"""
    from fastapi import FastAPI
    from api.routers.chat import create_router as create_chat_router
    from api.routers.session import create_router as create_session_router
    from api.routers.health import create_router as create_health_router
    from api.routers.task import create_router as create_task_router

    app = FastAPI()

    # mock 服务层
    mock_agent_service = AsyncMock()
    mock_session_manager = AsyncMock()
    mock_dify_client = AsyncMock()
    mock_db = AsyncMock()
    mock_task_queue = AsyncMock()

    app.include_router(create_chat_router(mock_agent_service))
    app.include_router(create_session_router(mock_session_manager))
    app.include_router(create_health_router(mock_session_manager, mock_dify_client))
    app.include_router(create_task_router(mock_db, mock_agent_service, mock_task_queue))

    return app, {
        "agent_service": mock_agent_service,
        "session_manager": mock_session_manager,
        "dify_client": mock_dify_client,
        "db": mock_db,
        "task_queue": mock_task_queue,
    }


# ── Chat 路由测试 ──────────────────────────────────────────────


class TestChatRoutes:
    """聊天路由测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_test_app()
        self.client = TestClient(self.app)

    def test_chat_success(self):
        """POST /chat 正常请求。"""
        self.mocks["agent_service"].chat.return_value = {
            "user_id": "u1",
            "answer": "你好",
            "conversation_id": "conv1",
            "source": "dify",
        }
        resp = self.client.post("/chat", json={"user_id": "u1", "message": "你好"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["answer"] == "你好"
        assert data["source"] == "dify"

    def test_chat_missing_user_id(self):
        """POST /chat 缺少 user_id 应返回 422。"""
        resp = self.client.post("/chat", json={"message": "你好"})
        assert resp.status_code == 422

    def test_chat_missing_message(self):
        """POST /chat 缺少 message 应返回 422。"""
        resp = self.client.post("/chat", json={"user_id": "u1"})
        assert resp.status_code == 422

    def test_chat_empty_body(self):
        """POST /chat 空请求体应返回 422。"""
        resp = self.client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_stream_endpoint_exists(self):
        """POST /chat/stream 端点存在。"""
        # mock chat_stream 返回一个异步生成器
        async def fake_stream(*args, **kwargs):
            yield json.dumps({"event": "message", "answer": "hi", "source": "dify"}) + "\n"
            yield json.dumps({"event": "message_end", "conversation_id": "conv1"}) + "\n"

        self.mocks["agent_service"].chat_stream = fake_stream
        resp = self.client.post("/chat/stream", json={"user_id": "u1", "message": "你好"})
        # SSE 响应码应为 200
        assert resp.status_code == 200


# ── Session 路由测试 ────────────────────────────────────────────


class TestSessionRoutes:
    """会话管理路由测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_test_app()
        self.client = TestClient(self.app)

    def test_reset_session(self):
        """POST /session/reset 正常重置。"""
        self.mocks["session_manager"].reset_async = AsyncMock()
        resp = self.client.post("/session/reset", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["status"] == "reset"

    def test_get_session_history(self):
        """GET /session/history 正常获取。"""
        mock_session = MagicMock()
        mock_session.history = [{"role": "user", "content": "hi"}]
        mock_session.message_count = 1
        self.mocks["session_manager"].get_or_create = AsyncMock(return_value=mock_session)

        resp = self.client.get("/session/history", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert data["message_count"] == 1

    def test_reset_session_missing_user_id(self):
        """POST /session/reset 缺少 user_id 应返回 422。"""
        resp = self.client.post("/session/reset")
        assert resp.status_code == 422


# ── Health 路由测试 ─────────────────────────────────────────────


class TestHealthRoutes:
    """健康检查路由测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_test_app()
        self.client = TestClient(self.app)

    def test_status_endpoint(self):
        """GET /status 返回活跃会话数。"""
        # active_count 是同步方法，不能用 AsyncMock 的 return_value
        self.mocks["session_manager"].active_count = MagicMock(return_value=5)
        resp = self.client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_sessions"] == 5

    def test_health_endpoint_structure(self):
        """GET /health 返回标准健康检查结构。"""
        # mock dify 和 deepseek 检查
        self.mocks["session_manager"].active_count.return_value = 0

        # mock dify_client 的内部方法
        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response
        self.mocks["dify_client"]._get_http_client = AsyncMock(return_value=mock_http_client)
        self.mocks["dify_client"]._build_url.return_value = "http://dify.test/v1/parameters"
        self.mocks["dify_client"]._headers.return_value = {}

        # mock db.get_stats
        self.mocks["session_manager"].db = AsyncMock()
        self.mocks["session_manager"].db.get_stats = AsyncMock(return_value={"tickets": 0, "sessions": 0})

        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data


# ── Task 路由测试 ──────────────────────────────────────────────


class TestTaskRoutes:
    """工单路由测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_test_app()
        self.client = TestClient(self.app)

    def test_submit_requirement(self):
        """POST /task/submit 正常提交工单。"""
        self.mocks["db"].create_ticket = AsyncMock(return_value="TKT-TEST01")
        self.mocks["task_queue"].submit = AsyncMock()

        resp = self.client.post("/task/submit", json={
            "user_id": "u1",
            "project_name": "测试项目",
            "description": "这是一个测试",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_id"] == "TKT-TEST01"
        assert data["status"] == "queued"

    def test_submit_requirement_missing_fields(self):
        """POST /task/submit 缺少必填字段应返回 422。"""
        resp = self.client.post("/task/submit", json={"user_id": "u1"})
        assert resp.status_code == 422

    def test_get_ticket_status(self):
        """GET /task/{ticket_id}/status 正常查询。"""
        self.mocks["db"].get_ticket = AsyncMock(return_value={
            "ticket_id": "TKT-01",
            "user_id": "u1",
            "project_name": "测试",
            "status": "queued",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        })

        resp = self.client.get("/task/TKT-01/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_id"] == "TKT-01"
        assert "progress" in data

    def test_get_ticket_status_not_found(self):
        """GET /task/{ticket_id}/status 工单不存在。"""
        self.mocks["db"].get_ticket = AsyncMock(return_value=None)

        resp = self.client.get("/task/TKT-NONEXIST/status")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_list_tickets(self):
        """GET /task/list 正常查询。"""
        self.mocks["db"].get_user_tickets = AsyncMock(return_value=[])

        resp = self.client.get("/task/list", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "tickets" in data

    def test_start_development_invalid_status(self):
        """POST /task/{ticket_id}/start-development 状态不允许时拒绝。"""
        self.mocks["db"].get_ticket = AsyncMock(return_value={
            "ticket_id": "TKT-01",
            "user_id": "u1",
            "status": "queued",
        })

        resp = self.client.post("/task/TKT-01/start-development")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
