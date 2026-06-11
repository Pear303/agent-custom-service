"""API 端到端测试 — 通过 HTTP 客户端调用 FastAPI 接口，验证完整请求-响应流。

使用 TestClient 模拟 HTTP 请求，mock 底层服务。
标记为 @pytest.mark.e2e。
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── 辅助 ──────────────────────────────────────────────────────


def _create_full_app():
    """创建完整的 FastAPI 应用（含所有路由），mock 所有外部依赖。"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from api.routers.chat import create_router as create_chat_router
    from api.routers.session import create_router as create_session_router
    from api.routers.health import create_router as create_health_router
    from api.routers.task import create_router as create_task_router

    app = FastAPI(title="Test API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # mock 所有服务
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


# ── 完整流程测试 ──────────────────────────────────────────────


@pytest.mark.e2e
class TestFullWorkflow:
    """端到端流程测试：从聊天到工单提交到状态查询。"""

    def setup_method(self):
        self.app, self.mocks = _create_full_app()
        self.client = TestClient(self.app)

    def test_chat_then_submit_task(self):
        """先聊天，再提交工单的完整流程。"""
        # 1. 聊天
        self.mocks["agent_service"].chat.return_value = {
            "user_id": "u1",
            "answer": "我可以帮你做需求分析",
            "conversation_id": "conv1",
            "source": "dify",
        }
        chat_resp = self.client.post("/chat", json={"user_id": "u1", "message": "帮我做需求分析"})
        assert chat_resp.status_code == 200
        assert chat_resp.json()["answer"] == "我可以帮你做需求分析"

        # 2. 提交工单
        self.mocks["db"].create_ticket = AsyncMock(return_value="TKT-E2E-01")
        self.mocks["task_queue"].submit = AsyncMock()
        task_resp = self.client.post("/task/submit", json={
            "user_id": "u1",
            "project_name": "E2E测试项目",
            "description": "端到端测试描述",
        })
        assert task_resp.status_code == 200
        assert task_resp.json()["ticket_id"] == "TKT-E2E-01"

        # 3. 查询工单状态
        self.mocks["db"].get_ticket = AsyncMock(return_value={
            "ticket_id": "TKT-E2E-01",
            "user_id": "u1",
            "project_name": "E2E测试项目",
            "status": "queued",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        })
        status_resp = self.client.get("/task/TKT-E2E-01/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "queued"

    def test_session_lifecycle(self):
        """会话生命周期：创建→查询→重置→查询。"""
        # 1. 聊天创建会话
        self.mocks["agent_service"].chat.return_value = {
            "user_id": "u2",
            "answer": "你好",
            "conversation_id": "conv2",
            "source": "dify",
        }
        self.client.post("/chat", json={"user_id": "u2", "message": "你好"})

        # 2. 查询会话历史
        mock_session = MagicMock()
        mock_session.history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好", "source": "dify"},
        ]
        mock_session.message_count = 2
        self.mocks["session_manager"].get_or_create = AsyncMock(return_value=mock_session)

        history_resp = self.client.get("/session/history", params={"user_id": "u2"})
        assert history_resp.status_code == 200
        assert history_resp.json()["message_count"] == 2

        # 3. 重置会话
        self.mocks["session_manager"].reset_async = AsyncMock()
        reset_resp = self.client.post("/session/reset", params={"user_id": "u2"})
        assert reset_resp.status_code == 200
        assert reset_resp.json()["status"] == "reset"

    def test_task_retry_workflow(self):
        """工单重试流程：失败→重试。"""
        # 1. 查询失败的工单
        self.mocks["db"].get_ticket = AsyncMock(return_value={
            "ticket_id": "TKT-RETRY",
            "user_id": "u1",
            "status": "failed",
            "project_name": "重试测试",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
        })

        # 2. 重试
        self.mocks["db"].update_ticket_status = AsyncMock()
        self.mocks["task_queue"].submit = AsyncMock()

        retry_resp = self.client.post("/task/TKT-RETRY/retry")
        assert retry_resp.status_code == 200
        data = retry_resp.json()
        assert data["status"] == "queued"

    def test_task_list_endpoint(self):
        """工单列表查询。"""
        self.mocks["db"].get_user_tickets = AsyncMock(return_value=[
            {
                "ticket_id": "TKT-01",
                "user_id": "u1",
                "project_name": "项目1",
                "status": "completed",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
            },
            {
                "ticket_id": "TKT-02",
                "user_id": "u1",
                "project_name": "项目2",
                "status": "queued",
                "created_at": "2025-01-02T00:00:00",
                "updated_at": "2025-01-02T00:00:00",
            },
        ])

        resp = self.client.get("/task/list", params={"user_id": "u1", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tickets"]) == 2


# ── CORS 和中间件测试 ──────────────────────────────────────────


@pytest.mark.e2e
class TestMiddleware:
    """中间件行为测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_full_app()
        self.client = TestClient(self.app)

    def test_cors_headers(self):
        """CORS 头正确返回。"""
        self.mocks["session_manager"].active_count.return_value = 0
        resp = self.client.options("/status", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        # FastAPI CORSMiddleware 应返回 CORS 头
        assert resp.status_code == 200

    def test_404_for_unknown_route(self):
        """未知路由返回 404。"""
        resp = self.client.get("/nonexistent/route")
        assert resp.status_code == 404


# ── 错误处理测试 ──────────────────────────────────────────────


@pytest.mark.e2e
class TestErrorHandling:
    """错误处理测试。"""

    def setup_method(self):
        self.app, self.mocks = _create_full_app()
        self.client = TestClient(self.app)

    def test_chat_internal_error(self):
        """chat 内部错误时返回 500。"""
        self.mocks["agent_service"].chat.side_effect = Exception("Internal error")
        # raise_server_exceptions=False 让 TestClient 返回 500 响应而非抛出异常
        client = TestClient(self.app, raise_server_exceptions=False)
        resp = client.post("/chat", json={"user_id": "u1", "message": "hi"})
        assert resp.status_code == 500

    def test_submit_task_internal_error(self):
        """工单提交内部错误时返回错误信息。"""
        self.mocks["db"].create_ticket.side_effect = Exception("DB error")
        resp = self.client.post("/task/submit", json={
            "user_id": "u1",
            "project_name": "测试",
            "description": "测试",
        })
        # 路由中有 try/except，返回 error
        assert resp.status_code == 200
        assert "error" in resp.json()
