"""会话管理路由"""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas.common import SessionResetResponse, SessionHistoryResponse


def create_router(session_manager) -> APIRouter:
    router = APIRouter(tags=["session"])

    @router.post("/session/reset", response_model=SessionResetResponse)
    async def reset_session(user_id: str):
        await session_manager.reset_async(user_id)
        return SessionResetResponse(user_id=user_id)
    
    @router.get("/session/history", response_model=SessionHistoryResponse)
    async def get_session_history(user_id: str):
        session = session_manager.get_or_create(user_id)  # 用同步版，只是读取
        return SessionHistoryResponse(
            user_id=user_id,
            history=session.history,
            message_count=session.message_count,
        )

    return router