"""通用响应模型"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SessionResetResponse(BaseModel):
    user_id: str
    status: str = "reset"

class SessionHistoryResponse(BaseModel):
    user_id: str
    history: list[dict] = Field(default_factory=list)
    message_count: int = 0


class ServiceStatus(BaseModel):
    active_sessions: int
    dify_status: str = "unknown"
