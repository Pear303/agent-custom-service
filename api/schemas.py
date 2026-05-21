"""客服 API 请求/响应模型"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    message: str = Field(..., description="用户消息内容")
    stream: bool = Field(False, description="是否使用流式响应")


class ChatResponse(BaseModel):
    user_id: str
    answer: str
    conversation_id: str | None = None
    source: str = Field("dify", description="回答来源: dify 或 agent")


class SessionResetResponse(BaseModel):
    user_id: str
    status: str = "reset"


class ServiceStatus(BaseModel):
    active_sessions: int
    dify_status: str = "unknown"
