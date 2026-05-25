"""聊天路由"""
from __future__ import annotations

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from ..schemas.chat import ChatRequest, ChatResponse


def create_router(agent_service) -> APIRouter:
    router = APIRouter(tags=["chat"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        result = await agent_service.chat(req.user_id, req.message)
        return ChatResponse(**result)

    @router.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        return EventSourceResponse(
            agent_service.chat_stream(req.user_id, req.message),
            sep="\n",
        )

    return router
