"""健康检查路由"""
from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter

from ..core.config import settings
from ..core.database import DB_PATH
from ..schemas.common import ServiceStatus

logger = logging.getLogger(__name__)


def create_router(session_manager, dify_client_factory) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/status", response_model=ServiceStatus)
    async def status():
        return ServiceStatus(active_sessions=session_manager.active_count())

    @router.get("/health")
    async def health():
        health_status = {
            "status": "ok",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {},
        }

        try:
            dify = dify_client_factory()
            await dify.chat(query="health_check", user_id="health_check", conversation_id="")
            health_status["checks"]["dify"] = "connected"
        except Exception as e:
            health_status["checks"]["dify"] = f"error: {str(e)[:100]}"
            health_status["status"] = "degraded"

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
            await client.chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            health_status["checks"]["deepseek"] = "connected"
        except Exception as e:
            health_status["checks"]["deepseek"] = f"error: {str(e)[:100]}"
            health_status["status"] = "degraded"

        try:
            import shutil
            total, used, free = shutil.disk_usage(str(DB_PATH.parent))
            health_status["checks"]["disk"] = {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round(used / total * 100, 1),
            }
            if used / total > 0.9:
                health_status["status"] = "warning"
        except Exception as e:
            health_status["checks"]["disk"] = f"error: {str(e)[:100]}"

        try:
            ticket_count = await session_manager.db._pool.execute("SELECT COUNT(*) FROM tickets")
            session_count = await session_manager.db._pool.execute("SELECT COUNT(*) FROM sessions")
            health_status["checks"]["database"] = {
                "tickets": (await ticket_count.fetchone())[0],
                "sessions": (await session_count.fetchone())[0],
                "status": "healthy",
            }
        except Exception as e:
            health_status["checks"]["database"] = f"error: {str(e)[:100]}"
            health_status["status"] = "error"

        return health_status

    return router
