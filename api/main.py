"""FastAPI 客服服务入口 —— 应用创建 + 中间件 + 路由挂载"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.database import Database
from .core.lifespan import create_lifespan
from .services.session_manager import SessionManager
from .services.agent_service import AgentService
from .task_queue import TaskQueue
from .clients.dify import DifyChatflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── 服务实例化 ──────────────────────────────────────────────
db = Database()
session_manager = SessionManager(
    timeout_minutes=settings.session_timeout_minutes,
    max_sessions=settings.max_sessions,
    db=db,
)
agent_service = AgentService(session_manager)
task_queue = TaskQueue(db, agent_service)

# ── FastAPI 应用创建 ────────────────────────────────────────
app = FastAPI(
    title="Agent Customer Service API",
    version="0.5.0",
    lifespan=create_lifespan(db, session_manager, task_queue),
)

# 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由挂载 ────────────────────────────────────────────────
from .routers.health import create_router as create_health_router
from .routers.chat import create_router as create_chat_router
from .routers.session import create_router as create_session_router
from .routers.task import create_router as create_task_router
from .routers.dify_tools import create_router as create_dify_tools_router

app.include_router(create_health_router(session_manager, lambda: DifyChatflowClient()))
app.include_router(create_chat_router(agent_service))
app.include_router(create_session_router(session_manager))
app.include_router(create_task_router(db, agent_service, task_queue))
app.include_router(create_dify_tools_router(db, agent_service, task_queue))

# 生产模式：挂载前端构建产物 + SPA 回退
# 注意：必须放在所有 API 路由之后，否则会拦截 API 请求
if not os.getenv("DEV_MODE", "").lower() in ("1", "true", "yes"):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            file_path = frontend_dist / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.service_port, reload=False)
