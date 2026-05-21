"""FastAPI 客服服务入口"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .agent_service import AgentService
from .config import settings
from .database import Database, DB_PATH
from .schemas import ChatRequest, ChatResponse, SessionResetResponse, ServiceStatus
from .session_manager import SessionManager
from .task_queue import TaskQueue
from services.dify import DifyChatflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

db = Database()
session_manager = SessionManager(
    timeout_minutes=settings.session_timeout_minutes,
    max_sessions=settings.max_sessions,
    db=db,
)
agent_service = AgentService(session_manager)
task_queue = TaskQueue(db, agent_service)


async def ensure_db_initialized():
    """确保数据库已初始化"""
    if db._pool is None:
        await db.init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()                         # 启动：初始化数据库
    await session_manager.load_from_db()    # 加载持久化会话
    
    # 恢复服务重启时中断的工单
    recovered = await db.recover_interrupted_tickets()
    if recovered:
        for r in recovered:
            logging.info("恢复中断工单 %s: %s -> %s", r["ticket_id"], r["old_status"], r["new_status"])
            if r["new_status"] == "queued":
                await task_queue.submit(r["ticket_id"])
    
    await task_queue.start()                # 启动后台 workers
    yield
    await task_queue.stop()
    expired = await session_manager.cleanup_expired()
    if expired:
        logging.info("清理了 %d 个过期会话", expired)
    await db.close()


app = FastAPI(title="Agent Customer Service API", version="0.4.0", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent.parent / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent.parent / "static" / "index.html"))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await agent_service.chat(req.user_id, req.message)
    return ChatResponse(**result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return EventSourceResponse(
        agent_service.chat_stream(req.user_id, req.message),
        sep="\n",
    )


@app.post("/session/reset", response_model=SessionResetResponse)
async def reset_session(user_id: str):
    await session_manager.reset_async(user_id)
    return SessionResetResponse(user_id=user_id)


@app.get("/status", response_model=ServiceStatus)
async def status():
    return ServiceStatus(active_sessions=session_manager.active_count())


class RequirementRequest(BaseModel):
    user_id: str
    project_name: str
    project_type: str = ""
    description: str
    deadline: str = ""
    budget: str = ""


@app.post("/task/submit")
async def submit_requirement(req: RequirementRequest):
    await ensure_db_initialized()
    try:
        logging.info("收到需求提交请求: user_id=%s, project=%s", req.user_id, req.project_name)
        ticket_id = await db.create_ticket(req.model_dump())
        logging.info("工单创建成功: %s", ticket_id)
        await task_queue.submit(ticket_id)
        logging.info("工单已入队: %s", ticket_id)
        return {
            "ticket_id": ticket_id,
            "status": "queued",
            "message": "工单已提交，正在排队处理"
        }
    except Exception as exc:
        logging.error("工单提交失败: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


@app.get("/task/{ticket_id}/status")
async def get_ticket_status(ticket_id: str):
    await ensure_db_initialized()
    ticket = await db.get_ticket(ticket_id)
    if not ticket:
        return {"error": "工单不存在"}
    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "progress": _calculate_progress(ticket["status"]),
        "analysis": ticket.get("analysis"),
        "prd": ticket.get("prd"),
        "quote": ticket.get("quote"),
        "development_output": ticket.get("development_output"),
        "development_error": ticket.get("development_error"),
        "error": ticket.get("error"),
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
    }


@app.get("/task/list")
async def list_tickets(user_id: str, limit: int = 50):
    await ensure_db_initialized()
    tickets = await db.get_user_tickets(user_id, limit)
    return {"tickets": tickets}


@app.post("/task/{ticket_id}/start-development")
async def start_development(ticket_id: str):
    await ensure_db_initialized()
    ticket = await db.get_ticket(ticket_id)
    if not ticket:
        return {"error": "工单不存在"}
    if ticket["status"] not in ("pending_development", "development_failed"):
        return {"error": f"当前状态 ({ticket['status']}) 不允许开始开发"}
    
    try:
        logging.info("开始开发工单: %s", ticket_id)
        await db.update_ticket_status(ticket_id, "developing", development_error=None)
        
        # 异步启动开发流程
        asyncio.create_task(_run_development(ticket_id, ticket))
        
        return {"status": "developing", "message": "开发已启动"}
    except Exception as exc:
        logging.error("启动开发失败: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


async def _run_development(ticket_id: str, ticket: dict):
    """后台执行开发流程"""
    try:
        result = await agent_service.develop_project(
            ticket["user_id"],
            {
                "project_name": ticket["project_name"],
                "analysis": ticket.get("analysis"),
                "prd": ticket.get("prd"),
                "quote": ticket.get("quote"),
            },
            ticket_id=ticket_id,
        )
        if result["status"] == "completed":
            await db.update_ticket_status(
                ticket_id, 
                "development_completed",
                development_output=result["data"]
            )
            logging.info("工单 %s 开发完成", ticket_id)
        else:
            await db.update_ticket_status(
                ticket_id,
                "development_failed",
                development_error=result.get("error", "开发失败")
            )
            logging.error("工单 %s 开发失败: %s", ticket_id, result.get("error"))
    except Exception as exc:
        logging.error("工单 %s 开发异常: %s", ticket_id, exc, exc_info=True)
        await db.update_ticket_status(ticket_id, "development_failed", development_error=str(exc))


@app.get("/health")
async def health():
    health_status = {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {},
    }
    
    # Dify 连接测试
    try:
        dify = DifyChatflowClient()
        await dify.chat(query="health_check", user_id="health_check", conversation_id="")
        health_status["checks"]["dify"] = "connected"
    except Exception as e:
        health_status["checks"]["dify"] = f"error: {str(e)[:100]}"
        health_status["status"] = "degraded"
    
    # DeepSeek API 测试
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
    
    # 磁盘空间检查
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
    
    # 数据库状态
    try:
        ticket_count = await db._pool.execute("SELECT COUNT(*) FROM tickets")
        session_count = await db._pool.execute("SELECT COUNT(*) FROM sessions")
        health_status["checks"]["database"] = {
            "tickets": (await ticket_count.fetchone())[0],
            "sessions": (await session_count.fetchone())[0],
            "status": "healthy",
        }
    except Exception as e:
        health_status["checks"]["database"] = f"error: {str(e)[:100]}"
        health_status["status"] = "error"
    
    return health_status


def _calculate_progress(status: str) -> int:
    progress_map = {
        "queued": 0,
        "analyzing": 15,
        "designing": 30,
        "estimating": 45,
        "completed": 50,
        "pending_development": 50,
        "developing": 75,
        "development_completed": 100,
        "development_failed": 50,
        "failed": -1,
    }
    return progress_map.get(status, 0)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.service_port, reload=False)
