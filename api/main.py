"""FastAPI 客服服务入口"""
from __future__ import annotations

import asyncio
import json
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
        "user_id": ticket["user_id"],
        "project_name": ticket.get("project_name", ""),
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
        "local_status": _enrich_local_status(ticket, _check_local_status(ticket["user_id"], ticket_id)),
    }


@app.get("/task/list")
async def list_tickets(user_id: str, limit: int = 50):
    await ensure_db_initialized()
    tickets = await db.get_user_tickets(user_id, limit)
    for ticket in tickets:
        ticket["local_status"] = _enrich_local_status(ticket, _check_local_status(user_id, ticket["ticket_id"]))
        ticket["progress"] = _calculate_progress(ticket["status"])
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


@app.post("/task/{ticket_id}/restore-local")
async def restore_local_files(ticket_id: str):
    await ensure_db_initialized()
    ticket = await db.get_ticket(ticket_id)
    if not ticket:
        return {"status": "error", "error": "工单不存在"}

    root = Path(__file__).parent.parent
    ticket_dir = root / "data" / "users" / ticket["user_id"] / ticket_id
    restored = {"ticket_json": False, "reports": [], "products": 0}

    try:
        ticket_dir.mkdir(parents=True, exist_ok=True)

        # 1. 恢复工单 JSON
        ticket_json_dir = ticket_dir / "工单"
        ticket_json_dir.mkdir(exist_ok=True)
        ticket_json_path = ticket_json_dir / "工单.json"
        if not ticket_json_path.exists():
            record = {
                "ticket_id": ticket_id,
                "user_id": ticket["user_id"],
                "project_name": ticket.get("project_name", ""),
                "project_type": ticket.get("project_type", ""),
                "description": ticket.get("description", ""),
                "deadline": ticket.get("deadline", ""),
                "budget": ticket.get("budget", ""),
                "status": ticket["status"],
            }
            ticket_json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            restored["ticket_json"] = True

        # 2. 恢复报告文件
        report_dir = ticket_dir / "报告"
        report_dir.mkdir(exist_ok=True)
        report_map = {
            "需求分析.json": ticket.get("analysis"),
            "PRD.json": ticket.get("prd"),
            "报价单.json": ticket.get("quote"),
        }
        for filename, data in report_map.items():
            if data:
                filepath = report_dir / filename
                if not filepath.exists():
                    content = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)
                    filepath.write_text(content, encoding="utf-8")
                    restored["reports"].append(filename)

        # 3. 恢复成品文件
        dev_output = ticket.get("development_output")
        files = None
        if isinstance(dev_output, dict):
            files = dev_output.get("files")
        if files and isinstance(files, list):
            product_dir = ticket_dir / "成品"
            product_dir.mkdir(exist_ok=True)
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                file_path = entry.get("path") or entry.get("file") or entry.get("name")
                file_path = file_path.lstrip("/\\")
                file_content = entry.get("content") or entry.get("code") or ""
                if not file_path or not file_content:
                    continue
                target = product_dir / file_path
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(file_content, encoding="utf-8")
                    restored["products"] += 1

        logging.info("工单 %s 本地文件恢复完成: %s", ticket_id, restored)
        return {"status": "ok", "restored": restored, "local_status": _check_local_status(ticket["user_id"], ticket_id)}

    except Exception as exc:
        logging.error("恢复工单 %s 本地文件失败: %s", ticket_id, exc, exc_info=True)
        return {"status": "error", "error": str(exc)}


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


def _check_local_status(user_id: str, ticket_id: str) -> dict:
    """检查工单在本地文件系统中的实际状态。"""
    _EXPECTED_REPORTS = ["需求分析.json", "PRD.json", "报价单.json"]
    root = Path(__file__).parent.parent
    ticket_dir = root / "data" / "users" / user_id / ticket_id
    result = {
        "directory_exists": False,
        "ticket_json_exists": False,
        "has_report": False,
        "report_files": [],
        "expected_reports": [],
        "missing_reports": [],
        "report_status": "not_expected",
        "has_product": False,
        "product_file_count": 0,
        "product_sample": [],
        "local_deleted": False,
        "is_empty_workspace": True,
    }
    
    if not ticket_dir.exists():
        result["local_deleted"] = True
        return result
    
    result["directory_exists"] = True
    
    # 检查工单 JSON
    ticket_json = ticket_dir / "工单" / "工单.json"
    if ticket_json.exists():
        result["ticket_json_exists"] = True
        result["is_empty_workspace"] = False
    
    # 检查报告目录 — 逐份检查三份标准报告
    report_dir = ticket_dir / "报告"
    if report_dir.exists():
        try:
            report_files = []
            for rpt in _EXPECTED_REPORTS:
                if (report_dir / rpt).is_file():
                    report_files.append(rpt)
            if report_files:
                result["has_report"] = True
                result["report_files"] = sorted(report_files)
                result["is_empty_workspace"] = False
        except OSError:
            pass
    
    # 检查成品目录
    product_dir = ticket_dir / "成品"
    if product_dir.exists():
        try:
            all_files = []
            for f in product_dir.rglob("*"):
                if f.is_file() and not any(p.startswith(".") for p in f.parts):
                    all_files.append(str(f.relative_to(product_dir)))
            if all_files:
                result["has_product"] = True
                result["product_file_count"] = len(all_files)
                result["product_sample"] = sorted(all_files)[:5]
                result["is_empty_workspace"] = False
        except OSError:
            pass
    
    return result


def _enrich_local_status(ticket: dict, local_status: dict) -> dict:
    """根据数据库中的数据，与本地文件系统状态做对比，填充 report_status / missing_reports 等字段。"""
    expected = []
    if ticket.get("analysis"):
        expected.append("需求分析.json")
    if ticket.get("prd"):
        expected.append("PRD.json")
    if ticket.get("quote"):
        expected.append("报价单.json")

    local_status["expected_reports"] = expected
    actual = set(local_status["report_files"])
    missing = [r for r in expected if r not in actual]

    if not expected:
        local_status["report_status"] = "not_expected"
    elif not missing:
        local_status["report_status"] = "complete"
    elif actual:
        local_status["report_status"] = "partial"
    else:
        local_status["report_status"] = "missing"

    local_status["missing_reports"] = missing
    return local_status


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
