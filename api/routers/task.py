"""工单相关路由"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter

from ..schemas.task import RequirementRequest
from ..utils.file_manager import _clean_output_path
from ..utils.progress import _calculate_progress, _check_local_status, _enrich_local_status

logger = logging.getLogger(__name__)


def create_router(db, agent_service, task_queue) -> APIRouter:
    router = APIRouter(tags=["task"])

    _PROJECT_ROOT = Path(__file__).parent.parent.parent

    async def _run_development(ticket_id: str, ticket: dict):
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
                logger.info("工单 %s 开发完成", ticket_id)
            else:
                await db.update_ticket_status(
                    ticket_id,
                    "development_failed",
                    development_error=result.get("error", "开发失败")
                )
                logger.error("工单 %s 开发失败: %s", ticket_id, result.get("error"))
        except Exception as exc:
            logger.error("工单 %s 开发异常: %s", ticket_id, exc, exc_info=True)
            await db.update_ticket_status(ticket_id, "development_failed", development_error=str(exc))

    @router.post("/task/submit")
    async def submit_requirement(req: RequirementRequest):
        try:
            logger.info("收到需求提交请求: user_id=%s, project=%s", req.user_id, req.project_name)
            ticket_id = await db.create_ticket(req.model_dump())
            logger.info("工单创建成功: %s", ticket_id)
            await task_queue.submit(ticket_id)
            logger.info("工单已入队: %s", ticket_id)
            return {
                "ticket_id": ticket_id,
                "status": "queued",
                "message": "工单已提交，正在排队处理"
            }
        except Exception as exc:
            logger.error("工单提交失败: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    @router.get("/task/{ticket_id}/status")
    async def get_ticket_status(ticket_id: str):
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
            "local_status": _enrich_local_status(
                ticket,
                _check_local_status(ticket["user_id"], ticket_id, _PROJECT_ROOT),
            ),
        }

    @router.get("/task/list")
    async def list_tickets(user_id: str, limit: int = 5):
        tickets = await db.get_user_tickets(user_id, limit)
        for ticket in tickets:
            ticket["local_status"] = _enrich_local_status(
                ticket,
                _check_local_status(user_id, ticket["ticket_id"], _PROJECT_ROOT),
            )
            ticket["progress"] = _calculate_progress(ticket["status"])
        return {"tickets": tickets}
    
    @router.get("/task/list/description")
    async def get_tickets_description(user_id: str, limit: int = 5):
        limit = min(limit, 10)
        tickets = await db.get_user_tickets(user_id, limit)
        tickets_description = []
        for ticket in tickets:
            ticket["local_status"] = _enrich_local_status(
                ticket,
                _check_local_status(user_id, ticket["ticket_id"], _PROJECT_ROOT),
            )
            ticket["progress"] = _calculate_progress(ticket["status"])
            tickets_description.append({
                "ticket_id": ticket["ticket_id"],
                "user_id": ticket["user_id"],
                "project_name": ticket.get("project_name", "/"),
                "project_type": ticket.get("project_type", "/"),
                 "status": ticket["status"],
                 "development_status": ticket.get("development_status", "/"),
            })
        return {"tickets_description": tickets_description}

    @router.post("/task/{ticket_id}/start-development")
    async def start_development(ticket_id: str):
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"error": "工单不存在"}
        if ticket["status"] not in ("pending_development", "development_failed"):
            return {"error": f"当前状态 ({ticket['status']}) 不允许开始开发"}

        try:
            logger.info("开始开发工单: %s", ticket_id)
            await db.update_ticket_status(ticket_id, "developing", development_error=None)
            asyncio.create_task(_run_development(ticket_id, ticket))
            return {"status": "developing", "message": "开发已启动"}
        except Exception as exc:
            logger.error("启动开发失败: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    @router.post("/task/{ticket_id}/retry")
    async def retry_ticket(ticket_id: str):
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"status": "error", "error": "工单不存在"}
        if ticket["status"] not in ("failed", "completed"):
            return {"status": "error", "error": f"当前状态 ({ticket['status']}) 不允许重试"}

        try:
            logger.info("重试工单: %s (原状态: %s)", ticket_id, ticket["status"])
            await db.update_ticket_status(
                ticket_id,
                "queued",
                error=None,
                analysis=None,
                prd=None,
                quote=None,
                development_error=None,
                development_output=None,
            )
            await task_queue.submit(ticket_id)
            return {"status": "queued", "message": "工单已重新提交，正在排队处理"}
        except Exception as exc:
            logger.error("重试工单失败: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    @router.post("/task/{ticket_id}/restore-local")
    async def restore_local_files(ticket_id: str):
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"status": "error", "error": "工单不存在"}

        ticket_dir = _PROJECT_ROOT / "data" / "users" / ticket["user_id"] / ticket_id
        restored = {"ticket_json": False, "reports": [], "products": 0}

        try:
            ticket_dir.mkdir(parents=True, exist_ok=True)

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
                    file_content = entry.get("content") or entry.get("code") or ""
                    if not file_path or not file_content:
                        continue
                    safe_path = _clean_output_path(file_path, ticket["user_id"], ticket_id)
                    target = product_dir / safe_path
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(file_content, encoding="utf-8")
                        restored["products"] += 1

            logger.info("工单 %s 本地文件恢复完成: %s", ticket_id, restored)
            return {
                "status": "ok",
                "restored": restored,
                "local_status": _check_local_status(ticket["user_id"], ticket_id, _PROJECT_ROOT),
            }

        except Exception as exc:
            logger.error("恢复工单 %s 本地文件失败: %s", ticket_id, exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    return router
