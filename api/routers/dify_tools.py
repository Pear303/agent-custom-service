"""Dify 智能客服工具函数路由

所有端点统一使用 /dify 前缀，返回值面向 AI 可读，
Dify Agent 可直接将结果转述给用户。
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query

from ..schemas.task import RequirementRequest
from ..utils.progress import _calculate_progress, _check_local_status, _enrich_local_status

logger = logging.getLogger(__name__)

_STATUS_LABELS = {
    "queued": "排队中",
    "analyzing": "需求分析中",
    "designing": "PRD 设计中",
    "estimating": "成本估算中",
    "completed": "分析完成（待确认开发）",
    "pending_development": "待开发（等待用户确认）",
    "developing": "开发中",
    "development_completed": "开发完成",
    "development_failed": "开发失败",
    "failed": "处理失败",
}

_DEV_STATUS_LABELS = {
    "not_started": "未开始",
    "pending": "等待开始",
    "in_progress": "进行中",
    "completed": "已完成",
    "failed": "失败",
}


def create_router(db, agent_service, task_queue) -> APIRouter:
    router = APIRouter(prefix="/dify", tags=["dify-tools"])

    _PROJECT_ROOT = Path(__file__).parent.parent.parent

    @router.get("/ticket/detail")
    async def get_ticket_detail(ticket_id: str):
        """查询工单详情，返回用户可读的摘要信息。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        progress = _calculate_progress(ticket["status"])
        status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
        dev_status_label = _DEV_STATUS_LABELS.get(
            ticket.get("development_status", "not_started"), ticket.get("development_status", "/")
        )

        detail = {
            "ticket_id": ticket["ticket_id"],
            "user_id": ticket["user_id"],
            "project_name": ticket.get("project_name", "/"),
            "project_type": ticket.get("project_type", "/"),
            "description": ticket.get("description", "/"),
            "status": ticket["status"],
            "status_label": status_label,
            "progress": progress,
            "development_status": ticket.get("development_status", "not_started"),
            "development_status_label": dev_status_label,
            "deadline": ticket.get("deadline", "/"),
            "budget": ticket.get("budget", "/"),
            "created_at": ticket.get("created_at", "/"),
            "updated_at": ticket.get("updated_at", "/"),
            "has_analysis": bool(ticket.get("analysis")),
            "has_prd": bool(ticket.get("prd")),
            "has_quote": bool(ticket.get("quote")),
            "has_development_output": bool(ticket.get("development_output")),
            "error": ticket.get("error"),
            "development_error": ticket.get("development_error"),
        }

        summary = (
            f"工单 {ticket_id}（项目：{detail['project_name']}）"
            f"当前状态：{status_label}，进度：{progress}%。"
        )
        if detail["has_analysis"]:
            summary += "已完成需求分析。"
        if detail["has_prd"]:
            summary += "已完成PRD设计。"
        if detail["has_quote"]:
            summary += "已生成报价单。"
        if detail["has_development_output"]:
            summary += "已产出开发成果。"
        if detail["error"]:
            summary += f" 错误信息：{detail['error']}"
        if detail["development_error"]:
            summary += f" 开发错误：{detail['development_error']}"

        return {"success": True, "data": detail, "summary": summary}

    @router.get("/ticket/progress")
    async def get_ticket_progress(ticket_id: str):
        """查询工单进度百分比和当前阶段描述。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        progress = _calculate_progress(ticket["status"])
        status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])

        stage_descriptions = {
            "queued": "您的需求已提交，正在排队等待分析师处理。",
            "analyzing": "需求分析师正在对您的需求进行结构化分析。",
            "designing": "产品经理正在根据需求分析设计产品需求文档（PRD）。",
            "estimating": "成本估算师正在根据PRD计算开发成本和报价。",
            "completed": "需求分析、PRD设计和成本估算均已完成，等待您确认是否开始开发。",
            "pending_development": "所有前期工作已完成，等待您确认开始开发。",
            "developing": "开发工程师正在根据PRD进行项目开发。",
            "development_completed": "项目开发已完成，您可以查看开发产出。",
            "development_failed": "项目开发过程中遇到问题，您可以重试或联系人工客服。",
            "failed": "工单处理过程中遇到问题，您可以重试或联系人工客服。",
        }
        stage_desc = stage_descriptions.get(ticket["status"], "未知状态")

        return {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "project_name": ticket.get("project_name", "/"),
                "status": ticket["status"],
                "status_label": status_label,
                "progress": progress,
                "stage_description": stage_desc,
            },
            "summary": f"工单 {ticket_id}（{ticket.get('project_name', '/')}）：{stage_desc} 进度 {progress}%。",
        }

    @router.get("/ticket/analysis")
    async def get_ticket_analysis(ticket_id: str):
        """查询工单的需求分析报告摘要。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        analysis = ticket.get("analysis")
        if not analysis:
            status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
            return {
                "success": False,
                "message": f"工单 {ticket_id} 尚未完成需求分析，当前状态：{status_label}",
            }

        if isinstance(analysis, str):
            import json
            try:
                analysis = json.loads(analysis)
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw": analysis}, "summary": analysis[:500]}

        summary_parts = []
        if isinstance(analysis, dict):
            if "项目概述" in analysis or "project_overview" in analysis:
                summary_parts.append(f"项目概述：{analysis.get('项目概述') or analysis.get('project_overview', '/')}")
            if "核心功能" in analysis or "core_features" in analysis:
                summary_parts.append(f"核心功能：{analysis.get('核心功能') or analysis.get('core_features', '/')}")
            if "复杂度评估" in analysis or "complexity" in analysis:
                summary_parts.append(f"复杂度评估：{analysis.get('复杂度评估') or analysis.get('complexity', '/')}")
            if "风险点" in analysis or "risks" in analysis:
                summary_parts.append(f"风险点：{analysis.get('风险点') or analysis.get('risks', '/')}")
            if "待澄清问题" in analysis or "clarifications" in analysis:
                summary_parts.append(f"待澄清问题：{analysis.get('待澄清问题') or analysis.get('clarifications', '/')}")

        summary = "；".join(summary_parts) if summary_parts else "需求分析已完成，请联系客服查看详细内容。"

        return {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "project_name": ticket.get("project_name", "/"),
                "analysis": analysis,
            },
            "summary": summary,
        }

    @router.get("/ticket/prd")
    async def get_ticket_prd(ticket_id: str):
        """查询工单的PRD文档摘要。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        prd = ticket.get("prd")
        if not prd:
            status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
            return {
                "success": False,
                "message": f"工单 {ticket_id} 尚未完成PRD设计，当前状态：{status_label}",
            }

        if isinstance(prd, str):
            import json
            try:
                prd = json.loads(prd)
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw": prd}, "summary": prd[:500]}

        summary_parts = []
        if isinstance(prd, dict):
            if "产品定位" in prd or "product_positioning" in prd:
                summary_parts.append(f"产品定位：{prd.get('产品定位') or prd.get('product_positioning', '/')}")
            if "功能清单" in prd or "feature_list" in prd:
                summary_parts.append(f"功能清单：{prd.get('功能清单') or prd.get('feature_list', '/')}")
            if "用户故事" in prd or "user_stories" in prd:
                summary_parts.append(f"用户故事：{prd.get('用户故事') or prd.get('user_stories', '/')}")
            if "技术复杂度" in prd or "tech_complexity" in prd:
                summary_parts.append(f"技术复杂度：{prd.get('技术复杂度') or prd.get('tech_complexity', '/')}")
            if "功能总数" in prd or "total_features" in prd:
                summary_parts.append(f"功能总数：{prd.get('功能总数') or prd.get('total_features', '/')}")

        summary = "；".join(summary_parts) if summary_parts else "PRD文档已生成，请联系客服查看详细内容。"

        return {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "project_name": ticket.get("project_name", "/"),
                "prd": prd,
            },
            "summary": summary,
        }

    @router.get("/ticket/quote")
    async def get_ticket_quote(ticket_id: str):
        """查询工单的报价单摘要。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        quote = ticket.get("quote")
        if not quote:
            status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
            return {
                "success": False,
                "message": f"工单 {ticket_id} 尚未生成报价单，当前状态：{status_label}",
            }

        if isinstance(quote, str):
            import json
            try:
                quote = json.loads(quote)
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw": quote}, "summary": quote[:500]}

        summary_parts = []
        if isinstance(quote, dict):
            if "总报价" in quote or "total_cost" in quote:
                summary_parts.append(f"总报价：{quote.get('总报价') or quote.get('total_cost', '/')}")
            if "交付周期" in quote or "delivery_period" in quote:
                summary_parts.append(f"交付周期：{quote.get('交付周期') or quote.get('delivery_period', '/')}")
            if "付款节点" in quote or "payment_milestones" in quote:
                summary_parts.append(f"付款节点：{quote.get('付款节点') or quote.get('payment_milestones', '/')}")
            if "售后支持" in quote or "support_period" in quote:
                summary_parts.append(f"售后支持：{quote.get('售后支持') or quote.get('support_period', '/')}")
            if "分项明细" in quote or "breakdown" in quote:
                summary_parts.append("已包含分项明细")

        summary = "；".join(summary_parts) if summary_parts else "报价单已生成，请联系客服查看详细内容。"

        return {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "project_name": ticket.get("project_name", "/"),
                "quote": quote,
            },
            "summary": summary,
        }

    @router.get("/ticket/development")
    async def get_ticket_development(ticket_id: str):
        """查询工单的开发产出摘要（文件列表、技术栈等）。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        dev_output = ticket.get("development_output")
        if not dev_output:
            dev_status = ticket.get("development_status", "not_started")
            dev_label = _DEV_STATUS_LABELS.get(dev_status, dev_status)
            return {
                "success": False,
                "message": f"工单 {ticket_id} 尚无开发产出，开发状态：{dev_label}",
            }

        if isinstance(dev_output, str):
            import json
            try:
                dev_output = json.loads(dev_output)
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw": dev_output}, "summary": dev_output[:500]}

        file_list = []
        tech_stack = None
        setup_instructions = None
        project_structure = None

        if isinstance(dev_output, dict):
            files = dev_output.get("files", [])
            for f in files:
                if isinstance(f, dict):
                    path = f.get("path") or f.get("file") or f.get("name", "")
                    if path:
                        file_list.append(path)
                elif isinstance(f, str):
                    file_list.append(f)
            tech_stack = dev_output.get("tech_stack")
            setup_instructions = dev_output.get("setup_instructions")
            project_structure = dev_output.get("project_structure")

        summary_parts = [f"共生成 {len(file_list)} 个文件"]
        if tech_stack:
            summary_parts.append(f"技术栈：{tech_stack}")
        if file_list:
            display_files = file_list[:10]
            summary_parts.append(f"主要文件：{', '.join(display_files)}")
            if len(file_list) > 10:
                summary_parts.append(f"等共 {len(file_list)} 个文件")

        summary = "；".join(summary_parts)

        return {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "project_name": ticket.get("project_name", "/"),
                "development_status": ticket.get("development_status", "not_started"),
                "file_count": len(file_list),
                "file_list": file_list,
                "tech_stack": tech_stack,
                "setup_instructions": setup_instructions,
                "project_structure": project_structure,
            },
            "summary": summary,
        }

    @router.post("/ticket/create")
    async def create_ticket(req: RequirementRequest):
        """为用户创建新工单/提交需求。"""
        try:
            ticket_id = await db.create_ticket(req.model_dump())
            await task_queue.submit(ticket_id)
            summary = (
                f"已为用户 {req.user_id} 创建工单 {ticket_id}，"
                f"项目名称：{req.project_name}，"
                f"已提交排队处理。"
            )
            return {
                "success": True,
                "data": {
                    "ticket_id": ticket_id,
                    "user_id": req.user_id,
                    "project_name": req.project_name,
                    "status": "queued",
                },
                "summary": summary,
            }
        except Exception as exc:
            logger.error("创建工单失败: %s", exc, exc_info=True)
            return {"success": False, "message": f"创建工单失败：{str(exc)}"}

    @router.post("/ticket/start-development")
    async def start_development(ticket_id: str):
        """启动工单的开发阶段。"""
        import asyncio

        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        if ticket["status"] not in ("pending_development", "development_failed"):
            status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
            return {
                "success": False,
                "message": f"工单 {ticket_id} 当前状态为「{status_label}」，不允许启动开发。需要先完成需求分析和报价。",
            }

        try:
            await db.update_ticket_status(ticket_id, "developing", development_error=None)

            async def _run_dev(tid: str, t: dict):
                try:
                    result = await agent_service.develop_project(
                        t["user_id"],
                        {
                            "project_name": t["project_name"],
                            "analysis": t.get("analysis"),
                            "prd": t.get("prd"),
                            "quote": t.get("quote"),
                        },
                        ticket_id=tid,
                    )
                    if result["status"] == "completed":
                        await db.update_ticket_status(tid, "development_completed", development_output=result["data"])
                    else:
                        await db.update_ticket_status(tid, "development_failed", development_error=result.get("error", "开发失败"))
                except Exception as exc:
                    logger.error("工单 %s 开发异常: %s", tid, exc, exc_info=True)
                    await db.update_ticket_status(tid, "development_failed", development_error=str(exc))

            asyncio.create_task(_run_dev(ticket_id, ticket))

            return {
                "success": True,
                "data": {"ticket_id": ticket_id, "status": "developing"},
                "summary": f"工单 {ticket_id}（{ticket.get('project_name', '/')}）已启动开发，请耐心等待。",
            }
        except Exception as exc:
            logger.error("启动开发失败: %s", exc, exc_info=True)
            return {"success": False, "message": f"启动开发失败：{str(exc)}"}

    @router.post("/ticket/retry")
    async def retry_ticket(ticket_id: str):
        """重试失败或已完成的工单。"""
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"success": False, "message": f"工单 {ticket_id} 不存在"}

        if ticket["status"] not in ("failed", "completed"):
            status_label = _STATUS_LABELS.get(ticket["status"], ticket["status"])
            return {
                "success": False,
                "message": f"工单 {ticket_id} 当前状态为「{status_label}」，不允许重试。仅失败或已完成的工单可重试。",
            }

        try:
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
            return {
                "success": True,
                "data": {"ticket_id": ticket_id, "status": "queued"},
                "summary": f"工单 {ticket_id}（{ticket.get('project_name', '/')}）已重新提交，正在排队处理。",
            }
        except Exception as exc:
            logger.error("重试工单失败: %s", exc, exc_info=True)
            return {"success": False, "message": f"重试工单失败：{str(exc)}"}

    @router.get("/user/summary")
    async def get_user_summary(
        user_id: str,
        ticket_limit: int = Query(default=5, ge=1, le=20),
    ):
        """获取用户全貌概览：工单数、最近工单、会话状态等。"""
        tickets = await db.get_user_tickets(user_id, ticket_limit)
        session = await db.load_session(user_id)

        total_tickets = len(tickets)
        status_counts: dict[str, int] = {}
        recent_tickets = []

        for ticket in tickets:
            s = ticket["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            status_label = _STATUS_LABELS.get(s, s)
            progress = _calculate_progress(s)
            recent_tickets.append({
                "ticket_id": ticket["ticket_id"],
                "project_name": ticket.get("project_name", "/"),
                "status": s,
                "status_label": status_label,
                "progress": progress,
                "created_at": ticket.get("created_at", "/"),
            })

        has_active_session = session is not None
        message_count = session.get("message_count", 0) if session else 0

        status_summary_parts = []
        for s, count in status_counts.items():
            label = _STATUS_LABELS.get(s, s)
            status_summary_parts.append(f"{label} {count}个")
        status_summary = "，".join(status_summary_parts) if status_summary_parts else "暂无工单"

        summary = f"用户 {user_id} 共有 {total_tickets} 个工单（{status_summary}）。"
        if has_active_session:
            summary += f" 当前有活跃会话，已发送 {message_count} 条消息。"
        else:
            summary += " 当前无活跃会话。"

        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "total_tickets": total_tickets,
                "status_counts": {_STATUS_LABELS.get(k, k): v for k, v in status_counts.items()},
                "recent_tickets": recent_tickets,
                "has_active_session": has_active_session,
                "message_count": message_count,
            },
            "summary": summary,
        }

    return router
