"""人工客服转接工具"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_HANDOFF_REQUESTS: dict[str, dict[str, Any]] = {}
_HANDOFF_LOCK = asyncio.Lock()

_HANDOFF_TIMEOUT_SECONDS = 300
_MAX_ACTIVE_REQUESTS = 50


class HandoffStatus:
    PENDING = "pending"
    ASSIGNED = "assigned"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


_STATUS_LABELS = {
    HandoffStatus.PENDING: "排队中",
    HandoffStatus.ASSIGNED: "已分配",
    HandoffStatus.RESOLVED: "已解决",
    HandoffStatus.CANCELLED: "已取消",
    HandoffStatus.TIMEOUT: "已超时",
}


class HumanHandoffTool:
    """人工转接。支持创建/查询/取消转接请求。"""

    @staticmethod
    async def _cleanup_expired():
        """清理超时的转接请求（由外部定时调用）。"""
        now = time.time()
        async with _HANDOFF_LOCK:
            expired = [
                rid for rid, req in _HANDOFF_REQUESTS.items()
                if req["status"] == HandoffStatus.PENDING and now - req["created_at"] > _HANDOFF_TIMEOUT_SECONDS
            ]
            for rid in expired:
                _HANDOFF_REQUESTS[rid]["status"] = HandoffStatus.TIMEOUT
                logger.info("转接请求 %s 超时自动取消", rid)

    async def create_handoff(self, user_id: str, reason: str, summary: str = "") -> str:
        async with _HANDOFF_LOCK:
            active = sum(
                1 for r in _HANDOFF_REQUESTS.values()
                if r.get("user_id") == user_id and r["status"] in (HandoffStatus.PENDING, HandoffStatus.ASSIGNED)
            )
            if active > 0:
                return f"您已有 {active} 个正在处理的转接请求，请勿重复提交。如需催办，可直接询问。"
            if len(_HANDOFF_REQUESTS) >= _MAX_ACTIVE_REQUESTS:
                return "当前转接请求较多，请稍后再试或留言描述问题，我们会在第一时间回复您。"

            request_id = str(uuid.uuid4())[:8]
            request = {
                "request_id": request_id,
                "user_id": user_id,
                "reason": reason,
                "summary": summary,
                "status": HandoffStatus.PENDING,
                "created_at": time.time(),
                "assigned_at": None,
                "resolved_at": None,
            }
            _HANDOFF_REQUESTS[request_id] = request
            logger.info("转接请求已创建: %s (用户: %s, 原因: %s)", request_id, user_id, reason)
            return (
                f"转接请求已提交（编号：{request_id}）。\n"
                "人工客服将在工作时间内尽快与您联系。\n"
                f"转接原因：{reason}\n"
                + (f"补充说明：{summary}" if summary else "")
            )

    async def check_handoff_status(self, user_id: str) -> str:
        user_requests = [
            (rid, r) for rid, r in _HANDOFF_REQUESTS.items()
            if r.get("user_id") == user_id
        ]
        if not user_requests:
            return "您当前没有转接请求。"

        lines = ["您的转接请求："]
        for rid, r in sorted(user_requests, key=lambda x: x[1]["created_at"], reverse=True):
            label = _STATUS_LABELS.get(r["status"], r["status"])
            created = time.strftime("%m-%d %H:%M", time.localtime(r["created_at"]))
            lines.append(f"  [{rid}] 状态：{label}，提交时间：{created}，原因：{r['reason']}")
        return "\n".join(lines)

    async def cancel_handoff(self, user_id: str) -> str:
        async with _HANDOFF_LOCK:
            for rid, r in list(_HANDOFF_REQUESTS.items()):
                if r.get("user_id") == user_id and r["status"] == HandoffStatus.PENDING:
                    r["status"] = HandoffStatus.CANCELLED
                    r["resolved_at"] = time.time()
                    logger.info("转接请求已取消: %s (用户: %s)", rid, user_id)
                    return f"转接请求 {rid} 已取消。如有需要，可随时重新发起转接。"
        return "您当前没有可取消的待处理转接请求。"
