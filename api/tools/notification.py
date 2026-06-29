"""消息推送工具"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .notification_storage import _TEMPLATES

logger = logging.getLogger(__name__)

_NOTIFICATION_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "notifications.jsonl"

_NOTIFICATION_LIMITS: dict[str, dict[str, Any]] = {}
_NOTIFICATION_LOCK = asyncio.Lock()

_RATE_LIMIT_MAX = 10
_RATE_LIMIT_WINDOW = 60
_BROADCAST_COOLDOWN = 30


class NotificationTool:
    """消息推送。支持单发/广播、限流、记录日志。"""

    def __init__(self):
        self._last_broadcast: float = 0.0

    @staticmethod
    def _render_template(template_type: str, title: str, content: str) -> str:
        template = _TEMPLATES.get(template_type, "{title}\n{content}")
        return template.format(title=title, content=content)

    @staticmethod
    async def _check_rate_limit(user_id: str) -> tuple[bool, str]:
        async with _NOTIFICATION_LOCK:
            now = time.time()
            entry = _NOTIFICATION_LIMITS.get(user_id, {"count": 0, "window_start": now})
            if now - entry["window_start"] > _RATE_LIMIT_WINDOW:
                entry = {"count": 0, "window_start": now}
            entry["count"] += 1
            _NOTIFICATION_LIMITS[user_id] = entry
            if entry["count"] > _RATE_LIMIT_MAX:
                return False, f"发送频率过高（{_RATE_LIMIT_WINDOW}秒内最多 {_RATE_LIMIT_MAX} 条），请稍后再试。"
            return True, ""

    @staticmethod
    def _log_notification(record: dict[str, Any]):
        try:
            _NOTIFICATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_NOTIFICATION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write notification log: %s", e)

    async def send_notification(
        self, user_id: str, title: str, content: str, priority: str = "normal"
    ) -> str:
        if priority not in ("low", "normal", "high", "urgent"):
            priority = "normal"

        allowed, error_msg = await self._check_rate_limit(user_id)
        if not allowed:
            return error_msg

        message = self._render_template("notice", title, content)
        record = {
            "type": "single",
            "user_id": user_id,
            "title": title,
            "content": content,
            "priority": priority,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(),
        }
        self._log_notification(record)
        logger.info("通知已发送给 %s: %s (优先级: %s)", user_id, title, priority)
        return message

    async def send_broadcast(self, title: str, content: str, target_group: str = "all") -> str:
        now = time.time()
        if now - self._last_broadcast < _BROADCAST_COOLDOWN:
            remaining = int(_BROADCAST_COOLDOWN - (now - self._last_broadcast))
            return f"广播冷却中，请 {remaining} 秒后再试。"

        self._last_broadcast = now
        message = self._render_template("promotion", title, content)
        record = {
            "type": "broadcast",
            "target_group": target_group,
            "title": title,
            "content": content,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(),
        }
        self._log_notification(record)
        logger.info("广播已发送 (目标群组: %s): %s", target_group, title)
        return message
