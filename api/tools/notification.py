"""消息推送工具"""
from __future__ import annotations


class NotificationTool:
    """消息推送。支持单发/广播、限流、记录日志。"""

    def send_notification(self, user_id: str, title: str, content: str, priority: str = "normal") -> str:
        return f"通知已发送给 {user_id}（待实现）"

    def send_broadcast(self, title: str, content: str, target_group: str = "all") -> str:
        return f"广播已发送（待实现）"
