"""人工客服转接工具"""
from __future__ import annotations


class HumanHandoffTool:
    """人工转接。支持创建/查询/取消转接请求。"""

    def create_handoff(self, user_id: str, reason: str, summary: str = "") -> str:
        return f"转接请求已提交（待实现）"

    def check_handoff_status(self, user_id: str) -> str:
        return "排队中（待实现）"

    def cancel_handoff(self, user_id: str) -> str:
        return "排队已取消（待实现）"
