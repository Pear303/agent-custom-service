"""工单数据访问层"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..core.database import Database


class TicketRepo:
    """工单数据访问，封装对 Database 的工单相关操作。"""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, ticket_data: dict) -> str:
        return await self._db.create_ticket(ticket_data)

    async def get(self, ticket_id: str) -> dict | None:
        return await self._db.get_ticket(ticket_id)

    async def list_by_user(self, user_id: str, limit: int = 50) -> list[dict]:
        return await self._db.get_user_tickets(user_id, limit)

    async def update_status(self, ticket_id: str, status: str, **kwargs: Any):
        await self._db.update_ticket_status(ticket_id, status, **kwargs)
