"""会话数据访问层"""
from __future__ import annotations

from ..core.database import Database


class SessionRepo:
    """会话数据访问，封装对 Database 的会话相关操作。"""

    def __init__(self, db: Database):
        self._db = db

    async def save(self, user_id: str, conversation_id: str | None, history: list, message_count: int):
        await self._db.save_session(user_id, conversation_id, history, message_count)

    async def load(self, user_id: str) -> dict | None:
        return await self._db.load_session(user_id)

    async def delete(self, user_id: str):
        await self._db.delete_session(user_id)
