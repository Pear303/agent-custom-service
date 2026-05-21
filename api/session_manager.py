"""多用户会话管理器 —— 支持过期清理 + 容量限制 + SQLite 持久化"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.database import Database


@dataclass
class Session:
    user_id: str
    conversation_id: str | None = None
    history: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0

    def touch(self):
        self.last_active = time.time()
        self.message_count += 1


class SessionManager:
    def __init__(self, timeout_minutes: int = 30, max_sessions: int = 1000, db: Database | None = None):
        self.timeout_minutes = timeout_minutes
        self.max_sessions = max_sessions
        self.db = db
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    async def load_from_db(self):
        """启动时，从数据库恢复活跃会话"""
        if not self.db:
            return
        sessions = await self.db.get_active_sessions(self.timeout_minutes)
        for s in sessions:
            self._sessions[s["user_id"]] = Session(
                user_id=s["user_id"],
                conversation_id=s.get("conversation_id"),
                history=s.get("history") or [],
                last_active=self._parse_timestamp(s.get("last_active")),
                message_count=s.get("message_count", 0),
            )

    async def _save_session(self, session: Session):
        """保存单个会话到数据库"""
        if not self.db:
            return
        await self.db.save_session(
            session.user_id,
            session.conversation_id,
            session.history,
            session.message_count,
        )

    async def _delete_session(self, user_id: str):
        """从数据库删除会话"""
        if not self.db:
            return
        await self.db.delete_session(user_id)

    async def cleanup_expired(self) -> int:
        """清理过期会话（内存 + 数据库）"""
        with self._lock:
            now = time.time()
            cutoff = now - self.timeout_minutes * 60
            expired = [uid for uid, s in self._sessions.items() if s.last_active < cutoff]
            for uid in expired:
                del self._sessions[uid]
        
        db_cleaned = 0
        if self.db:
            db_cleaned = await self.db.cleanup_expired_sessions(self.timeout_minutes)
        
        return len(expired) + db_cleaned

    def get_or_create(self, user_id: str) -> Session:
        with self._lock:
            if user_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    self._evict_expired()
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                    del self._sessions[oldest.user_id]
                self._sessions[user_id] = Session(user_id=user_id)
            session = self._sessions[user_id]
            session.touch()
            return session

    async def get_or_create_async(self, user_id: str) -> Session:
        """异步版本：获取或创建会话，并持久化到数据库"""
        session = self.get_or_create(user_id)
        await self._save_session(session)
        return session

    def reset(self, user_id: str) -> None:
        with self._lock:
            self._sessions.pop(user_id, None)

    async def reset_async(self, user_id: str) -> None:
        """异步版本：重置会话并从数据库删除"""
        self.reset(user_id)
        await self._delete_session(user_id)

    def active_count(self) -> int:
        return len(self._sessions)

    def _evict_expired(self) -> int:
        now = time.time()
        cutoff = now - self.timeout_minutes * 60
        expired = [uid for uid, s in self._sessions.items() if s.last_active < cutoff]
        for uid in expired:
            del self._sessions[uid]
        return len(expired)

    @staticmethod
    def _parse_timestamp(ts: str | None) -> float:
        """解析 SQLite 时间戳为 Unix 时间戳"""
        if not ts:
            return time.time()
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, AttributeError):
            return time.time()
