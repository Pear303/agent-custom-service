"""SQLite 数据库层：工单持久化与异步查询"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


DB_PATH = Path(__file__).parent.parent.parent / "data" / "users" / "opc.db"


class Database:
    """异步 SQLite 数据库管理器"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._pool: aiosqlite.Connection | None = None

    async def init(self):
        """初始化数据库连接和表结构"""
        if isinstance(self.db_path, str):
            db_path = Path(self.db_path)
        else:
            db_path = self.db_path
        
        if self.db_path != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._pool = await aiosqlite.connect(self.db_path)
        await self._pool.execute("PRAGMA journal_mode=WAL")
        await self._pool.execute("PRAGMA busy_timeout=5000")
        await self._create_tables()
        await self._pool.commit()

    async def close(self):
        """关闭数据库连接"""
        if self._pool:
            await self._pool.close()

    async def _create_tables(self):
        """创建工单表和会话表及索引"""
        await self._pool.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                project_type TEXT DEFAULT '',
                description TEXT NOT NULL,
                deadline TEXT DEFAULT '',
                budget TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'queued' 
                    CHECK(status IN ('queued', 'analyzing', 'designing', 'estimating', 'completed', 'failed', 'pending_development', 'developing', 'development_completed', 'development_failed')),
                analysis JSON,
                prd JSON,
                quote JSON,
                development_status TEXT DEFAULT 'not_started'
                    CHECK(development_status IN ('not_started', 'pending', 'in_progress', 'completed', 'failed')),
                development_output JSON,
                development_error TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await self._pool.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                history JSON DEFAULT '[]',
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        """)
        await self._pool.execute("CREATE INDEX IF NOT EXISTS idx_tickets_user ON tickets(user_id)")
        await self._pool.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)")
        await self._pool.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at DESC)")
        await self._pool.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active DESC)")
        
        # 迁移：为已存在的表添加新字段（如果不存在）
        cursor = await self._pool.execute("PRAGMA table_info(tickets)")
        columns = [row[1] for row in await cursor.fetchall()]
        if 'development_status' not in columns:
            await self._pool.execute("ALTER TABLE tickets ADD COLUMN development_status TEXT DEFAULT 'not_started'")
        if 'development_output' not in columns:
            await self._pool.execute("ALTER TABLE tickets ADD COLUMN development_output JSON")
        if 'development_error' not in columns:
            await self._pool.execute("ALTER TABLE tickets ADD COLUMN development_error TEXT")

        # 迁移：将 session history 中 "role": "bot" 统一为 "role": "assistant"
        await self._migrate_bot_to_assistant()

        await self._pool.commit()

    """
    tickets:
    ticket_id TEXT PRIMARY KEY
    user_id TEXT NOT NULL
    status TEXT CHECK(status IN (...))
    analysis JSON        -- 需求分析结果
    prd JSON             -- PRD 文档
    quote JSON           -- 成本估算
    development_status TEXT  -- 开发状态
    development_output JSON  -- 开发产出
    development_error TEXT   -- 开发错误
    error TEXT
    created_at TIMESTAMP
    updated_at TIMESTAMP

    sessions:
    user_id TEXT PRIMARY KEY
    conversation_id TEXT
    history JSON
    last_active TIMESTAMP
    message_count INTEGER
    """
    
    async def create_ticket(self, ticket_data: dict) -> str:
        """创建工单，返回 ticket_id"""
        ticket_id = ticket_data.get("ticket_id") or f"TKT-{uuid.uuid4().hex[:8].upper()}"
        await self._pool.execute(
            """INSERT INTO tickets 
               (ticket_id, user_id, project_name, project_type, description, deadline, budget, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')""",
            (
                ticket_id,
                ticket_data["user_id"],
                ticket_data["project_name"],
                ticket_data.get("project_type", ""),
                ticket_data["description"],
                ticket_data.get("deadline", ""),
                ticket_data.get("budget", ""),
            ),
        )
        await self._pool.commit()

        # 工单数据写入文件系统：data/users/{user_id}/{ticket_id}/工单/工单.json
        try:
            root = Path(__file__).parent.parent.parent
            ticket_dir = root / "data" / "users" / ticket_data["user_id"] / ticket_id / "工单"
            ticket_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ticket_id": ticket_id,
                "user_id": ticket_data["user_id"],
                "project_name": ticket_data["project_name"],
                "project_type": ticket_data.get("project_type", ""),
                "description": ticket_data["description"],
                "deadline": ticket_data.get("deadline", ""),
                "budget": ticket_data.get("budget", ""),
                "status": "queued",
            }
            ticket_file = ticket_dir / "工单.json"
            ticket_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("工单文件已保存: %s", ticket_file)
        except Exception as e:
            logger.warning("保存工单文件失败: %s", e)

        return ticket_id

    async def update_ticket_status(self, ticket_id: str, status: str, **kwargs: Any):
        """更新工单状态和附加数据"""
        fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: list[Any] = [status]
        for key, value in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        values.append(ticket_id)
        await self._pool.execute(
            f"UPDATE tickets SET {', '.join(fields)} WHERE ticket_id = ?",
            values,
        )
        await self._pool.commit()

    async def get_ticket(self, ticket_id: str) -> dict | None:
        """查询单个工单"""
        cursor = await self._pool.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = await cursor.fetchone()
        if row:
            return self._row_to_dict(cursor, row)
        return None

    async def get_user_tickets(self, user_id: str, limit: int = 50) -> list[dict]:
        """查询用户的工单列表"""
        cursor = await self._pool.execute(
            "SELECT * FROM tickets WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(cursor, row) for row in rows]

    async def get_queued_tickets(self, limit: int = 10) -> list[dict]:
        """获取待处理的工单（用于重启后恢复队列）"""
        cursor = await self._pool.execute(
            "SELECT * FROM tickets WHERE status = 'queued' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(cursor, row) for row in rows]

    async def get_interrupted_tickets(self, statuses: list[str] | None = None) -> list[dict]:
        """获取服务重启时可能中断的工单"""
        statuses = statuses or ["analyzing", "designing", "estimating", "developing"]
        placeholders = ",".join("?" for _ in statuses)
        cursor = await self._pool.execute(
            f"SELECT * FROM tickets WHERE status IN ({placeholders}) ORDER BY updated_at ASC",
            tuple(statuses),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(cursor, row) for row in rows]

    async def recover_interrupted_tickets(self) -> list[dict]:
        """恢复服务重启时中断的工单，返回恢复列表"""
        interrupted = await self.get_interrupted_tickets()
        recovered = []
        for ticket in interrupted:
            old_status = ticket["status"]
            if old_status == "developing":
                new_status = "pending_development"
            else:
                new_status = "queued"
            await self._pool.execute(
                "UPDATE tickets SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE ticket_id = ?",
                (new_status, ticket["ticket_id"]),
            )
            recovered.append({
                "ticket_id": ticket["ticket_id"],
                "old_status": old_status,
                "new_status": new_status,
            })
        if recovered:
            await self._pool.commit()
        return recovered

    async def save_session(self, user_id: str, conversation_id: str | None, history: list, message_count: int):
        """保存会话数据"""
        import json
        await self._pool.execute(
            """INSERT INTO sessions (user_id, conversation_id, history, message_count, last_active)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id) DO UPDATE SET
                   conversation_id = excluded.conversation_id,
                   history = excluded.history,
                   message_count = excluded.message_count,
                   last_active = CURRENT_TIMESTAMP""",
            (user_id, conversation_id, json.dumps(history, ensure_ascii=False), message_count),
        )
        await self._pool.commit()

    async def load_session(self, user_id: str) -> dict | None:
        """加载会话数据"""
        import json
        cursor = await self._pool.execute(
            "SELECT * FROM sessions WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        if row:
            data = self._row_to_dict(cursor, row)
            if data.get("history") and isinstance(data["history"], str):
                try:
                    data["history"] = json.loads(data["history"])
                except json.JSONDecodeError:
                    data["history"] = []
            return data
        return None

    async def delete_session(self, user_id: str):
        """删除会话"""
        await self._pool.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        await self._pool.commit()

    async def get_active_sessions(self, timeout_minutes: int = 30) -> list[dict]:
        """获取活跃会话"""
        cursor = await self._pool.execute(
            "SELECT * FROM sessions WHERE last_active >= datetime('now', ?) ORDER BY last_active DESC",
            (f"-{timeout_minutes} minutes",),
        )
        rows = await cursor.fetchall()
        return [self._row_to_dict(cursor, row) for row in rows]

    async def cleanup_expired_sessions(self, timeout_minutes: int = 30) -> int:
        """清理过期会话"""
        cursor = await self._pool.execute(
            "DELETE FROM sessions WHERE last_active < datetime('now', ?)",
            (f"-{timeout_minutes} minutes",),
        )
        await self._pool.commit()
        return cursor.rowcount

    async def _migrate_bot_to_assistant(self):
        """将 sessions 表中 history JSON 里 "role": "bot" 迁移为 "role": "assistant"。
        
        一次性的向后兼容迁移，确保旧数据与新代码的 role 命名一致。
        """
        cursor = await self._pool.execute("SELECT user_id, history FROM sessions")
        rows = await cursor.fetchall()
        for user_id, history_raw in rows:
            if not history_raw:
                continue
            try:
                if isinstance(history_raw, str):
                    history = json.loads(history_raw)
                else:
                    history = history_raw
            except (json.JSONDecodeError, TypeError):
                continue

            changed = False
            for entry in history:
                if isinstance(entry, dict) and entry.get("role") == "bot":
                    entry["role"] = "assistant"
                    changed = True

            if changed:
                await self._pool.execute(
                    "UPDATE sessions SET history = ? WHERE user_id = ?",
                    (json.dumps(history, ensure_ascii=False), user_id),
                )
                logger.info("迁移 session %s: bot → assistant", user_id)

    @staticmethod
    def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict:
        """将数据库行转换为字典，自动解析 JSON 字段"""
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))
        for json_field in ["analysis", "prd", "quote", "development_output"]:
            if data.get(json_field) and isinstance(data[json_field], str):
                try:
                    data[json_field] = json.loads(data[json_field])
                except json.JSONDecodeError:
                    data[json_field] = None
        return data
