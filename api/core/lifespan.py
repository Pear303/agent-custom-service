"""FastAPI 应用生命周期管理"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import Database
from ..services.session_manager import SessionManager
from ..task_queue import TaskQueue

logger = logging.getLogger(__name__)


def create_lifespan(db: Database, session_manager: SessionManager, task_queue: TaskQueue):
    """创建 FastAPI lifespan 上下文管理器。
    
    每个组件通过参数注入，避免模块级全局变量。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.init()
        await session_manager.load_from_db()

        recovered = await db.recover_interrupted_tickets()
        if recovered:
            for r in recovered:
                logger.info("恢复中断工单 %s: %s -> %s", r["ticket_id"], r["old_status"], r["new_status"])
                if r["new_status"] == "queued":
                    await task_queue.submit(r["ticket_id"])

        await task_queue.start()
        yield
        await task_queue.stop()
        expired = await session_manager.cleanup_expired()
        if expired:
            logger.info("清理了 %d 个过期会话", expired)
        await db.close()

    return lifespan
