"""异步任务队列：工单后台处理与 Worker Pool"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .utils.file_manager import _save_report

if TYPE_CHECKING:
    from .core.database import Database
    from .services.agent_service import AgentService

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent


class TaskQueue:
    """异步任务队列管理器"""

    def __init__(self, db: Database, agent_service: AgentService, maxsize: int = 100, num_workers: int = 3):
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)
        self.db = db
        self.agent_service = agent_service
        self.num_workers = num_workers
        self._workers: list[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(num_workers)
        self._running = False

    async def submit(self, ticket_id: str):
        """提交工单到队列"""
        await self.queue.put(ticket_id)
        logger.info("工单 %s 已入队，当前队列大小: %d", ticket_id, self.queue.qsize())

    async def start(self):
        """启动后台 workers"""
        if self._running:
            return
        self._running = True
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(task)
        logger.info("TaskQueue 启动，%d 个 workers 开始工作", self.num_workers)

    async def stop(self):
        """停止所有 workers"""
        self._running = False
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("TaskQueue 已停止")

    async def _worker(self, name: str):
        """Worker 协程：从队列取任务并处理"""
        while self._running:
            try:
                ticket_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # 超时后回到 while 循环开头，检查是否继续运行
            
            # 如果成功获取到 ticket_id，继续处理
            async with self._semaphore:
                try:
                    await self._process_ticket(ticket_id)
                except Exception as e:
                    logger.error("[%s] 处理工单 %s 失败: %s", name, ticket_id, e, exc_info=True)
                    try:
                        await self.db.update_ticket_status(ticket_id, "failed", error=str(e))
                    except Exception:
                        pass
                finally:
                    self.queue.task_done()

    async def _process_ticket(self, ticket_id: str):
        """处理单个工单的三阶段工作流"""
        ticket = await self.db.get_ticket(ticket_id)
        if not ticket:
            logger.warning("工单 %s 不存在，跳过", ticket_id)
            return

        logger.info("[%s] 开始处理工单 %s: %s", "worker", ticket_id, ticket["project_name"])

        try:
            # Stage 1: 需求分析
            await self.db.update_ticket_status(ticket_id, "analyzing")
            analysis = await self.agent_service.analyze_requirement(
                ticket["user_id"],
                {
                    "project_name": ticket["project_name"],
                    "project_type": ticket.get("project_type", ""),
                    "description": ticket["description"],
                    "deadline": ticket.get("deadline", ""),
                    "budget": ticket.get("budget", ""),
                },
                ticket_id=ticket_id,
            )
            if analysis["status"] != "completed":
                await self.db.update_ticket_status(ticket_id, "failed", error=analysis.get("error", "需求分析失败"))
                return
            _save_report(ticket["user_id"], ticket_id, "需求分析.json", analysis["data"])
            await self.db.update_ticket_status(ticket_id, "designing", analysis=analysis["data"])
            logger.info("工单 %s 需求分析完成", ticket_id)

            # Stage 2: PRD 设计
            prd = await self.agent_service.design_prd(ticket["user_id"], analysis["data"], ticket_id=ticket_id)
            if prd["status"] != "completed":
                await self.db.update_ticket_status(ticket_id, "failed", error=prd.get("error", "PRD 设计失败"))
                return
            _save_report(ticket["user_id"], ticket_id, "PRD.json", prd["data"])
            await self.db.update_ticket_status(ticket_id, "estimating", prd=prd["data"])
            logger.info("工单 %s PRD 设计完成", ticket_id)

            # Stage 3: 成本估算
            cost = await self.agent_service.estimate_cost(ticket["user_id"], prd["data"], analysis["data"], ticket_id=ticket_id)
            if cost["status"] == "completed":
                _save_report(ticket["user_id"], ticket_id, "报价单.json", cost["data"])
                await self.db.update_ticket_status(ticket_id, "pending_development", quote=cost["data"])
                logger.info("工单 %s 成本估算完成，等待用户确认开发", ticket_id)
            else:
                await self.db.update_ticket_status(ticket_id, "completed", quote=None, error=cost.get("error", "成本估算失败"))
                logger.warning("工单 %s 成本估算失败: %s", ticket_id, cost.get("error"))

        except Exception as e:
            logger.error("工单 %s 处理异常: %s", ticket_id, e, exc_info=True)
            await self.db.update_ticket_status(ticket_id, "failed", error=str(e))
            raise
