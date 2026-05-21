"""异步队列与数据库集成测试"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import Database
from api.task_queue import TaskQueue


async def test_db_crud():
    """测试数据库 CRUD 操作"""
    db = Database(db_path=":memory:")
    await db.init()

    ticket_id = await db.create_ticket({
        "user_id": "test_user",
        "project_name": "测试项目",
        "description": "测试描述",
    })
    assert ticket_id.startswith("TKT-"), f"Invalid ticket_id: {ticket_id}"

    ticket = await db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket["status"] == "queued"
    assert ticket["user_id"] == "test_user"

    await db.update_ticket_status(ticket_id, "analyzing")
    ticket = await db.get_ticket(ticket_id)
    assert ticket["status"] == "analyzing"

    await db.update_ticket_status(ticket_id, "completed", analysis={"key": "value"})
    ticket = await db.get_ticket(ticket_id)
    assert ticket["status"] == "completed"
    assert ticket["analysis"] == {"key": "value"}

    tickets = await db.get_user_tickets("test_user")
    assert len(tickets) == 1

    print("[PASS] Database CRUD test passed")
    await db.close()


async def test_queue_submission():
    """测试队列提交"""
    db = Database(db_path=":memory:")
    await db.init()

    class MockAgentService:
        async def analyze_requirement(self, *args, **kwargs):
            return {"status": "completed", "data": {"analysis": "mock"}}

        async def design_prd(self, *args, **kwargs):
            return {"status": "completed", "data": {"prd": "mock"}}

        async def estimate_cost(self, *args, **kwargs):
            return {"status": "completed", "data": {"cost": 1000}}

    queue = TaskQueue(db, MockAgentService(), maxsize=10, num_workers=2)
    await queue.start()

    ticket_id = await db.create_ticket({
        "user_id": "test_user",
        "project_name": "队列测试",
        "description": "测试队列处理",
    })
    await queue.submit(ticket_id)

    await asyncio.sleep(3)

    ticket = await db.get_ticket(ticket_id)
    assert ticket["status"] == "completed", f"Expected completed, got {ticket['status']}"

    print("[PASS] Queue submission test passed")
    await queue.stop()
    await db.close()


async def main():
    print("=" * 60)
    print("异步队列与数据库集成测试")
    print("=" * 60)

    print("\n--- Test 1: Database CRUD ---")
    await test_db_crud()

    print("\n--- Test 2: Queue Submission ---")
    await test_queue_submission()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
