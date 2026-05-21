"""SessionManager 持久化测试"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import Database
from api.session_manager import SessionManager


async def test_session_persistence():
    """测试会话持久化"""
    db = Database(db_path=":memory:")
    await db.init()

    sm = SessionManager(timeout_minutes=30, max_sessions=100, db=db)

    # 创建会话
    session = await sm.get_or_create_async("user_1")
    session.conversation_id = "conv_123"
    session.message_count = 5
    await sm._save_session(session)

    # 验证数据库保存
    db_session = await db.load_session("user_1")
    assert db_session is not None
    assert db_session["conversation_id"] == "conv_123"
    assert db_session["message_count"] == 5

    # 模拟服务重启：创建新的 SessionManager 并从数据库恢复
    sm2 = SessionManager(timeout_minutes=30, max_sessions=100, db=db)
    await sm2.load_from_db()

    # 验证会话恢复
    assert "user_1" in sm2._sessions
    restored = sm2._sessions["user_1"]
    assert restored.conversation_id == "conv_123"
    assert restored.message_count == 5

    print("[PASS] Session persistence test passed")

    # 测试会话删除
    await sm2.reset_async("user_1")
    assert "user_1" not in sm2._sessions
    db_session = await db.load_session("user_1")
    assert db_session is None

    print("[PASS] Session deletion test passed")

    # 测试过期清理
    session2 = await sm2.get_or_create_async("user_2")
    session2.last_active = time.time() - 3600  # 1 小时前
    await sm2._save_session(session2)

    cleaned = await sm2.cleanup_expired()
    assert cleaned >= 1

    print("[PASS] Session cleanup test passed")

    await db.close()


async def main():
    print("=" * 60)
    print("SessionManager 持久化测试")
    print("=" * 60)

    print("\n--- Test: Session Persistence ---")
    await test_session_persistence()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
