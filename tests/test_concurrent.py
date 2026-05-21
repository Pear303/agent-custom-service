"""并发测试：验证 5 个用户同时提交需求无状态冲突"""
import asyncio
import json
import time
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.factory import create_agent
from agent.lc_tools import set_workspace, set_skills_loader, set_todo_store, set_subagent_deps


async def test_concurrent_agents():
    """测试 5 个用户同时创建 LCAgent 实例"""
    user_ids = ["user_a", "user_b", "user_c", "user_d", "user_e"]
    
    # 并发创建 5 个 agent 实例
    agents = await asyncio.gather(*[
        asyncio.to_thread(create_agent, user_id=uid)
        for uid in user_ids
    ])
    
    # 验证每个 agent 的状态隔离
    for i, (uid, agent) in enumerate(zip(user_ids, agents)):
        # 验证 memory_store 路径包含 user_id
        assert uid in str(agent.memory_store.memory_dir), \
            f"User {uid}: memory_dir should contain user_id, got {agent.memory_store.memory_dir}"
        
        # 验证 todo_store 的 user_id
        assert agent.todo_store.user_id == uid, \
            f"User {uid}: todo_store.user_id should be {uid}, got {agent.todo_store.user_id}"
        
        # 验证 token_tracker 路径包含 user_id
        assert uid in str(agent.token_tracker.log_file), \
            f"User {uid}: token_log_path should contain user_id, got {agent.token_tracker.log_file}"
        
        print(f"[PASS] User {uid}: state isolation verified")
    
    print("\n[PASS] All 5 agents created with isolated state")


async def test_concurrent_context_vars():
    """测试 ContextVar 上下文隔离"""
    from agent.lc_tools import _ctx_workspace, _ctx_skills_loader
    from pathlib import Path
    
    results = {}
    
    async def set_and_get_context(user_id):
        """设置并读取上下文"""
        workspace = Path(f"/tmp/{user_id}")
        set_workspace(workspace)
        set_skills_loader(f"loader_{user_id}")
        
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        # 读取当前上下文
        results[user_id] = {
            "workspace": _ctx_workspace.get(),
            "skills_loader": _ctx_skills_loader.get(),
        }
    
    # 并发设置 5 个用户的上下文
    await asyncio.gather(*[
        set_and_get_context(f"user_{i}")
        for i in range(5)
    ])
    
    # 验证每个用户的上下文隔离
    for i in range(5):
        uid = f"user_{i}"
        expected_workspace = Path(f"/tmp/{uid}")
        expected_loader = f"loader_{uid}"
        
        assert results[uid]["workspace"] == expected_workspace, \
            f"User {uid}: workspace should be {expected_workspace}, got {results[uid]['workspace']}"
        assert results[uid]["skills_loader"] == expected_loader, \
            f"User {uid}: skills_loader should be {expected_loader}, got {results[uid]['skills_loader']}"
        
        print(f"[PASS] User {uid}: ContextVar isolation verified")
    
    print("\n[PASS] ContextVar isolation test passed")


async def test_concurrent_tool_calls():
    """测试并发工具调用互不干扰"""
    import tempfile
    from agent.lc_tools import read_file, write_file
    
    results = {}
    
    async def tool_test(user_id):
        """每个用户写入和读取自己的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            set_workspace(workspace)
            
            # 写入文件
            test_file = workspace / f"{user_id}.txt"
            test_file.write_text(f"Hello from {user_id}", encoding="utf-8")
            
            # 模拟异步操作
            await asyncio.sleep(0.1)
            
            # 读取文件
            result = read_file.invoke({"path": f"{user_id}.txt"})
            results[user_id] = result
    
    # 并发执行 5 个用户的工具调用
    await asyncio.gather(*[
        tool_test(f"user_{i}")
        for i in range(5)
    ])
    
    # 验证每个用户读取到自己的内容
    for i in range(5):
        uid = f"user_{i}"
        expected_content = f"Hello from {uid}"
        assert expected_content in results[uid], \
            f"User {uid}: expected '{expected_content}' in result, got: {results[uid][:50]}"
        
        print(f"[PASS] User {uid}: tool call isolation verified")
    
    print("\n[PASS] Concurrent tool call isolation test passed")


async def main():
    print("=" * 60)
    print("并发测试开始")
    print("=" * 60)
    
    print("\n--- Test 1: 并发创建 LCAgent 实例 ---")
    await test_concurrent_agents()
    
    print("\n--- Test 2: ContextVar 上下文隔离 ---")
    await test_concurrent_context_vars()
    
    print("\n--- Test 3: 并发工具调用隔离 ---")
    await test_concurrent_tool_calls()
    
    print("\n" + "=" * 60)
    print("所有并发测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
