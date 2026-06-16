"""Agent 实战测试脚本 — 非交互式运行，自动测试并记录输出。"""
from __future__ import annotations

import asyncio
import logging
import sys
import time

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 启用 DEBUG 级别日志
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
# 只启用 lg_graph 的 WARNING 日志（包含我们的 DEBUG 输出）
logging.getLogger("agent_by_langgraph.lg_graph").setLevel(logging.WARNING)

from agent_by_langgraph.lg_agent import LGAgent


async def test_simple_task():
    """测试简单任务：创建计算器程序。"""
    print("=" * 60)
    print("  Agent 实战测试 — 创建简单计算器程序")
    print("=" * 60)
    print()

    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    user_input = "帮我创建一个简单的Python计算器程序，支持加减乘除"

    print(f"You: {user_input}")
    print("-" * 60)

    start_time = time.time()
    full_reply = ""

    try:
        async for token in agent.arun_stream(user_input):
            full_reply += token
            print(token, end="", flush=True)
    except Exception as exc:
        print(f"\n\n[ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print()
    print("-" * 60)
    print(f"[完成] 耗时 {elapsed:.1f}s, 回复长度 {len(full_reply)} 字符")

    # 关闭资源
    agent.close()


if __name__ == "__main__":
    asyncio.run(test_simple_task())
