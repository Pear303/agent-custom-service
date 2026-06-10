"""Agent 入口 —— python agent.py 启动（LangGraph StateGraph 引擎）"""
from __future__ import annotations

import sys

# Windows 控制台默认 GBK，强制 UTF-8 避免子进程输出解码崩溃和打印乱码
if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent_by_langgraph.lg_agent import LGAgent

if __name__ == "__main__":
    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    agent.run()
