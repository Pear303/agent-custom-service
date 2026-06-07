"""LangGraph Agent 入口 —— python agent_lg.py 启动"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from agent_by_langgraph.lg_agent import LGAgent

if __name__ == "__main__":
    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    agent.run()
