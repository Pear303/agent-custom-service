"""agent_by_langgraph —— LangGraph StateGraph 实现的智能代理包。

共享基础设施（MemoryStore, ContextBuilder 等）从 `agent` 模块导入，
仅重写 Agent 循环层为 LangGraph StateGraph。

本包已通过 agent_service.py（USE_LANGGRAPH=true）投放到生产 API 路径，
覆盖需求分析、PRD、成本估算、项目开发等业务场景。
同时仍可由 agent_lg.py（独立 CLI 入口）直接使用。
"""
from agent.compactor import Compactor
from agent.context import ContextBuilder
from agent.memory import MemoryStore
from agent.skills import SkillsLoader
from agent.telemetry import TokenTracker
from agent.todo import TodoStore
from agent_by_langgraph.factory import create_lg_agent
from agent_by_langgraph.lg_agent import LGAgent

__all__ = [
    "LGAgent",
    "create_lg_agent",
    "MemoryStore",
    "ContextBuilder",
    "SkillsLoader",
    "Compactor",
    "TokenTracker",
    "TodoStore",
]
