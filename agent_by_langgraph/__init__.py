"""agent_by_langgraph —— LangGraph StateGraph 实现的智能代理包。

共享基础设施（MemoryStore, ContextBuilder 等）从 `agent_core` 模块导入，
仅重写 Agent 循环层为 LangGraph StateGraph。

本包通过 agent_service.py 投放到生产 API 路径，
覆盖需求分析、PRD、成本估算、项目开发等业务场景。
同时可由 agent.py（独立 CLI 入口）直接使用。

级别路由：
    LevelRouter 根据用户输入判断任务复杂度（级别 1-6），
    按级别选择不同的系统提示词和工具集。
"""
from agent_core.compactor import Compactor
from agent_core.context import ContextBuilder
from agent_core.memory import MemoryStore
from agent_core.skills import SkillsLoader
from agent_core.telemetry import TokenTracker
from agent_core.todo import TodoStore
from agent_by_langgraph.factory import create_lg_agent
from agent_by_langgraph.lg_agent import LGAgent
from agent_by_langgraph.level_router import LevelRouter, TaskLevel, LevelConfig

__all__ = [
    "LGAgent",
    "create_lg_agent",
    "LevelRouter",
    "TaskLevel",
    "LevelConfig",
    "MemoryStore",
    "ContextBuilder",
    "SkillsLoader",
    "Compactor",
    "TokenTracker",
    "TodoStore",
]
