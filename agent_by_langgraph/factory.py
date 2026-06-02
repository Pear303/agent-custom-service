"""LangGraph Agent 工厂 — 管理共享资源并按 user_id 创建独立实例。"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_llm_cache = {}
_skills_loader_cache = None
_subagent_registry_cache = None
_cache_lock = threading.Lock()


def _get_llm(model: str = None):
    from agent.lc_agent import DeepSeekChatOpenAI
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if model not in _llm_cache:
        _llm_cache[model] = DeepSeekChatOpenAI(
            model=model,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            streaming=True,
        )
    return _llm_cache[model]


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _get_skills_loader():
    from agent.skills import get_skills_loader
    global _skills_loader_cache
    if _skills_loader_cache is None:
        skills_dir = Path(__file__).parent.parent / "skills"
        _skills_loader_cache = get_skills_loader(skills_dir)
    return _skills_loader_cache


def _get_subagent_registry():
    from agent.subagents.registry import SubagentRegistry
    global _subagent_registry_cache
    if _subagent_registry_cache is None:
        templates_dir = Path(__file__).parent.parent / "templates" / "subagents"
        _subagent_registry_cache = SubagentRegistry(
            templates_dir,
            skills_loader=_get_skills_loader(),
        )
    return _subagent_registry_cache


def create_lg_agent(
    user_id: str,
    ticket_id: str | None = None,
    model: str = None,
    max_iterations: int = 50,
):
    """为指定用户创建独立的 LangGraphAgent 实例。

    共享 LLM / SkillsLoader / SubagentRegistry 实例，只隔离 MemoryStore / TodoStore。
    """
    from agent_by_langgraph.lg_agent import LangGraphAgent
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    with _cache_lock:
        llm = _get_llm(model)
        skills = _get_skills_loader()
        sub_reg = _get_subagent_registry()

    return LangGraphAgent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=model,
        max_iterations=max_iterations,
        llm=llm,
        skills_loader=skills,
        sub_reg=sub_reg,
    )
