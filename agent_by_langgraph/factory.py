"""LangGraph Agent 工厂 — 管理共享资源并按 user_id 创建/缓存独立实例。

Agent 实例缓存策略：
- 同一 user_id 复用同一个 LangGraphAgent 实例，使 Checkpointer 增量更新生效
- 首轮调用传入完整上下文（_first_turn=True），后续调用只传 HumanMessage
- 缓存上限 _MAX_CACHE_SIZE，LRU 淘汰最久未使用的实例
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_llm_cache = {}
_skills_loader_cache = None
_subagent_registry_cache = None
_cache_lock = threading.Lock()

# Agent 实例缓存：key=(user_id, ticket_id), value=LangGraphAgent
_agent_cache: OrderedDict[str, "LangGraphAgent"] = OrderedDict()
_MAX_CACHE_SIZE = 50


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
    """为指定用户获取或创建 LangGraphAgent 实例。

    同一 user_id 复用实例，使 Checkpointer 增量更新生效：
    - 首次调用：创建新实例，_first_turn=True
    - 后续调用：返回缓存实例，_first_turn=False（由 _invoke_lg_in_thread 管理）

    共享 LLM / SkillsLoader / SubagentRegistry 实例，只隔离 MemoryStore / TodoStore / Checkpointer。
    """
    from agent_by_langgraph.lg_agent import LangGraphAgent
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    cache_key = f"{user_id}:{ticket_id or ''}"

    with _cache_lock:
        if cache_key in _agent_cache:
            # 缓存命中：移动到末尾（LRU）
            _agent_cache.move_to_end(cache_key)
            agent = _agent_cache[cache_key]
            logger.info(
                "[Factory] Agent 缓存命中: user_id=%s, first_turn=%s, checkpointer=%s",
                user_id, agent._first_turn, agent.graph.checkpointer is not None,
            )
            return agent

    # 缓存未命中：创建新实例
    with _cache_lock:
        llm = _get_llm(model)
        skills = _get_skills_loader()
        sub_reg = _get_subagent_registry()

    agent = LangGraphAgent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=model,
        max_iterations=max_iterations,
        llm=llm,
        skills_loader=skills,
        sub_reg=sub_reg,
    )

    with _cache_lock:
        _agent_cache[cache_key] = agent
        _agent_cache.move_to_end(cache_key)
        # LRU 淘汰
        while len(_agent_cache) > _MAX_CACHE_SIZE:
            evicted_key, evicted = _agent_cache.popitem(last=False)
            evicted.close()  # 释放 Checkpointer SQLite 连接
            logger.info("[Factory] Agent 缓存淘汰: %s", evicted_key)

    logger.info(
        "[Factory] Agent 新建: user_id=%s, checkpointer=%s, 缓存大小=%d",
        user_id, agent.graph.checkpointer is not None, len(_agent_cache),
    )
    return agent


def reset_lg_agent(user_id: str, ticket_id: str | None = None):
    """重置指定用户的 Agent 缓存（强制下次创建新实例）。

    适用于：用户会话结束、需要清除 Checkpointer 状态等场景。
    """
    cache_key = f"{user_id}:{ticket_id or ''}"
    with _cache_lock:
        evicted = _agent_cache.pop(cache_key, None)
        if evicted is not None:
            evicted.close()  # 释放 Checkpointer SQLite 连接
            logger.info("[Factory] Agent 缓存已清除: user_id=%s", user_id)
