"""LangGraph Agent 工厂 — 管理共享资源并按 user_id 创建/缓存独立实例。

Agent 实例缓存策略：
- 同一 user_id 复用同一个 LGAgent 实例，使 Checkpointer 增量更新生效
- 首轮调用传入完整上下文（_first_turn=True），后续调用只传 HumanMessage
- 缓存上限 _MAX_CACHE_SIZE，LRU 淘汰最久未使用的实例
- D9: TTL 过期机制，超过 _CACHE_TTL_S 未访问的实例自动淘汰
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_llm_cache = {}
_skills_loader_cache = None
_subagent_registry_cache = None
_cache_lock = threading.Lock()

# Agent 实例缓存：key=(user_id, ticket_id), value=LGAgent
_agent_cache: OrderedDict[str, "LGAgent"] = OrderedDict()
_MAX_CACHE_SIZE = 50

# D9: TTL 过期机制
_CACHE_TTL_S = 3600  # 1 小时
_agent_cache_timestamps: dict[str, float] = {}


def _get_llm(model: str = None):
    from agent_core.llm import DeepSeekChatOpenAI
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if model not in _llm_cache:
        _llm_cache[model] = DeepSeekChatOpenAI(
            model=model,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            streaming=True,
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "16384")),
        )
    return _llm_cache[model]


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _get_skills_loader():
    from agent_core.skills import get_skills_loader
    global _skills_loader_cache
    if _skills_loader_cache is None:
        skills_dir = Path(__file__).parent.parent / "skills"
        _skills_loader_cache = get_skills_loader(skills_dir)
    return _skills_loader_cache


def _get_subagent_registry():
    from agent_core.subagents.registry import SubagentRegistry
    global _subagent_registry_cache
    if _subagent_registry_cache is None:
        templates_dir = Path(__file__).parent.parent / "templates" / "subagents"
        _subagent_registry_cache = SubagentRegistry(
            templates_dir,
            skills_loader=_get_skills_loader(),
        )
    return _subagent_registry_cache


def _cleanup_expired_cache():
    """D9: 清理超过 TTL 的缓存实例，释放 SQLite 连接和内存。"""
    now = time.time()
    with _cache_lock:
        expired = [
            k for k, t in _agent_cache_timestamps.items()
            if now - t > _CACHE_TTL_S
        ]
        for k in expired:
            evicted = _agent_cache.pop(k, None)
            _agent_cache_timestamps.pop(k, None)
            if evicted is not None:
                evicted.close()
                logger.info("[Factory] Agent 缓存 TTL 过期淘汰: %s", k)


def create_lg_agent(
    user_id: str,
    ticket_id: str | None = None,
    model: str = None,
    max_iterations: int = 50,
):
    """为指定用户获取或创建 LGAgent 实例。

    同一 user_id 复用实例，使 Checkpointer 增量更新生效：
    - 首次调用：创建新实例，_first_turn=True
    - 后续调用：返回缓存实例，_first_turn=False（由 _invoke_lg_async 管理）

    共享 LLM / SkillsLoader / SubagentRegistry 实例，只隔离 MemoryStore / TodoStore / Checkpointer。
    """
    from agent_by_langgraph.lg_agent import LGAgent
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    cache_key = f"{user_id}:{ticket_id or ''}"

    # D9: 清理过期缓存
    _cleanup_expired_cache()

    # 第一次检查（快速路径，无锁）
    with _cache_lock:
        if cache_key in _agent_cache:
            _agent_cache.move_to_end(cache_key)
            _agent_cache_timestamps[cache_key] = time.time()  # D9: 更新访问时间
            agent = _agent_cache[cache_key]
            logger.info(
                "[Factory] Agent 缓存命中: user_id=%s, first_turn=%s, checkpointer_ready=%s",
                user_id, agent._first_turn, agent.checkpointer_ready,
            )
            return agent

    # 缓存未命中：在锁外创建新实例（耗时操作，避免长时间持锁）
    llm = _get_llm(model)
    skills = _get_skills_loader()
    sub_reg = _get_subagent_registry()

    agent = LGAgent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=model,
        max_iterations=max_iterations,
        llm=llm,
        skills_loader=skills,
        sub_reg=sub_reg,
    )

    # Double-check：在锁内再次检查，防止并发创建同一 user_id 的实例
    with _cache_lock:
        if cache_key in _agent_cache:
            # 另一个线程已创建，丢弃本实例，返回已有的
            agent.close()
            _agent_cache.move_to_end(cache_key)
            logger.info("[Factory] Agent 并发创建冲突，复用已有实例: user_id=%s", user_id)
            return _agent_cache[cache_key]

        _agent_cache[cache_key] = agent
        _agent_cache.move_to_end(cache_key)
        _agent_cache_timestamps[cache_key] = time.time()  # D9: 记录创建时间
        # LRU 淘汰
        while len(_agent_cache) > _MAX_CACHE_SIZE:
            evicted_key, evicted = _agent_cache.popitem(last=False)
            evicted.close()  # 释放 Checkpointer SQLite 连接
            logger.info("[Factory] Agent 缓存淘汰: %s", evicted_key)

    # 区分"无 checkpointer"和"延迟初始化待完成"
    cp_status = "ready" if agent.checkpointer_ready else (
        "lazy" if agent.will_have_checkpointer else "none"
    )
    logger.info(
        "[Factory] Agent 新建: user_id=%s, checkpointer=%s, 缓存大小=%d",
        user_id, cp_status, len(_agent_cache),
    )
    return agent


def reset_lg_agent(user_id: str, ticket_id: str | None = None):
    """重置指定用户的 Agent 缓存（强制下次创建新实例）。

    适用于：用户会话结束、需要清除 Checkpointer 状态等场景。
    """
    cache_key = f"{user_id}:{ticket_id or ''}"
    with _cache_lock:
        evicted = _agent_cache.pop(cache_key, None)
        _agent_cache_timestamps.pop(cache_key, None)  # D9: 同步清理时间戳
        if evicted is not None:
            evicted.close()  # 释放 Checkpointer SQLite 连接
            logger.info("[Factory] Agent 缓存已清除: user_id=%s", user_id)
