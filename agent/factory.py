"""AgentFactory: LCAgent 工厂类，管理共享资源并按 user_id 创建独立实例"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 全局缓存（进程级共享）
_llm_cache = {}
_skills_loader_cache = None
_subagent_registry_cache = None
_cache_lock = threading.Lock()


def _get_llm(model: str = None):
    """获取共享 LLM 客户端实例（按 model 缓存）"""
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
    """获取共享 OpenAI 兼容客户端"""
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def _get_skills_loader():
    """获取共享 SkillsLoader 单例"""
    from agent.skills import get_skills_loader
    global _skills_loader_cache
    if _skills_loader_cache is None:
        skills_dir = Path(__file__).parent.parent / "skills"
        _skills_loader_cache = get_skills_loader(skills_dir)
    return _skills_loader_cache


def _get_subagent_registry():
    """获取共享 SubagentRegistry 单例"""
    from agent.subagents.registry import SubagentRegistry
    global _subagent_registry_cache
    if _subagent_registry_cache is None:
        templates_dir = Path(__file__).parent.parent / "templates" / "subagents"
        _subagent_registry_cache = SubagentRegistry(
            templates_dir,
            skills_loader=_get_skills_loader(),
        )
    return _subagent_registry_cache


def create_agent(user_id: str, ticket_id: str | None = None, model: str = None, max_iterations: int = 50):
    """为指定用户创建独立的 LCAgent 实例。
    
    使用缓存的共享资源（LLM、SkillsLoader、SubagentRegistry），
    只创建按 user_id 隔离的部分：MemoryStore、TodoStore、TokenTracker
    
    Args:
        user_id: 用户唯一标识
        ticket_id: 工单唯一标识（提供时文件保存到对应工单目录）
        model: 使用的模型名称（默认 DEEPSEEK_MODEL）
        max_iterations: Agent 最大迭代次数
    
    Returns:
        LCAgent 实例
    """
    from agent.lc_agent import LCAgent
    import os
    model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    
    # 上锁，获取共享资源
    with _cache_lock:
        llm = _get_llm(model)
        skills = _get_skills_loader()
        sub_reg = _get_subagent_registry()
    
    return LCAgent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=model,
        max_iterations=max_iterations,
        llm=llm,           # 共享（HTTP 连接池）
        skills_loader=skills,  # 共享（技能元数据只读）
        sub_reg=sub_reg,       # 共享（子代理定义只读）
    )
