"""技能加载工具：load_skill。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

from langchain_core.tools import tool

from .context_vars import _ctx_skills_loader


@tool
def load_skill(name: str) -> str:
    """加载指定名称的技能（skill）。技能提供专题知识和操作指南。
    Args:
        name: 技能名称（如 "pdf", "github", "weather" 等）
    """
    skills_loader = _ctx_skills_loader.get()
    if skills_loader is None:
        return "Error: Skills loader not initialized"
    return skills_loader.get_content(name)
