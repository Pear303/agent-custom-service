"""环境冒烟测试——验证依赖正确安装"""
from __future__ import annotations

import langchain_core
import langchain_community
import langgraph
import pytest


def test_langchain_core_imports():
    """验证 langchain-core 关键模块可导入"""
    assert hasattr(langchain_core, "__version__")
    from langchain_core.tools import tool  # noqa: F401
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # noqa: F401
    from langchain_core.prompts import ChatPromptTemplate  # noqa: F401


def test_langchain_openai_imports():
    """验证 langchain-openai 可导入（连接 DeepSeek 用）"""
    from langchain_openai import ChatOpenAI  # noqa: F401


def test_langgraph_imports():
    """验证 langgraph StateGraph 可导入"""
    from langgraph.graph import StateGraph, END  # noqa: F401


def test_pytest_version():
    """验证 pytest >= 7.0"""
    major = int(pytest.__version__.split(".")[0])
    assert major >= 7, f"pytest 版本 {pytest.__version__} 过低"


def test_existing_project_imports():
    """验证核心模块可导入"""
    from agent.memory import MemoryStore  # noqa: F401
    from agent.skills import SkillsLoader  # noqa: F401
