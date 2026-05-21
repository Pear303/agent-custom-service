from __future__ import annotations

from langchain_openai import ChatOpenAI


def test_create_deepseek_llm_defaults(monkeypatch):
    """验证 DeepSeek LLM 工厂使用正确的默认配置"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    from agent.lc_agent import create_deepseek_llm

    llm = create_deepseek_llm()
    assert llm.model_name == "deepseek-v4-flash"
    assert "api.deepseek.com" in str(llm.openai_api_base)


def test_create_deepseek_llm_custom_model(monkeypatch):
    """验证可以自定义模型名称"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from agent.lc_agent import create_deepseek_llm

    llm = create_deepseek_llm(model="deepseek-chat")
    assert llm.model_name == "deepseek-chat"
