from __future__ import annotations

import pytest


def test_lc_agent_constructs_without_error(monkeypatch):
    """验证 LCAgent 成功构造：所有工具和 executor 初始化"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from agent.lc_agent import LCAgent

    agent = LCAgent(max_iterations=5)
    assert agent.executor is not None
    assert len(agent.tools) == 10
    tool_names = {t.name for t in agent.tools}
    for name in ("read_file", "write_file", "edit_file", "run_command",
                 "web_fetch", "load_skill", "glob_tool", "grep_tool",
                 "update_todos"):
        assert name in tool_names, f"Missing tool: {name}"


def test_lc_agent_memory_store_messages(monkeypatch, tmp_path):
    """验证 MemoryStore.messages 返回正确的 LangChain 消息格式（替代原 _format_chat_history）"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from agent.memory import MemoryStore
    from langchain_core.messages import HumanMessage, AIMessage

    user_file = tmp_path / "USER.md"
    user_file.write_text("")
    store = MemoryStore(tmp_path, user_file)
    store.append_history("user", "你好")
    store.append_history("assistant", "你好！")

    messages = store.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "你好"
    assert messages[1].content == "你好！"
