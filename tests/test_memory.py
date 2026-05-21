from __future__ import annotations


def test_memory_store_persists_history(temp_memory_dir):
    from agent.memory import MemoryStore
    store = MemoryStore(temp_memory_dir, temp_memory_dir.parent / "USER.md")
    store.append_history("user", "hello")
    store.append_history("assistant", "hi there")

    rows = store.load_unarchived_history()
    assert len(rows) >= 2
    assert rows[-2]["role"] == "user"
    assert rows[-2]["content"] == "hello"


def test_memory_store_read_write_long_term(temp_memory_dir):
    from agent.memory import MemoryStore
    store = MemoryStore(temp_memory_dir, temp_memory_dir.parent / "USER.md")
    store.write_memory("重要：用户喜欢简洁回答")
    content = store.read_memory()
    assert "简洁回答" in content


def test_lc_agent_memory_integration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    from agent.lc_agent import LCAgent

    agent = LCAgent(max_iterations=1)
    agent.memory_store.append_history("user", "测试消息")
    rows = agent.memory_store.load_unarchived_history()
    assert any("测试消息" in str(r.get("content", "")) for r in rows)
