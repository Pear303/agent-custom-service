from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_subagent_registry_loads_specs(tmp_path, monkeypatch):
    """验证 SubagentRegistry 正确加载内建子代理规格"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    templates_dir = tmp_path / "subagents"
    templates_dir.mkdir()
    (templates_dir / "quick_helper.md").write_text("你是快速助手。", encoding="utf-8")

    from agent.subagents.registry import SubagentRegistry

    reg = SubagentRegistry(templates_dir)
    names = reg.names()
    for expected in ("quick_helper", "web_researcher", "doc_analyzer",
                     "engine_executor", "validator"):
        assert expected in names, f"Missing subagent: {expected}"


def test_subagent_spec_has_tool_whitelist():
    """验证子代理规格包含工具白名单和 max_turns"""
    from agent.subagents.registry import SubagentRegistry
    from pathlib import Path

    reg = SubagentRegistry(Path("templates/subagents"))
    spec = reg.get("web_researcher")
    assert spec is not None
    assert "web_fetch" in spec.tool_names
    assert spec.max_turns >= 3
    # dispatch_subagent 绝不应在白名单中（防递归）
    assert "dispatch_subagent" not in spec.tool_names


def test_dispatch_subagent_tool_definition(monkeypatch):
    """验证 dispatch_subagent @tool 可导入且参数正确"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    from agent.lc_tools import dispatch_subagent

    assert dispatch_subagent.name == "dispatch_subagent"
    assert "agent_name" in str(dispatch_subagent.args_schema.model_json_schema())
    assert "task" in str(dispatch_subagent.args_schema.model_json_schema())
