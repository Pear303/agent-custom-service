from __future__ import annotations

from pathlib import Path


def test_context_builder_produces_non_empty_prompt(temp_dir, monkeypatch):
    """验证系统提示词构建器输出非空且包含关键身份信息"""
    templates_dir = temp_dir / "templates"
    templates_dir.mkdir()
    (templates_dir / "SOUL.md").write_text("# Test Agent\n身份：测试智能体", encoding="utf-8")
    (templates_dir / "USER.md").write_text("用户偏好：简洁", encoding="utf-8")

    agent_templates = templates_dir / "agent"
    agent_templates.mkdir()
    (agent_templates / "identity.md").write_text(
        "## 身份\n我是 {{ workspace }} 中的助手", encoding="utf-8"
    )
    (agent_templates / "skills_section.md").write_text(
        "## 技能\n{{ skills_summary }}", encoding="utf-8"
    )

    skills_dir = temp_dir / "skills"
    skills_dir.mkdir()

    from agent.skills import SkillsLoader
    from agent.context import ContextBuilder

    loader = SkillsLoader(skills_dir)
    builder = ContextBuilder(templates_dir, loader)

    prompt = builder.build_system_prompt()
    assert "测试智能体" in prompt
    assert "简洁" in prompt
