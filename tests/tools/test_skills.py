from __future__ import annotations

from pathlib import Path


def test_load_skill_returns_content(temp_dir, monkeypatch):
    from agent.lc_tools import load_skill
    from agent.skills import SkillsLoader

    skills_dir = temp_dir / "skills"
    skills_dir.mkdir()
    skill_dir = skills_dir / "test-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n\n"
        "This is the skill body.\n",
        encoding="utf-8",
    )

    loader = SkillsLoader(skills_dir)
    monkeypatch.setattr("agent.lc_tools._skills_loader", loader)

    result = load_skill.invoke({"skill_name": "test-skill"})
    assert "test-skill" in result
    assert "skill body" in result


def test_load_skill_unknown_returns_error(temp_dir, monkeypatch):
    from agent.lc_tools import load_skill
    from agent.skills import SkillsLoader

    skills_dir = temp_dir / "skills"
    skills_dir.mkdir()
    loader = SkillsLoader(skills_dir)
    monkeypatch.setattr("agent.lc_tools._skills_loader", loader)

    result = load_skill.invoke({"skill_name": "nonexistent"})
    assert "Error" in result
