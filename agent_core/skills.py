"""技能加载器：负责从 skills 目录加载和管理可插拔技能包。

从 agent.skills 迁移，无 import 变更。
"""
from __future__ import annotations
import re
from pathlib import Path

import yaml

_skills_loader_instance: SkillsLoader | None = None


class SkillsLoader:
    """技能加载器类。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_content(self, skill_name: str) -> str:
        skill = self.skills.get(skill_name)
        if not skill:
            return f"Error: Unknown skill '{skill_name}'. Available: {', '.join(self.skills.keys())}"
        return f'<skill name="{skill_name}">\n{skill["body"]}\n</skill>'

    def get_always_skills(self) -> list[str]:
        return [name for name, skill in self.skills.items() if skill["meta"].get("always", False)]

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts = []
        for skill_name in skill_names:
            content = self.get_content(skill_name=skill_name)
            if not content.startswith("Error:"):
                parts.append(content)
        return "\n\n".join(parts) if parts else ""

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        if not self.skills:
            return ""
        lines = []
        for skill_name, skill in self.skills.items():
            if skill_name in exclude:
                continue
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"- **{skill_name}**: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines) if lines else ""


def get_skills_loader(skills_dir: Path | None = None) -> SkillsLoader:
    """获取全局单例技能加载器。"""
    global _skills_loader_instance
    if _skills_loader_instance is None:
        if skills_dir is None:
            skills_dir = Path(__file__).parent.parent / "skills"
        _skills_loader_instance = SkillsLoader(skills_dir)
    return _skills_loader_instance
