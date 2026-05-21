"""技能加载器：负责从 skills 目录加载和管理可插拔技能包。

每个技能包是一个包含 SKILL.md 文件的目录，SKILL.md 使用 YAML frontmatter 
描述元数据（名称、描述、触发条件等），Markdown 正文包含技能的具体知识和指令。
"""
from __future__ import annotations
import re
from pathlib import Path

import yaml

_skills_loader_instance: SkillsLoader | None = None


class SkillsLoader:
    """技能加载器类。
    
    负责扫描 skills 目录，解析所有 SKILL.md 文件，提供技能的查询、
    加载和摘要生成功能。支持 always 标记的技能自动激活。
    """
    
    def __init__(self, skills_dir: Path):
        """初始化技能加载器。
        
        Args:
            skills_dir: 技能包根目录路径
        """
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        """扫描 skills 目录并加载所有 SKILL.md 文件。
        
        递归查找所有名为 SKILL.md 的文件，解析其 frontmatter 和正文，
        以技能名称为键存储到 self.skills 字典中。
        """
        if not self.skills_dir.exists():
            return
            
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """解析 Markdown 文件的 YAML frontmatter。"""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        return meta, match.group(2).strip()

    def get_content(self, skill_name: str) -> str:
        """获取指定技能的完整内容。"""
        skill = self.skills.get(skill_name)
        if not skill:
            return f"Error: Unknown skill '{skill_name}'. Available: {', '.join(self.skills.keys())}"
        return f'<skill name="{skill_name}">\n{skill["body"]}\n</skill>'

    def get_always_skills(self) -> list[str]:
        """获取所有标记为 always 的技能名称列表。"""
        return [name for name, skill in self.skills.items() if skill["meta"].get("always", False)]

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """批量加载指定技能的内容。"""
        parts = []
        for skill_name in skill_names:
            content = self.get_content(skill_name=skill_name)
            if not content.startswith("Error:"):
                parts.append(content)
        return "\n\n".join(parts) if parts else ""

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        """生成所有可用技能的摘要列表。"""
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
