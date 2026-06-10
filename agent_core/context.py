"""上下文构建器：负责组装系统提示词（System Prompt）。

从 agent.context 迁移，更新 import: from .skills import SkillsLoader, from .memory import MemoryStore
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .skills import SkillsLoader

if TYPE_CHECKING:
    from .memory import MemoryStore


def _get_os_info() -> str:
    os_name = platform.system()
    if os_name == "Windows":
        shell_info = "PowerShell"
        return (
            f"当前运行环境: Windows (PowerShell)\n"
            f"- 使用 PowerShell 语法执行命令（如 Get-ChildItem 而非 ls，Select-String 而非 grep）\n"
            f"- 路径使用反斜杠 \\ 或正斜杠 /\n"
            f"- 不要使用 Unix 命令如 find、cat、grep、ls、pwd 等\n"
            f"- 使用 dir 代替 ls，type 代替 cat，findstr 代替 grep"
        )
    elif os_name == "Darwin":
        return f"当前运行环境: macOS (Unix shell)"
    elif os_name == "Linux":
        return f"当前运行环境: Linux (Unix shell)"
    else:
        return f"当前运行环境: {os_name}"


class ContextBuilder:
    """系统提示词上下文构建器。"""

    _BOOTSTRAP_FILES = ["SOUL.md", "USER.md"]

    def __init__(
        self,
        docs_dir: Path,
        skills_loader: SkillsLoader,
        memory: MemoryStore | None = None,
    ):
        self.docs_dir = docs_dir
        self.skills = skills_loader
        self.memory = memory
        self._env = Environment(
            loader=FileSystemLoader(docs_dir / "agent"),
            autoescape=select_autoescape(enabled_extensions=("html",)),
        )

    def render_template(self, template_name: str, **kwargs) -> str:
        try:
            template = self._env.get_template(template_name)
            return template.render(**kwargs)
        except Exception:
            return ""

    def build_system_prompt(self) -> str:
        parts: list[str] = []

        bootstrap = "\n\n".join(
            (self.docs_dir / name).read_text(encoding="utf-8").strip()
            for name in self._BOOTSTRAP_FILES
            if (self.docs_dir / name).exists()
        )
        if bootstrap:
            parts.append(bootstrap)

        identity = self.render_template("identity.md", workspace=str(self.docs_dir.parent))
        if identity:
            parts.append(identity)

        os_info = _get_os_info()
        if os_info:
            parts.append(f"# Runtime Environment\n\n{os_info}")

        if self.memory:
            memory = self.memory.read_memory().strip()
            if memory:
                parts.append(f"# Long-term Memory\n\n{memory}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(
                self.render_template("skills_section.md", skills_summary=skills_summary)
            )

        result = "\n\n---\n\n".join(parts)
        return result.replace("{", "{{").replace("}", "}}")
