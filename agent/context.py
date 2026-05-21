"""上下文构建器：负责组装系统提示词（System Prompt）。

整合以下内容为完整的系统提示：
- 引导文档（SOUL.md、USER.md）
- Agent 身份模板
- 长期记忆
- 技能列表和摘要
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .skills import SkillsLoader

if TYPE_CHECKING:
    from .memory import MemoryStore


class ContextBuilder:
    """系统提示词上下文构建器。
    
    负责从多个来源收集信息并组装成完整的系统提示词，
    包括引导文档、身份定义、长期记忆和技能信息。
    """
    
    # 启动时必须加载的引导文件列表
    _BOOTSTRAP_FILES = ["SOUL.md", "USER.md"]

    def __init__(
        self,
        docs_dir: Path,
        skills_loader: SkillsLoader,
        memory: MemoryStore | None = None,
    ):
        """初始化上下文构建器。
        
        Args:
            docs_dir: 文档目录路径，包含模板文件和引导文档
            skills_loader: 技能加载器实例，用于获取技能信息
            memory: 可选的记忆存储管理器实例
        """
        self.docs_dir = docs_dir
        self.skills = skills_loader
        self.memory = memory
        
        # 初始化 Jinja2 模板引擎，从 agent 子目录加载模板
        self._env = Environment(
            loader=FileSystemLoader(docs_dir / "agent"),
            autoescape=select_autoescape(enabled_extensions=("html",)),
        )

    def render_template(self, template_name: str, **kwargs) -> str:
        """渲染指定的 Jinja2 模板文件。
        
        【调用方】context.py (内部调用)
        
        Args:
            template_name: 模板文件名（相对于 agent 目录）
            **kwargs: 传递给模板的变量
            
        Returns:
            渲染后的字符串，如果出错则返回空字符串
        """
        try:
            template = self._env.get_template(template_name)
            return template.render(**kwargs)
        except Exception:
            return ""

    def build_system_prompt(self) -> str:
        """构建完整的系统提示词。
        
        【调用方】lc_agent.py, tests/test_context.py
        
        按顺序组装以下内容块：
        1. 引导文档（SOUL.md + USER.md）
        2. Agent 身份信息（identity.md 模板）
        3. 长期记忆内容（如果存在）
        4. 始终激活的技能详情（always_skills）
        5. 其他可用技能的摘要列表
        
        Returns:
            完整的系统提示词字符串，各部分用分隔线隔开
        """
        parts: list[str] = []

        # 1. 加载引导文档（SOUL.md 和 USER.md）
        bootstrap = "\n\n".join(
            (self.docs_dir / name).read_text(encoding="utf-8").strip()
            for name in self._BOOTSTRAP_FILES
            if (self.docs_dir / name).exists()
        )
        if bootstrap:
            parts.append(bootstrap)

        # 2. 渲染 Agent 身份模板，注入工作区路径
        identity = self.render_template("identity.md", workspace=str(self.docs_dir.parent))
        if identity:
            parts.append(identity)

        # 3. 附加长期记忆内容（如果已配置 memory）
        if self.memory:
            memory = self.memory.read_memory().strip()
            if memory:
                parts.append(f"# Long-term Memory\n\n{memory}")

        # 4. 加载始终激活的技能详情
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 5. 生成其他可用技能的摘要列表（排除已加载的 always_skills）
        skills_summary = self.skills.build_skills_summary(exclude=set(always_skills))
        if skills_summary:
            parts.append(
                self.render_template("skills_section.md", skills_summary=skills_summary)
            )

        # 使用分隔线连接所有部分
        result = "\n\n---\n\n".join(parts)
        # 转义花括号：防止 LangChain ChatPromptTemplate 将文档中的 {xxx} 误解析为变量占位符
        return result.replace("{", "{{").replace("}", "}}")