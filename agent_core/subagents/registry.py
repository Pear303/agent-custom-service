"""子代理注册表：从模板文件加载系统提示词，与代码内置的工具白名单合并。

从 agent.subagents.registry 迁移，更新 import: from .spec import SubagentSpec
"""
from __future__ import annotations
from pathlib import Path

from .spec import SubagentSpec


_BUILTIN_SPECS: dict[str, dict] = {
    "quick_helper": {
        "description": (
            "快速助手。轻量只读，适合短命令、快速确认、简单查询。"
            "若发现任务变复杂，应报告上级改派专职子代理。"
        ),
        "tool_names": (
            "run_command", "read_file", "glob", "grep",
        ),
        "max_turns": 8,
    },
    "doc_analyzer": {
        "description": (
            "文档分析器。只读文档，适合阅读代码、查阅文档、"
            "整理提纲、归纳结论。"
        ),
        "tool_names": (
            "load_skill", "read_file", "glob", "grep",
        ),
        "max_turns": 12,
    },
    "web_researcher": {
        "description": (
            "网络研究员。只读查访，适合抓网页、查资料、"
            "探索性搜索、比对外部线索。"
        ),
        "tool_names": (
            "run_command", "web_fetch", "load_skill",
            "read_file", "glob", "grep",
        ),
        "max_turns": 15,
    },
    "validator": {
        "description": (
            "校验员。只读核验，适合盘点文件、校对清单、"
            "检查遗漏、整理表册。"
        ),
        "tool_names": (
            "run_command", "load_skill",
            "read_file", "glob", "grep",
        ),
        "max_turns": 12,
    },
    "engine_executor": {
        "description": (
            "工程执行器。可读写可执行命令，适合修改文件、"
            "搭建工程、跑命令验收。"
        ),
        "tool_names": (
            "run_command", "web_fetch", "load_skill",
            "read_file", "write_file", "edit_file", "glob", "grep",
        ),
        "max_turns": 20,
    },
    "skill_manager": {
        "description": (
            "技能管理员。搜索、安装、创建和管理技能包，"
            "维护技能生态。"
        ),
        "tool_names": (
            "run_command", "load_skill",
            "read_file", "write_file", "edit_file", "glob", "grep",
        ),
        "max_turns": 15,
    },
    "document_processor": {
        "description": (
            "文档处理器。创建、编辑和转换各类文档格式，"
            "包括 PDF、PPT、Word、Excel 等。"
        ),
        "tool_names": (
            "run_command", "web_fetch", "load_skill",
            "read_file", "write_file", "edit_file", "glob", "grep",
        ),
        "max_turns": 20,
    },
    "system_maintainer": {
        "description": (
            "系统维护员。负责系统自我改进、自动更新和知识管理，"
            "长期维护系统健康状态。"
        ),
        "tool_names": (
            "run_command", "load_skill",
            "read_file", "write_file", "edit_file", "glob", "grep",
        ),
        "max_turns": 20,
    },
    "requirement_analyst": {
        "description": (
            "需求分析师。将客户模糊需求转化为结构化需求简报，"
            "识别核心功能、目标用户、约束条件和风险点。"
            "支持 RAG 知识库检索，可查询历史需求模板和行业框架。"
        ),
        "tool_names": (
            "read_file", "write_file", "grep", "rag_search", "web_fetch",
        ),
        "max_turns": 12,
        "is_rag": True,
    },
    "product_manager": {
        "description": (
            "产品经理。将需求简报转化为完整 PRD，"
            "输出功能清单、用户故事、信息架构和验收标准。"
        ),
        "tool_names": (
            "read_file", "write_file", "edit_file",
        ),
        "max_turns": 12,
    },
    "cost_estimator": {
        "description": (
            "成本估算师。基于 PRD 计算开发成本和报价，"
            "输出分项明细、付款节点和交付周期。"
        ),
        "tool_names": (
            "read_file", "write_file", "grep",
        ),
        "max_turns": 8,
    },
}

_SKILL_AGENT_MAP: dict[str, list[str]] = {
    "Agent Browser":       ["web_researcher", "engine_executor"],
    "auto-updater":        ["system_maintainer"],
    "clawhub":             ["skill_manager"],
    "ddg-search":          ["web_researcher"],
    "find-skills":         ["skill_manager"],
    "github":              ["engine_executor"],
    "ontology":            ["doc_analyzer", "system_maintainer"],
    "pdf":                 ["doc_analyzer", "engine_executor", "document_processor", "validator"],
    "pptx":                ["doc_analyzer", "engine_executor", "document_processor"],
    "self-improvement":    ["system_maintainer"],
    "skill-creator":       ["skill_manager"],
    "summarize":           ["doc_analyzer", "web_researcher", "validator"],
    "ui-ux-pro-max":       ["engine_executor", "document_processor"],
    "weather":             [],
    "Word / DOCX":         ["doc_analyzer", "engine_executor", "document_processor"],
    "xlsx":                ["doc_analyzer", "engine_executor", "document_processor", "validator"],
}

_ALIASES: dict[str, str] = {}

_DEFAULT_PROMPT = (
    "你是一个专职处理特定任务的子代理。\n"
    "- 用工具尽快把任务完成，最后用一段简短中文向上级汇报。\n"
    "- 只汇报结论与关键信息，不要复述每一步细节。\n"
    "- 你不能再派遣其他子代理，所有任务自己使用工具完成。"
)


class SubagentRegistry:
    """子代理注册表。"""

    def __init__(self, templates_dir: Path, skills_loader=None):
        self.templates_dir = Path(templates_dir)
        self._skills_loader = skills_loader
        self._specs: dict[str, SubagentSpec] = {}
        self._load_all()

    def _load_all(self) -> None:
        for agent_name, cfg in _BUILTIN_SPECS.items():
            prompt_file = self.templates_dir / f"{agent_name}.md"
            if prompt_file.exists():
                system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            else:
                system_prompt = _DEFAULT_PROMPT

            if self._skills_loader and "load_skill" in cfg["tool_names"]:
                relevant_skills = self._build_relevant_skills_summary(agent_name)
                if relevant_skills:
                    system_prompt += (
                        "\n\n## 相关技能 (load_skill)\n\n"
                        f"{relevant_skills}\n\n"
                        "以上是与当前工作相关的技能。遇到对应专题时，先调 load_skill 把技能内容拉进上下文。"
                    )

            system_prompt += self._CONCLUSION_SUFFIX
            self._specs[agent_name] = SubagentSpec(
                name=agent_name,
                description=cfg["description"],
                system_prompt=system_prompt,
                tool_names=tuple(cfg["tool_names"]),
                max_turns=cfg["max_turns"],
                is_rag=cfg.get("is_rag", False),
            )

    _CONCLUSION_SUFFIX = (
        "\n\n## 输出要求\n\n"
        "任务完成后，在回复末尾必须包含 `## 结论` 标题，"
        "简要总结你的发现和操作（包括涉及的文件路径和行号）。"
    )

    def _build_relevant_skills_summary(self, agent_name: str) -> str:
        if not self._skills_loader:
            return ""
        mapped_skills = {
            skill_name
            for skill_name, agents in _SKILL_AGENT_MAP.items()
            if agent_name in agents
        }
        if not mapped_skills:
            return ""
        lines = []
        for skill_name, skill in self._skills_loader.skills.items():
            if skill_name not in mapped_skills:
                continue
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"- **{skill_name}**: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines) if lines else ""

    def resolve_name(self, name_or_alias: str) -> str:
        return _ALIASES.get(name_or_alias, name_or_alias)

    def get(self, agent_name: str) -> SubagentSpec | None:
        return self._specs.get(self.resolve_name(agent_name))

    def names(self, *, include_aliases: bool = False) -> list[str]:
        names = set(self._specs.keys())
        if include_aliases:
            names.update(_ALIASES.keys())
        return sorted(names)

    def aliases(self) -> dict[str, str]:
        return dict(_ALIASES)

    def describe(self) -> str:
        lines = [
            f"  - {spec.name}: {spec.description}"
            for spec in self._specs.values()
        ]
        if _ALIASES:
            alias_text = ", ".join(f"{k} -> {v}" for k, v in sorted(_ALIASES.items()))
            lines.append(f"  - 兼容别名: {alias_text}")
        return "\n".join(lines)
