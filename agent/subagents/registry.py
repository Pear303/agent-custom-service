"""子代理注册表：从模板文件加载系统提示词，与代码内置的工具白名单合并。

功能：
- 管理所有预定义的子代理规格
- 从 templates/subagents/{name}.md 读取身份定义
- 自动注入可用技能摘要（如果子代理支持 load_skill）
- 提供子代理查询和描述功能
"""
from __future__ import annotations
from pathlib import Path

from .spec import SubagentSpec

"""
目前的 Agent 集群
LCAgent 主脑
├── 开发者子代理
│   ├── quick_helper
│   ├── doc_analyzer
│   ├── web_researcher
│   ├── validator
│   ├── engine_executor
│   ├── skill_manager
│   ├── document_processor
│   └── system_maintainer
│
└── 业务子代理
    ├── requirement_analyst  ← 需求分析
    ├── product_manager      ← PRD 设计
    └── cost_estimator       ← 成本估算
"""


# 工具白名单写在代码里，不放模板中
# 模板里只写身份/口吻/职责文案
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
    # ── 业务 Agent ──────────────────────────────────────────
    "requirement_analyst": {
        "description": (
            "需求分析师。将客户模糊需求转化为结构化需求简报，"
            "识别核心功能、目标用户、约束条件和风险点。"
        ),
        "tool_names": (
            "read_file", "write_file", "grep",
        ),
        "max_turns": 10,
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

# 技能 → 子代理映射
# 本项目默认映射以下skills（有部分不允许开源传播，）
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
    "weather":             [],        # 通用轻量技能，main agent 直接用，不进子代理
    "Word / DOCX":         ["doc_analyzer", "engine_executor", "document_processor"],
    "xlsx":                ["doc_analyzer", "engine_executor", "document_processor", "validator"],
}

# 子代理名称别名映射
_ALIASES = {
}

# 默认系统提示词（当模板文件不存在时使用）
_DEFAULT_PROMPT = (
    "你是一个专职处理特定任务的子代理。\n"
    "- 用工具尽快把任务完成，最后用一段简短中文向上级汇报。\n"
    "- 只汇报结论与关键信息，不要复述每一步细节。\n"
    "- 你不能再派遣其他子代理，所有任务自己使用工具完成。"
)


class SubagentRegistry:
    """子代理注册表。
    
    从 templates/subagents/{name}.md 读取 system prompt，与代码内置的
    工具白名单 / max_turns 配置合并，构造 SubagentSpec。
    
    如果提供 skills_loader，在子代理白名单含 load_skill 时，把 skills 摘要
    注入到 system prompt 末尾，让子代理知道有哪些技能可加载。
    """

    def __init__(self, templates_dir: Path, skills_loader=None):
        """初始化子代理注册表。
        
        Args:
            templates_dir: 子代理模板目录路径
            skills_loader: 可选的技能加载器，用于注入技能摘要
        """
        self.templates_dir = Path(templates_dir)
        self._skills_loader = skills_loader
        self._specs: dict[str, SubagentSpec] = {}
        self._load_all()

    def _load_all(self) -> None:
        """加载所有内置子代理规格。
        
        对每个内置子代理：
        1. 尝试从模板文件读取系统提示词
        2. 如果模板不存在，使用默认提示词
        3. 如果子代理支持 load_skill，注入相关技能摘要（基于 _SKILL_AGENT_MAP 筛选）
        4. 创建 SubagentSpec 并存储
        """
        for agent_name, cfg in _BUILTIN_SPECS.items():
            # 读取模板文件中的系统提示词
            prompt_file = self.templates_dir / f"{agent_name}.md"
            if prompt_file.exists():
                system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            else:
                system_prompt = _DEFAULT_PROMPT

            # 如果子代理支持 load_skill，注入相关技能摘要
            if self._skills_loader and "load_skill" in cfg["tool_names"]:
                # 使用 _SKILL_AGENT_MAP 筛选该子代理相关的技能
                relevant_skills = self._build_relevant_skills_summary(agent_name)
                if relevant_skills:
                    system_prompt += (
                        "\n\n## 相关技能 (load_skill)\n\n"
                        f"{relevant_skills}\n\n"
                        "以上是与当前工作相关的技能。遇到对应专题时，先调 load_skill 把技能内容拉进上下文。"
                    )

            # 创建并存储子代理规格
            self._specs[agent_name] = SubagentSpec(
                description=cfg["description"],
                system_prompt=system_prompt,
                tool_names=tuple(cfg["tool_names"]),
                max_turns=cfg["max_turns"],
            )

    def _build_relevant_skills_summary(self, agent_name: str) -> str:
        """基于 _SKILL_AGENT_MAP 构建该子代理相关的技能摘要。
        
        只列出映射到该子代理的技能，避免注入无关技能耗费上下文。
        
        Args:
            agent_name: 子代理名称
            
        Returns:
            格式化的技能摘要字符串，如果没有相关技能则返回空字符串
        """
        if not self._skills_loader:
            return ""

        # 找出映射到该子代理的所有技能名称
        mapped_skills = {
            skill_name
            for skill_name, agents in _SKILL_AGENT_MAP.items()
            if agent_name in agents
        }
        if not mapped_skills:
            return ""

        # 用 skills_loader 构建摘要（只取 mapped 的技能）
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
        """解析子代理名称（处理别名）。
        
        Args:
            name_or_alias: 子代理名称或别名
            
        Returns:
            实际的子代理名称
        """
        return _ALIASES.get(name_or_alias, name_or_alias)

    def get(self, agent_name: str) -> SubagentSpec | None:
        """获取指定子代理的规格。
        
        【调用方】lc_tools.py
        
        Args:
            agent_name: 子代理名称
            
        Returns:
            SubagentSpec 对象，如果不存在则返回 None
        """
        return self._specs.get(self.resolve_name(agent_name))

    def names(self, *, include_aliases: bool = False) -> list[str]:
        """获取所有子代理名称列表。
        
        【调用方】lc_tools.py, tests/test_subagents.py
        
        Args:
            include_aliases: 是否包含别名
            
        Returns:
            排序后的名称列表
        """
        names = set(self._specs.keys())
        if include_aliases:
            names.update(_ALIASES.keys())
        return sorted(names)

    def aliases(self) -> dict[str, str]:
        """获取所有别名映射。
        
        Returns:
            别名到实际名称的字典
        """
        return dict(_ALIASES)

    def describe(self) -> str:
        """生成所有可用子代理的描述文本，供主 agent 工具的 description 使用。
        
        【调用方】lc_tools.py (内部使用)
        
        Returns:
            格式化的子代理列表字符串，每行一个子代理
        """
        lines = [
            f"  - {spec.name}: {spec.description}"
            for spec in self._specs.values()
        ]
        if _ALIASES:
            alias_text = ", ".join(f"{k} -> {v}" for k, v in sorted(_ALIASES.items()))
            lines.append(f"  - 兼容别名: {alias_text}")
        return "\n".join(lines)