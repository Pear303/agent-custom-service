"""级别路由器 — 根据任务复杂度选择处理管线。

级别定义：
    1: 文章/报告撰写   ← 最成熟，专精
    2: 静态网页开发    ← 已验证可行，标准化模板
    3: 自动化脚本      ← 有基础，缺编排能力
    ── 以上是当前系统的舒适区 ──
    4: 网站开发        ← 需要前后端脚手架
    5: 微信小程序      ← 预留扩展点（空）
    6: 通用程序开发    ← 预留扩展点（空）

路由策略：
    - 使用 LLM 对用户输入做意图分类，输出级别编号
    - 级别 1-3 使用标准工具集 + 对应系统提示词
    - 级别 4 使用扩展工具集（含脚手架工具）+ 网站开发提示词
    - 级别 5-6 暂未实现，返回提示信息
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class TaskLevel(IntEnum):
    """任务复杂度级别。"""
    ARTICLE = 1       # 文章/报告撰写
    STATIC_PAGE = 2   # 静态网页开发
    SCRIPT = 3        # 自动化脚本
    WEBSITE = 4       # 网站开发（前后端）
    MINI_APP = 5      # 微信小程序（预留）
    GENERAL_DEV = 6   # 通用程序开发（预留）


@dataclass
class LevelConfig:
    """某个级别的配置：系统提示词模板名、工具白名单、额外约束。"""
    level: TaskLevel
    label: str
    template_name: str           # templates/ 下的系统提示词模板文件名（不含 .md）
    tool_whitelist: tuple[str, ...] = ()  # 空 = 使用全部工具
    extra_prompt: str = ""       # 追加到系统提示词末尾的额外指令
    max_iterations: int = 50     # 该级别的默认最大迭代数


# ── 级别配置表 ──────────────────────────────────────────────

_LEVEL_CONFIGS: dict[TaskLevel, LevelConfig] = {
    TaskLevel.ARTICLE: LevelConfig(
        level=TaskLevel.ARTICLE,
        label="文章/报告撰写",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 1: 文章/报告撰写）\n\n"
            "你当前专注于文章和报告撰写任务。\n"
            "- 优先使用 `write_file` 生成 Markdown 文档\n"
            "- 不需要编写代码或运行程序\n"
            "- 如果用户要求开发程序，请提示这超出了当前级别的能力范围\n"
        ),
        max_iterations=30,
    ),
    TaskLevel.STATIC_PAGE: LevelConfig(
        level=TaskLevel.STATIC_PAGE,
        label="静态网页开发",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 2: 静态网页开发）\n\n"
            "你当前专注于静态网页开发任务。\n"
            "- 生成 HTML/CSS/JavaScript 文件\n"
            "- 不涉及后端逻辑或数据库\n"
            "- 使用 `write_file` 创建文件，`run_command` 预览效果\n"
            "- 如果用户要求后端功能，请提示这超出了当前级别\n"
        ),
    ),
    TaskLevel.SCRIPT: LevelConfig(
        level=TaskLevel.SCRIPT,
        label="自动化脚本",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 3: 自动化脚本）\n\n"
            "你当前专注于自动化脚本开发任务。\n"
            "- 生成 Python/Shell 等脚本文件\n"
            "- 可以使用 `run_command` 执行和测试脚本\n"
            "- 不涉及复杂的项目结构或多个模块\n"
        ),
    ),
    TaskLevel.WEBSITE: LevelConfig(
        level=TaskLevel.WEBSITE,
        label="网站开发",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 4: 网站开发）\n\n"
            "你当前专注于网站开发任务（含前后端）。\n"
            "- 可以生成前端页面和后端代码\n"
            "- 使用 `run_command` 安装依赖、启动服务\n"
            "- 注意项目结构的合理性\n"
            "- 优先使用成熟框架（如 Flask/FastAPI 后端 + 简单前端）\n"
        ),
        max_iterations=80,
    ),
    TaskLevel.MINI_APP: LevelConfig(
        level=TaskLevel.MINI_APP,
        label="微信小程序",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 5: 微信小程序）\n\n"
            "⚠️ 微信小程序开发功能尚未实现。\n"
            "当前系统暂不支持小程序开发，请关注后续更新。\n"
        ),
    ),
    TaskLevel.GENERAL_DEV: LevelConfig(
        level=TaskLevel.GENERAL_DEV,
        label="通用程序开发",
        template_name="SYSTEM",
        extra_prompt=(
            "\n\n---\n\n# 任务级别约束（级别 6: 通用程序开发）\n\n"
            "⚠️ 通用程序开发功能尚未实现。\n"
            "当前系统暂不支持完整的通用程序开发，请关注后续更新。\n"
        ),
    ),
}

# ── 关键词快速匹配规则 ──────────────────────────────────────

_KEYWORD_RULES: list[tuple[re.Pattern, TaskLevel]] = [
    # 级别 1: 文章/报告
    (re.compile(r"(写|撰写|生成|帮忙写|帮我写).*(文章|报告|文档|论文|总结|方案|计划书|需求文档|PRD|简报)", re.I), TaskLevel.ARTICLE),
    # 级别 2: 静态网页
    (re.compile(r"(写|开发|制作|生成|创建).*(静态网页|落地页|landing|宣传页|H5|单页)", re.I), TaskLevel.STATIC_PAGE),
    (re.compile(r"(html|css|javascript).*(页面|网页|网站)", re.I), TaskLevel.STATIC_PAGE),
    # 级别 3: 自动化脚本
    (re.compile(r"(写|开发|编写|生成).*(脚本|script|爬虫|批处理|自动化)", re.I), TaskLevel.SCRIPT),
    (re.compile(r"(python|shell|bash).*(脚本|script|工具)", re.I), TaskLevel.SCRIPT),
    # 级别 4: 网站开发
    (re.compile(r"(开发|搭建|构建).*(网站|web应用|全栈|前后端|后端|api|服务)", re.I), TaskLevel.WEBSITE),
    (re.compile(r"(flask|fastapi|django|express|node).*(应用|服务|项目)", re.I), TaskLevel.WEBSITE),
    # 级别 5: 小程序
    (re.compile(r"(微信|小程序|miniprogram|taro)", re.I), TaskLevel.MINI_APP),
    # 级别 6: 通用程序
    (re.compile(r"(开发|构建|创建).*(应用|程序|软件|系统|项目|app|project)", re.I), TaskLevel.GENERAL_DEV),
]


class LevelRouter:
    """级别路由器：根据用户输入判断任务级别，返回对应配置。"""

    def __init__(self, llm=None):
        """
        Args:
            llm: 可选的 LLM 实例，用于意图分类。
                 为 None 时仅使用关键词匹配。
        """
        self.llm = llm

    def route(self, user_input: str) -> LevelConfig:
        """根据用户输入判断任务级别。

        路由策略：
            1. 先尝试关键词快速匹配
            2. 关键词无匹配时，若有 LLM 则用 LLM 分类
            3. 都没有时，默认级别 3（自动化脚本）

        Args:
            user_input: 用户的原始输入

        Returns:
            LevelConfig: 对应级别的配置
        """
        # 步骤1: 关键词快速匹配
        level = self._keyword_match(user_input)
        if level is not None:
            logger.info("[LevelRouter] 关键词匹配 → 级别 %d (%s)", level, _LEVEL_CONFIGS[level].label)
            return _LEVEL_CONFIGS[level]

        # 步骤1.5: 短输入快速路径（≤30 字且无关键词匹配，默认级别 3，跳过 LLM）
        _SHORT_INPUT_THRESHOLD = 30
        if len(user_input) <= _SHORT_INPUT_THRESHOLD:
            default = TaskLevel.SCRIPT
            logger.info("[LevelRouter] 短输入快速路径 → 级别 %d (%s)", default, _LEVEL_CONFIGS[default].label)
            return _LEVEL_CONFIGS[default]

        # 步骤2: LLM 分类（如果可用）
        if self.llm is not None:
            level = self._llm_classify(user_input)
            if level is not None:
                logger.info("[LevelRouter] LLM 分类 → 级别 %d (%s)", level, _LEVEL_CONFIGS[level].label)
                return _LEVEL_CONFIGS[level]

        # 步骤3: 默认级别 3
        default = TaskLevel.SCRIPT
        logger.info("[LevelRouter] 默认 → 级别 %d (%s)", default, _LEVEL_CONFIGS[default].label)
        return _LEVEL_CONFIGS[default]

    def _keyword_match(self, user_input: str) -> TaskLevel | None:
        """关键词快速匹配。"""
        for pattern, level in _KEYWORD_RULES:
            if pattern.search(user_input):
                return level
        return None

    def _llm_classify(self, user_input: str) -> TaskLevel | None:
        """使用 LLM 对用户输入做意图分类。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        classify_prompt = (
            "你是一个任务分类器。根据用户的输入，判断任务属于哪个级别：\n\n"
            "1 - 文章/报告撰写（纯文本输出，不涉及代码开发）\n"
            "2 - 静态网页开发（HTML/CSS/JS，无后端）\n"
            "3 - 自动化脚本（Python/Shell 脚本，单文件或少量文件）\n"
            "4 - 网站开发（前后端，多文件项目）\n"
            "5 - 微信小程序\n"
            "6 - 通用程序开发（复杂软件项目）\n\n"
            "只输出级别编号（1-6），不要输出其他内容。"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=classify_prompt),
                HumanMessage(content=user_input),
            ])
            content = response.content if isinstance(response.content, str) else str(response.content)
            # 提取数字
            match = re.search(r'[1-6]', content)
            if match:
                return TaskLevel(int(match.group()))
        except Exception as exc:
            logger.warning("[LevelRouter] LLM 分类失败: %s", exc)

        return None

    @staticmethod
    def get_config(level: TaskLevel) -> LevelConfig:
        """获取指定级别的配置。"""
        return _LEVEL_CONFIGS[level]

    @staticmethod
    def is_implemented(level: TaskLevel) -> bool:
        """判断指定级别是否已实现。"""
        return level in (TaskLevel.ARTICLE, TaskLevel.STATIC_PAGE, TaskLevel.SCRIPT, TaskLevel.WEBSITE)

    @staticmethod
    def get_unimplemented_message(level: TaskLevel) -> str:
        """获取未实现级别的提示信息。"""
        config = _LEVEL_CONFIGS.get(level)
        if config:
            return f"级别 {level}（{config.label}）功能尚未实现，请关注后续更新。"
        return f"级别 {level} 功能尚未实现。"
