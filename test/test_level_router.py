"""LevelRouter 单元测试 — 验证关键词匹配、LLM 分类、默认路由、级别配置。

不依赖 LLM，纯函数逻辑测试，秒级完成。
"""
from __future__ import annotations

import pytest

from agent_by_langgraph.level_router import (
    LevelConfig,
    LevelRouter,
    TaskLevel,
    _LEVEL_CONFIGS,
)


# ── 关键词匹配测试 ──────────────────────────────────────────


class TestKeywordMatch:
    """关键词快速匹配规则覆盖。"""

    def setup_method(self):
        self.router = LevelRouter(llm=None)

    # 级别 1: 文章/报告
    @pytest.mark.parametrize("text", [
        "帮我写一篇文章",
        "撰写一份市场分析报告",
        "生成需求文档",
        "写一篇论文",
        "帮忙写一个总结",
        "帮我写方案",
        "写一份计划书",
        "生成PRD",
        "写一个需求文档",
    ])
    def test_article_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.ARTICLE, f"'{text}' 应路由到 ARTICLE，实际 {config.level}"

    # 级别 2: 静态网页
    @pytest.mark.parametrize("text", [
        "开发一个静态网页",
        "制作落地页",
        "生成landing page",
        "写一个宣传页",
        "开发H5页面",
        "创建一个单页应用",
        "html页面开发",
        "css网页制作",
    ])
    def test_static_page_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.STATIC_PAGE, f"'{text}' 应路由到 STATIC_PAGE，实际 {config.level}"

    # 级别 3: 自动化脚本
    @pytest.mark.parametrize("text", [
        "写一个爬虫脚本",
        "开发自动化脚本",
        "编写Python脚本",
        "生成批处理脚本",
        "shell脚本工具",
        "写一个script",
    ])
    def test_script_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.SCRIPT, f"'{text}' 应路由到 SCRIPT，实际 {config.level}"

    # 级别 4: 网站开发
    @pytest.mark.parametrize("text", [
        "开发一个网站",
        "搭建web应用",
        "构建全栈项目",
        "开发前后端应用",
        "开发后端api服务",
        "flask应用开发",
        "fastapi服务开发",
    ])
    def test_website_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.WEBSITE, f"'{text}' 应路由到 WEBSITE，实际 {config.level}"

    # 级别 5: 小程序
    @pytest.mark.parametrize("text", [
        "开发微信小程序",
        "做一个miniprogram",
        "用taro开发小程序",
    ])
    def test_mini_app_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.MINI_APP, f"'{text}' 应路由到 MINI_APP，实际 {config.level}"

    # 级别 6: 通用程序
    @pytest.mark.parametrize("text", [
        "开发一个应用程序",
        "构建软件系统",
        "创建一个app项目",
    ])
    def test_general_dev_keywords(self, text):
        config = self.router.route(text)
        assert config.level == TaskLevel.GENERAL_DEV, f"'{text}' 应路由到 GENERAL_DEV，实际 {config.level}"


# ── 默认路由测试 ──────────────────────────────────────────────


class TestDefaultRoute:
    """无关键词匹配 + 无 LLM 时，默认路由到级别 3。"""

    def test_default_to_script(self):
        router = LevelRouter(llm=None)
        config = router.route("随便说点什么不匹配的")
        assert config.level == TaskLevel.SCRIPT

    def test_empty_input(self):
        router = LevelRouter(llm=None)
        config = router.route("")
        assert config.level == TaskLevel.SCRIPT


# ── LLM 分类测试 ──────────────────────────────────────────────


class TestLLMClassify:
    """LLM 分类逻辑（mock LLM）。"""

    def test_llm_classify_success(self):
        """LLM 返回有效级别编号时，使用 LLM 结果。"""
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="4")

        router = LevelRouter(llm=mock_llm)
        # 使用一个不触发关键词匹配的输入
        config = router.route("做一个东西")
        assert config.level == TaskLevel.WEBSITE

    def test_llm_classify_failure_fallback(self):
        """LLM 调用失败时，回退到默认级别 3。"""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        router = LevelRouter(llm=mock_llm)
        config = router.route("做一个东西")
        assert config.level == TaskLevel.SCRIPT

    def test_llm_classify_invalid_response(self):
        """LLM 返回非数字内容时，回退到默认级别 3。"""
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="我不确定")

        router = LevelRouter(llm=mock_llm)
        config = router.route("做一个东西")
        assert config.level == TaskLevel.SCRIPT

    def test_keyword_takes_priority_over_llm(self):
        """关键词匹配优先于 LLM 分类。"""
        from unittest.mock import MagicMock
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="6")

        router = LevelRouter(llm=mock_llm)
        # "写一篇文章" 应该被关键词匹配为级别 1，而非 LLM 返回的 6
        config = router.route("写一篇文章")
        assert config.level == TaskLevel.ARTICLE
        # LLM 不应被调用
        mock_llm.invoke.assert_not_called()


# ── 级别配置测试 ──────────────────────────────────────────────


class TestLevelConfig:
    """级别配置表完整性。"""

    def test_all_levels_have_config(self):
        for level in TaskLevel:
            assert level in _LEVEL_CONFIGS, f"级别 {level} 缺少配置"

    def test_config_fields(self):
        for level, config in _LEVEL_CONFIGS.items():
            assert isinstance(config, LevelConfig)
            assert config.level == level
            assert config.label, f"级别 {level} 缺少 label"
            assert config.template_name, f"级别 {level} 缺少 template_name"

    def test_is_implemented(self):
        assert LevelRouter.is_implemented(TaskLevel.ARTICLE) is True
        assert LevelRouter.is_implemented(TaskLevel.STATIC_PAGE) is True
        assert LevelRouter.is_implemented(TaskLevel.SCRIPT) is True
        assert LevelRouter.is_implemented(TaskLevel.WEBSITE) is True
        assert LevelRouter.is_implemented(TaskLevel.MINI_APP) is False
        assert LevelRouter.is_implemented(TaskLevel.GENERAL_DEV) is False

    def test_get_unimplemented_message(self):
        msg = LevelRouter.get_unimplemented_message(TaskLevel.MINI_APP)
        assert "微信小程序" in msg
        assert "尚未实现" in msg

    def test_get_config(self):
        config = LevelRouter.get_config(TaskLevel.SCRIPT)
        assert config.level == TaskLevel.SCRIPT
        assert config.label == "自动化脚本"

    def test_max_iterations(self):
        assert _LEVEL_CONFIGS[TaskLevel.ARTICLE].max_iterations == 30
        assert _LEVEL_CONFIGS[TaskLevel.WEBSITE].max_iterations == 80
