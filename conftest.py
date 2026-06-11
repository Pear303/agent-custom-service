"""pytest 全局配置。"""
from __future__ import annotations


def pytest_configure(config):
    """注册自定义 markers，避免 PytestUnknownMarkWarning。"""
    config.addinivalue_line("markers", "rag: RAG 集成测试（需要本地模型）")
    config.addinivalue_line("markers", "e2e: 端到端测试")
    config.addinivalue_line("markers", "integration: 集成测试（需要外部服务）")
