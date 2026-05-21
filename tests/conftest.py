"""pytest 共享 fixtures 与配置"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir():
    """提供临时工作目录，测试后自动清理"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_openai_client():
    """模拟 OpenAI 客户端，用于隔离 API 调用"""
    client = MagicMock()
    return client


@pytest.fixture
def temp_memory_dir(temp_dir):
    """在临时目录中创建 memory 子目录"""
    mem = temp_dir / "memory"
    mem.mkdir()
    # 创建空的 MEMORY.md
    (mem / "MEMORY.md").write_text("# 长期记忆\n\n", encoding="utf-8")
    # 创建空的 history.jsonl
    (mem / "history.jsonl").write_text("", encoding="utf-8")
    return mem
