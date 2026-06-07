"""_resolve 路径解析单元测试 — 覆盖 LLM 常见路径格式。"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.lc_tools import _resolve, _ctx_workspace, _ctx_user_id, _ctx_ticket_id


@pytest.fixture
def workspace():
    """创建临时 workspace 并设置上下文变量。"""
    tmp = Path(tempfile.gettempdir()) / "test_resolve_ws"
    tmp.mkdir(exist_ok=True)
    ws = tmp / "workspace"
    ws.mkdir(exist_ok=True)
    token = _ctx_workspace.set(ws)
    uid_token = _ctx_user_id.set("test_user")
    tid_token = _ctx_ticket_id.set("test_ticket")
    yield ws
    _ctx_workspace.reset(token)
    _ctx_user_id.reset(uid_token)
    _ctx_ticket_id.reset(tid_token)


class TestResolveRelativePath:
    """相对路径解析。"""

    def test_simple_relative(self, workspace):
        result = _resolve("src/app.js")
        assert result == workspace / "src" / "app.js"

    def test_single_file(self, workspace):
        result = _resolve("index.html")
        assert result == workspace / "index.html"

    def test_nested_path(self, workspace):
        result = _resolve("css/style/main.css")
        assert result == workspace / "css" / "style" / "main.css"

    def test_dot_path(self, workspace):
        result = _resolve("./README.md")
        assert result == workspace / "README.md"


class TestResolveAbsolutePath:
    """绝对路径解析 — 应剥离项目根前缀。"""

    def test_absolute_within_project(self, workspace):
        project_root = Path(__file__).parent
        abs_path = str(project_root / "agent" / "main.py")
        result = _resolve(abs_path)
        assert result == workspace / "agent" / "main.py"

    def test_absolute_outside_project(self, workspace):
        # 绝对路径不在项目根下，应回退为只用文件名
        result = _resolve("C:/Windows/System32/drivers/etc/hosts")
        assert result.name == "hosts"
        assert str(workspace.resolve()).lower() in str(result.resolve()).lower()


class TestResolveDataPrefix:
    """带 data/users 前缀的路径剥离。"""

    def test_full_data_prefix(self, workspace):
        result = _resolve("data/users/test_user/test_ticket/成品/src/app.js")
        assert result == workspace / "src" / "app.js"

    def test_partial_data_prefix(self, workspace):
        result = _resolve("data/users/test_user/test_ticket/src/app.js")
        assert result == workspace / "src" / "app.js"

    def test_data_users_prefix(self, workspace):
        result = _resolve("data/users/test_user/src/app.js")
        assert result == workspace / "src" / "app.js"

    def test_data_prefix_only(self, workspace):
        result = _resolve("data/src/app.js")
        assert result == workspace / "src" / "app.js"


class TestResolvePathEscape:
    """路径逃逸防护。"""

    def test_parent_traversal(self, workspace):
        result = _resolve("../../etc/passwd")
        # 应回退到 workspace 内（Windows 路径大小写不敏感，用 resolve 后比较）
        assert str(workspace.resolve()).lower() in str(result.resolve()).lower()

    def test_complex_traversal(self, workspace):
        result = _resolve("src/../../../etc/shadow")
        assert str(workspace.resolve()).lower() in str(result.resolve()).lower()


class TestResolveEdgeCases:
    """边界情况。"""

    def test_empty_path(self, workspace):
        result = _resolve("")
        assert str(workspace.resolve()) in str(result)

    def test_path_with_spaces(self, workspace):
        result = _resolve("my project/main.py")
        assert result == workspace / "my project" / "main.py"

    def test_chinese_path(self, workspace):
        result = _resolve("源代码/主程序.py")
        assert result == workspace / "源代码" / "主程序.py"

    def test_no_workspace_raises(self):
        # 保存当前值，强制设为 None
        old = _ctx_workspace.get()
        token = _ctx_workspace.set(None)
        try:
            with pytest.raises(RuntimeError, match="工作区未初始化"):
                _resolve("test.py")
        finally:
            _ctx_workspace.reset(token)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:asyncio"])
