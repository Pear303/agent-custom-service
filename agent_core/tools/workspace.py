"""Workspace 路径构建与解析。

从 agent.lc_tools 提取 _build_workspace, _resolve, get_workspace_path。
"""
from __future__ import annotations

import logging
from pathlib import Path

from .context_vars import (
    _ctx_workspace,
    _ctx_user_id,
    _ctx_ticket_id,
)

_logger = logging.getLogger(__name__)


def _build_workspace(root: Path, user_id: str | None, ticket_id: str | None) -> Path:
    """构建完整的工作目录路径：{root}/data/users/{user_id}/{ticket_id}"""
    if user_id and ticket_id:
        ws = root / "data" / "users" / user_id / ticket_id
    elif user_id:
        ws = root / "data" / "users" / user_id
    else:
        ws = root / "data" / "users" / "_anonymous"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def get_workspace_path(root: Path | None = None, user_id: str | None = None,
                       ticket_id: str | None = None) -> Path:
    """获取 Agent 的工作目录路径（不创建目录）。

    供外部调用方（测试脚本等）获取与 Agent 相同的工作目录，
    确保 CWD 一致性。目录不存在时自动创建。

    Args:
        root: 项目根目录，默认为 agent_core 目录的父目录
        user_id: 用户 ID
        ticket_id: 工单 ID

    Returns:
        工作目录的 Path 对象
    """
    if root is None:
        root = Path(__file__).parent.parent.parent
    return _build_workspace(root, user_id, ticket_id)


def _resolve(path: str) -> Path:
    """解析路径，所有路径均约束在 workspace 内，防止文件逃逸到根目录。

    LLM 可能传入各种格式的路径：
    - 相对路径：src/app.js → workspace/src/app.js
    - 绝对路径：f:/XiangMu/.../projects/myapp/app.js → 先剥离项目根，再拼到 workspace
    - 带项目目录的路径：projects/myapp/app.js → 拼到 workspace/projects/myapp/app.js
    - 带完整 data/users 前缀的路径：data/users/uid/tid/成品/src/app.js → 剥离前缀后 src/app.js

    关键约束：最终路径必须严格在 workspace 目录内。
    """
    p = Path(path).expanduser()
    workspace = _ctx_workspace.get()

    if workspace is not None:
        if p.is_absolute():
            _project_root = Path(__file__).parent.parent.parent
            try:
                p = p.relative_to(_project_root)
            except ValueError:
                p = Path(p.name)

        # 剥离 workspace 自身路径前缀，防止路径嵌套
        if not p.is_absolute():
            # D18: 宽容匹配 — LLM 可能使用带错误后缀的路径
            if p.parts and len(p.parts) >= 2 and p.parts[0] == "data" and p.parts[1] == "users":
                _uid = _ctx_user_id.get() or ""
                _tid = _ctx_ticket_id.get() or ""
                _expected_prefix_parts = ("data", "users", _uid, _tid)
                _matched = 0
                for i, ep in enumerate(_expected_prefix_parts):
                    if i < len(p.parts):
                        if i < 3:  # data, users, uid 必须精确匹配
                            if p.parts[i] == ep:
                                _matched += 1
                        elif i == 3:  # tid 层：只要前缀匹配即可（容忍 _bugfix 等后缀）
                            if p.parts[i].startswith(ep):
                                _matched += 1
                if _matched == 4 and len(p.parts) > 4:
                    stripped = Path(*p.parts[4:])
                    _logger.info(
                        "D18 宽容路径剥离: '%s' → '%s' (LLM 使用了错误的 tid 后缀: %s → %s)",
                        p, stripped, _tid, p.parts[3],
                    )
                    p = stripped

            _ws_prefixes = [
                Path("data") / "users" / _ctx_user_id.get() / _ctx_ticket_id.get() / "成品",
                Path("data") / "users" / _ctx_user_id.get() / _ctx_ticket_id.get() / "_build",
                Path("data") / "users" / _ctx_user_id.get() / _ctx_ticket_id.get(),
                Path("data") / "users" / _ctx_user_id.get(),
                Path("data") / "users",
                Path("data"),
            ]
            for prefix in _ws_prefixes:
                if not prefix.parts or prefix == Path("."):
                    continue
                try:
                    stripped = p.relative_to(prefix)
                    if stripped != p:
                        _logger.info(
                            "路径前缀剥离: '%s' → '%s' (剥离前缀 '%s')",
                            p, stripped, prefix,
                        )
                        p = stripped
                        break
                except ValueError:
                    continue

        resolved = (workspace / p).resolve()

        # 安全检查：确保解析后的路径仍在 workspace 内
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            _logger.warning(
                "路径逃逸检测: %s 解析到 workspace 外 (%s)，回退为文件名 %s",
                path, resolved, p.name,
            )
            resolved = workspace / p.name

        return resolved

    _logger.critical(
        "_resolve: _ctx_workspace is None — 拒绝文件写入！"
        " path=%s, thread=%s, user_id=%s, ticket_id=%s",
        str(p), __import__("threading").get_ident(),
        _ctx_user_id.get(), _ctx_ticket_id.get(),
    )
    raise RuntimeError(
        f"工作区未初始化，拒绝写入文件 '{path}'。"
        f" (user_id={_ctx_user_id.get()}, ticket_id={_ctx_ticket_id.get()})"
    )
