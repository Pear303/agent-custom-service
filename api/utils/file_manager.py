"""文件操作工具：路径净化 + 报告保存"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# 标准项目子目录名——如果路径的第一级是这些名称，说明已经是项目内部路径，不需要再剥离项目名
_STANDARD_DIRS = frozenset({
    "app", "src", "pages", "components", "static", "public",
    "miniprogram", "cloudfunctions", "backend", "frontend",
    "css", "js", "images", "templates", "utils", "config",
    "tests", "test", "scripts", "docs", "assets", "styles",
    "fonts", "icons", "api", "lib", "types", "hooks", "store",
    "views", "models", "controllers", "routes", "middleware",
    "services", "helpers", "constants", "interfaces", "server",
    "client", "database", "migrations", "seeders",
})


def _clean_output_path(file_path: str, user_id: str, ticket_id: str) -> Path:
    """清理文件路径，反复剥离所有可能的嵌套前缀，防止路径二次拼接。
    
    LLM 返回的文件路径千奇百怪，可能包含：
    - 绝对路径（f:/XiangMu/.../projects/myapp/src/app.js）
    - 带项目目录前缀（projects/myapp/src/app.js）
    - 带 output/_build 前缀（output/myapp/src/app.js）
    - 正常的相对路径（src/app.js）
    
    此函数的目标是将以上所有格式统一归约为纯项目文件路径（如 src/app.js），
    然后再拼接到 output_root（成品/）下。
    """
    p = Path(file_path)

    if p.is_absolute():
        try:
            p = p.relative_to(_PROJECT_ROOT)
        except ValueError:
            # 绝对路径不在项目根下，只保留文件名
            p = Path(p.name)

    # 需要剥离的嵌套前缀（按从长到短排列，优先匹配更具体的前缀）
    nesting_prefixes = [
        Path("data") / "users" / user_id / ticket_id / "成品",
        Path("data") / "users" / user_id / ticket_id / "_build",
        Path("data") / "users" / user_id / ticket_id,
        Path("data") / "users" / user_id,
        Path("data") / "users",
        Path("data"),
        # LLM 常用的项目输出目录名
        Path("projects"),
        Path("output"),
        Path("_build"),
    ]

    # 循环剥离所有匹配的前缀
    changed = True
    while changed and len(p.parts) > 1:
        changed = False
        for prefix in nesting_prefixes:
            try:
                stripped = p.relative_to(prefix)
                if stripped != p:
                    p = stripped
                    changed = True
                    break
            except ValueError:
                continue

    # 剥离项目名目录：LLM 经常创建 "projects/myapp/..." 或 "output/myapp/..."，
    # 剥离 projects/ 后剩下 "myapp/src/app.js"，myapp 是项目名包了一层，也需要剥离。
    # 策略：如果第一级目录名不是标准项目子目录，就认为是项目名目录，逐层剥离。
    while len(p.parts) > 1 and p.parts[0].lower() not in _STANDARD_DIRS and not p.parts[0].startswith("."):
        p = Path(*p.parts[1:])

    clean = str(p).lstrip("/\\").rstrip("/\\")
    return Path(clean) if clean else p


def _save_report(user_id: str, ticket_id: str, filename: str, data: dict) -> Path:
    """将报告保存到 data/users/{user_id}/{ticket_id}/报告/{filename}"""
    report_dir = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "报告"
    report_dir.mkdir(parents=True, exist_ok=True)
    filepath = report_dir / filename
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("报告已保存: %s", filepath)
    return filepath
