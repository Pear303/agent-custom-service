"""微信小程序开发工具：封装微信开发者工具CLI接口。

支持创建项目、打开项目、构建npm、预览、上传等操作。
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool

from .context_vars import _ctx_workspace

_logger = logging.getLogger(__name__)

# 当前选中的小程序项目路径（用于快速切换）
_current_project_path: Optional[str] = None


def _get_cli_path() -> str:
    """获取微信开发者工具CLI路径。
    
    优先从环境变量 WECHAT_DEVTOOLS_PATH 获取，
    否则使用默认路径。
    """
    env_path = os.environ.get("WECHAT_DEVTOOLS_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    if sys.platform == "win32":
        # Windows 默认路径
        default_path = r"C:\Program Files (x86)\Tencent\微信web开发者工具\cli.bat"
        if os.path.exists(default_path):
            return default_path
        # 检查其他常见路径
        alternative_paths = [
            r"C:\Program Files\Tencent\微信web开发者工具\cli.bat",
            r"D:\Program Files (x86)\Tencent\微信web开发者工具\cli.bat",
            r"D:\Program Files\Tencent\微信web开发者工具\cli.bat",
            r"E:\微信web开发工具\微信web开发者工具\cli.bat",
        ]
        for path in alternative_paths:
            if os.path.exists(path):
                return path
    else:
        # macOS/Linux 默认路径
        default_path = "/Applications/微信开发者工具.app/Contents/MacOS/cli"
        if os.path.exists(default_path):
            return default_path
    
    return "cli"  # 让系统 PATH 查找


def _run_cli_command(command: str, project_path: str = None) -> str:
    """执行微信开发者工具CLI命令。
    
    Args:
        command: CLI命令（不含cli前缀）
        project_path: 项目路径
    
    Returns:
        命令输出结果
    """
    cli_path = _get_cli_path()
    
    if project_path:
        # 解析项目路径
        workspace = _ctx_workspace.get()
        if workspace:
            full_path = os.path.join(str(workspace), project_path)
        else:
            full_path = project_path
        
        # 转换为绝对路径
        full_path = os.path.abspath(full_path)
        if not os.path.exists(full_path):
            return f"Error: 项目路径不存在: {full_path}"
        
        command = command.replace("{project_path}", full_path)
    
    full_command = f'"{cli_path}" {command}'
    
    try:
        _logger.info(f"执行微信开发者工具命令: {full_command}")
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        
        output = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        if result.returncode == 0:
            return output
        else:
            return f"Error (code {result.returncode}): {output}"
    
    except subprocess.TimeoutExpired:
        return "Error: 命令执行超时"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def create_miniapp_project(project_name: str, appid: str = "") -> str:
    """创建微信小程序项目。
    
    Args:
        project_name: 项目名称，将创建同名目录
        appid: 小程序AppID（可选，不传则创建测试项目）
    
    Returns:
        创建结果信息
    """
    workspace = _ctx_workspace.get()
    if workspace is None:
        return "Error: 工作区未初始化"
    
    project_path = workspace / project_name
    
    if project_path.exists():
        return f"Error: 项目目录已存在: {project_path}"
    
    try:
        # 创建项目目录结构
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "pages").mkdir(exist_ok=True)
        (project_path / "pages" / "index").mkdir(exist_ok=True)
        (project_path / "pages" / "logs").mkdir(exist_ok=True)
        
        # 获取模板路径
        skill_dir = Path(__file__).parent.parent.parent / "skills" / "miniapp" / "templates"
        
        # 复制模板文件
        template_files = [
            ("app.js", "app.js"),
            ("app.json", "app.json"),
            ("app.wxss", "app.wxss"),
            ("sitemap.json", "sitemap.json"),
            ("project.config.json", "project.config.json"),
            ("pages/index/index.wxml", "pages/index/index.wxml"),
            ("pages/index/index.wxss", "pages/index/index.wxss"),
            ("pages/index/index.js", "pages/index/index.js"),
            ("pages/index/index.json", "pages/index/index.json"),
            ("pages/logs/logs.wxml", "pages/logs/logs.wxml"),
            ("pages/logs/logs.wxss", "pages/logs/logs.wxss"),
            ("pages/logs/logs.js", "pages/logs/logs.js"),
            ("pages/logs/logs.json", "pages/logs/logs.json"),
        ]
        
        for src, dst in template_files:
            src_path = skill_dir / src
            dst_path = project_path / dst
            if src_path.exists():
                content = src_path.read_text(encoding="utf-8")
                # 替换appid
                if appid and "wx0000000000000000" in content:
                    content = content.replace("wx0000000000000000", appid)
                # 替换项目名
                if "demo" in content:
                    content = content.replace("demo", project_name)
                dst_path.write_text(content, encoding="utf-8")
            else:
                _logger.warning(f"模板文件不存在: {src_path}")
        
        return f"✅ 小程序项目创建成功\n路径: {project_path}\n项目结构:\n- app.js (入口文件)\n- app.json (全局配置)\n- app.wxss (全局样式)\n- pages/index/ (首页)\n- pages/logs/ (日志页面)"
    
    except Exception as e:
        return f"Error: 创建项目失败 - {str(e)}"


@tool
def open_miniapp_project(project_path: str = "") -> str:
    """在微信开发者工具中打开项目。
    
    Args:
        project_path: 项目路径（相对于工作区，可选，不传则使用当前项目）
    
    Returns:
        操作结果
    """
    resolved_path = _resolve_project_path(project_path)
    if not resolved_path:
        return "❌ 未提供项目路径且未设置当前项目，请先使用 set_current_project 设置项目"
    return _run_cli_command(f"open --project {{project_path}}", resolved_path)


@tool
def build_miniapp_npm(project_path: str = "") -> str:
    """构建npm依赖。
    
    Args:
        project_path: 项目路径（相对于工作区，可选，不传则使用当前项目）
    
    Returns:
        操作结果
    """
    resolved_path = _resolve_project_path(project_path)
    if not resolved_path:
        return "❌ 未提供项目路径且未设置当前项目，请先使用 set_current_project 设置项目"
    return _run_cli_command(f"build-npm --project {{project_path}}", resolved_path)


@tool
def preview_miniapp(project_path: str = "", page_path: str = "") -> str:
    """预览小程序（生成二维码）。
    
    Args:
        project_path: 项目路径（相对于工作区，可选，不传则使用当前项目）
        page_path: 预览页面路径（可选，如 "pages/index/index"）
    
    Returns:
        操作结果（包含二维码信息）
    """
    resolved_path = _resolve_project_path(project_path)
    if not resolved_path:
        return "❌ 未提供项目路径且未设置当前项目，请先使用 set_current_project 设置项目"
    
    if page_path:
        compile_cond = f'--compile-cond {{"path":"{page_path}"}}'
    else:
        compile_cond = ""
    
    return _run_cli_command(f"preview --project {{project_path}} {compile_cond}", resolved_path)


@tool
def upload_miniapp(project_path: str = "", version: str = "1.0.0", desc: str = "") -> str:
    """上传小程序代码到微信公众平台。
    
    Args:
        project_path: 项目路径（相对于工作区，可选，不传则使用当前项目）
        version: 版本号（格式如 1.0.0）
        desc: 版本描述（可选）
    
    Returns:
        操作结果
    """
    resolved_path = _resolve_project_path(project_path)
    if not resolved_path:
        return "❌ 未提供项目路径且未设置当前项目，请先使用 set_current_project 设置项目"
    
    desc_param = f'-d "{desc}"' if desc else ""
    return _run_cli_command(f"upload --project {{project_path}} -u {version} {desc_param}", resolved_path)


@tool
def check_miniapp_login() -> str:
    """检查微信开发者工具登录状态。
    
    Returns:
        登录状态信息
    """
    result = _run_cli_command("islogin")
    if "已登录" in result or "logged" in result.lower():
        return f"✅ {result}"
    else:
        return f"⚠️ {result}\n提示：请先在微信开发者工具中登录，或使用 cli login 命令登录"


@tool
def login_miniapp() -> str:
    """登录微信开发者工具。
    
    Returns:
        操作结果（可能需要扫码登录）
    """
    return _run_cli_command("login")


@tool
def close_miniapp_project(project_path: str = "") -> str:
    """关闭项目窗口或工具。
    
    Args:
        project_path: 项目路径（可选，不传则关闭整个工具）
    
    Returns:
        操作结果
    """
    if project_path:
        return _run_cli_command(f"close --project {{project_path}}", project_path)
    else:
        return _run_cli_command("quit")


@tool
def set_current_project(project_path: str) -> str:
    """设置当前工作的小程序项目路径（快速切换目录）。
    
    设置后，后续的 open_miniapp_project、build_miniapp_npm、preview_miniapp、
    upload_miniapp 等操作可以省略 project_path 参数，自动使用当前项目。
    
    Args:
        project_path: 项目路径（相对于工作区或绝对路径）
    
    Returns:
        操作结果
    """
    global _current_project_path
    
    workspace = _ctx_workspace.get()
    if workspace:
        full_path = os.path.join(str(workspace), project_path)
    else:
        full_path = project_path
    
    full_path = os.path.abspath(full_path)
    
    if not os.path.exists(full_path):
        return f"❌ 项目路径不存在: {full_path}"
    
    # 验证是否是小程序项目
    required_files = ["app.json", "project.config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(full_path, f))]
    if missing_files:
        return f"⚠️ 警告：路径存在但缺少小程序必需文件: {', '.join(missing_files)}\n仍将设置为当前项目"
    
    _current_project_path = full_path
    return f"✅ 当前项目已切换到:\n{full_path}"


@tool
def get_current_project() -> str:
    """获取当前设置的小程序项目路径。
    
    Returns:
        当前项目路径或提示信息
    """
    if _current_project_path:
        return f"📌 当前项目路径:\n{_current_project_path}"
    else:
        return "⚠️ 未设置当前项目，请先使用 set_current_project 设置项目路径"


@tool
def list_miniapp_projects(dir_path: str = "") -> str:
    """列出指定目录下的小程序项目。
    
    自动识别包含 app.json 和 project.config.json 的目录作为小程序项目。
    
    Args:
        dir_path: 目录路径（可选，不传则列出工作区下的项目）
    
    Returns:
        项目列表
    """
    if dir_path:
        workspace = _ctx_workspace.get()
        if workspace:
            full_path = os.path.join(str(workspace), dir_path)
        else:
            full_path = dir_path
        full_path = os.path.abspath(full_path)
    else:
        workspace = _ctx_workspace.get()
        if workspace:
            full_path = str(workspace)
        else:
            return "❌ 工作区未初始化，请提供目录路径"
    
    if not os.path.exists(full_path):
        return f"❌ 目录不存在: {full_path}"
    
    projects: List[str] = []
    try:
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                # 检查是否是小程序项目
                if os.path.exists(os.path.join(item_path, "app.json")):
                    projects.append(item)
        
        if projects:
            result = f"📁 找到 {len(projects)} 个小程序项目:\n"
            result += "─" * 50 + "\n"
            for i, project in enumerate(projects, 1):
                # 检查是否是当前项目
                marker = "⭐" if _current_project_path and project in _current_project_path else " "
                result += f"{i}. {marker} {project}\n"
            result += "─" * 50 + "\n"
            result += "提示：使用 set_current_project <项目名> 快速切换"
            return result
        else:
            return f"⚠️ 目录中未找到小程序项目（需包含 app.json）\n目录: {full_path}"
    
    except Exception as e:
        return f"Error: {str(e)}"


def _resolve_project_path(project_path: Optional[str]) -> Optional[str]:
    """解析项目路径，如果未提供则使用当前项目路径。
    
    Args:
        project_path: 项目路径或None
    
    Returns:
        解析后的项目路径
    """
    if project_path:
        return project_path
    elif _current_project_path:
        # 返回相对路径（相对于工作区）
        workspace = _ctx_workspace.get()
        if workspace and _current_project_path.startswith(str(workspace)):
            return os.path.relpath(_current_project_path, str(workspace))
        return _current_project_path
    return None