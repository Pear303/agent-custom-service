"""LangGraph Agent 入口 —— python agent_lg.py 启动"""
from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    try:
        # Windows 终端编码策略：
        # - stdin 必须用 UTF-8（确保接收到的中文输入正确解码）
        # - stdout/stderr 使用终端实际编码（通常 GBK），errors="replace"
        #   确保中文正常显示，遇到无法编码的字符（如 emoji）用 ? 替代而非崩溃
        # - chcp 65001 在某些终端（如 Windows Terminal）下能让 UTF-8 输出正常显示
        # - PYTHONUTF8=1 确保文件读写默认用 UTF-8
        os.system('chcp 65001 >nul 2>&1')
        os.environ.setdefault("PYTHONUTF8", "1")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        # stdout/stderr：优先 UTF-8（chcp 65001 生效时），失败则回退到默认编码
        _stdout_enc = sys.stdout.encoding or "utf-8"
        _stderr_enc = sys.stderr.encoding or "utf-8"
        # 如果 chcp 65001 生效，终端代码页是 65001，可以安全用 UTF-8 输出
        # 如果 chcp 65001 未生效（如某些旧版 CMD），终端仍用 GBK，
        # 此时 UTF-8 输出会乱码，应回退到终端默认编码
        import ctypes
        _console_cp = 0
        try:
            _console_cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        except Exception:
            pass
        if _console_cp == 65001:
            # 终端输出代码页已设为 UTF-8，可以安全输出 UTF-8
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        else:
            # 终端仍用默认代码页（如 GBK/936），用终端编码输出
            sys.stdout.reconfigure(encoding=_stdout_enc, errors="replace")
            sys.stderr.reconfigure(encoding=_stderr_enc, errors="replace")
    except (AttributeError, OSError):
        pass

from agent_by_langgraph.lg_agent import LGAgent

if __name__ == "__main__":
    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    agent.run()
