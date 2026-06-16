"""子代理规格定义：描述子代理的身份、能力和约束。

从 agent.subagents.spec 迁移，无 import 变更。
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SUBAGENT_DANGEROUS_TOOLS = frozenset({
    "write_file", "edit_file", "run_command",
})


@dataclass(frozen=True)
class SubagentSpec:
    """子代理身份的完整定义。"""

    name: str
    description: str
    system_prompt: str
    tool_names: tuple[str, ...]
    max_turns: int = 15
    is_rag: bool = False
    read_only: bool = False

    def __post_init__(self):
        import os
        dangerous = set(self.tool_names) & _SUBAGENT_DANGEROUS_TOOLS
        if dangerous:
            strict = os.environ.get("SUBAGENT_STRICT_TOOL_CHECK", "false").lower() in ("true", "1", "yes")
            if strict:
                raise ValueError(
                    f"[D8] 子代理 '{self.name}' 包含危险工具: {dangerous}。"
                    f"子代理子图无 interrupt 审批门，这些工具将直接执行。"
                    f"设置 SUBAGENT_STRICT_TOOL_CHECK=false 可降级为警告。"
                )
            else:
                logger.debug(
                    "[D8] 子代理 '%s' 包含危险工具: %s。"
                    "子代理子图无 interrupt 审批门，这些工具将直接执行。"
                    "设置 SUBAGENT_STRICT_TOOL_CHECK=true 可改为抛出异常。",
                    self.name, dangerous,
                )
        if not self.read_only and not dangerous:
            object.__setattr__(self, 'read_only', True)
