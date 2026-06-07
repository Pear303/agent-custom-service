"""子代理规格定义：描述子代理的身份、能力和约束。"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# D8: 子代理不应包含的危险工具（应有 interrupt 审批门但子图没有）
_SUBAGENT_DANGEROUS_TOOLS = frozenset({
    "write_file", "edit_file", "run_command",
})


@dataclass(frozen=True)
class SubagentSpec:
    """子代理身份的完整定义。

    每个子代理都有独立的身份、工具白名单和最大迭代次数。

    安全约束：
    - tool_names 绝不应包含 'dispatch_subagent'（防止递归派遣）
    - tool_names 绝不应包含 'update_todos'（todolist 是主 agent 的状态，子代理无权修改）

    Attributes:
        name: 子代理的唯一标识名称
        description: 子代理的功能描述
        system_prompt: 子代理的系统提示词（定义其身份和行为准则）
        tool_names: 允许使用的工具名称元组（白名单）
        max_turns: 子代理单次任务的最大迭代轮数（默认 15）
        is_rag: 是否使用 CRAG 子图（带查询改写+文档评估的 RAG 流程）
        read_only: 是否为只读子代理（不含写操作工具）。gather 阶段只允许只读子代理。
    """
    name: str
    description: str              # 从模板文件加载
    system_prompt: str              # 从模板文件加载
    tool_names: tuple[str, ...]     # 工具白名单
    max_turns: int = 15             # 最大迭代轮数
    is_rag: bool = False            # 是否使用 CRAG 子图
    read_only: bool = False         # 是否为只读子代理

    def __post_init__(self):
        """D8: 验证工具白名单，对危险工具发出警告或抛出异常。自动推断 read_only。"""
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
                logger.warning(
                    "[D8] 子代理 '%s' 包含危险工具: %s。"
                    "子代理子图无 interrupt 审批门，这些工具将直接执行。"
                    "设置 SUBAGENT_STRICT_TOOL_CHECK=true 可改为抛出异常。",
                    self.name, dangerous,
                )
        # 自动推断 read_only：若未显式设置且不含危险工具，则标记为只读
        if not self.read_only and not dangerous:
            object.__setattr__(self, 'read_only', True)
