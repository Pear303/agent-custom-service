"""子代理规格定义：描述子代理的身份、能力和约束。"""
from __future__ import annotations
from dataclasses import dataclass, field


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
    """
    name: str
    description: str
    system_prompt: str              # 从模板文件加载
    tool_names: tuple[str, ...]     # 工具白名单
    max_turns: int = 15             # 最大迭代轮数