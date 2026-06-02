"""工具层 —— LangGraph 版工具集合。

工具主体从 `agent.lc_tools` re-export（与原版共享实现，避免重复）；
此处新增：
- `dispatch_subagent_lg`: 派遣子代理走 LangGraph StateGraph 子图
- `make_subagent_tools`: 按 spec 工具白名单构造工具列表
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agent.lc_tools import (
    edit_file,
    glob_tool,
    grep_tool,
    load_skill,
    read_file,
    run_command,
    set_skills_loader,
    set_subagent_deps,
    set_ticket_id,
    set_todo_store,
    set_user_id,
    set_workspace,
    update_todos,
    web_fetch,
    write_file,
    _build_workspace,
)


def make_subagent_tools(spec_tool_names: tuple[str, ...]) -> list:
    """按子代理 spec 的工具白名单构造工具列表。

    工具主体来自 `agent.lc_tools`（与主 agent 共享）。
    **不**包含 `dispatch_subagent`（防递归）和 `update_todos`（防改主 todolist）。

    Args:
        spec_tool_names: SubagentSpec.tool_names —— 工具名称元组

    Returns:
        工具实例列表；spec 中未识别的工具名静默丢弃（spec 应来自可信 registry）
    """
    tool_map = {
        "run_command": run_command,
        "web_fetch": web_fetch,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "glob": glob_tool,
        "grep": grep_tool,
        "load_skill": load_skill,
    }
    return [tool_map[n] for n in spec_tool_names if n in tool_map]


@tool
def dispatch_subagent_lg(agent_type: str, task: str) -> str:
    """派遣子代理独立处理任务（LangGraph StateGraph 实现）。

    与原版 `agent.lc_tools.dispatch_subagent` 的区别：
    - 子代理用独立 StateGraph 子图（`create_subagent_graph`），不依赖 AgentExecutor
    - max_turns 在子图 `post_tools` 节点递减，每轮 agent→tools 循环计 1
    - **不**做连续只读 tool 的并发执行（原 `ParallelAgentExecutor` 的能力）；
      原因见 `lg_subagent` 模块顶部注释

    办完只回传最后一条 AIMessage 的文本内容（与原版行为一致）。

    agent_type 可用值: quick_helper, web_researcher, doc_analyzer,
    engine_executor, validator, skill_manager, document_processor,
    system_maintainer（详见 `agent.subagents.registry.SubagentRegistry`）

    Args:
        agent_type: 子代理类型名
        task: 委派给子代理的具体任务描述

    Returns:
        子代理的最终文本回复（错误时返回以 "Error:" 开头的诊断信息）
    """
    from agent.lc_tools import _ctx_llm_ref, _ctx_subagent_registry
    from agent_by_langgraph.lg_subagent import get_subagent_graph

    registry = _ctx_subagent_registry.get()
    llm = _ctx_llm_ref.get()
    if registry is None:
        return "Error: Subagent registry not initialized"
    if llm is None:
        return "Error: LLM not initialized"

    spec = registry.get(agent_type)
    if spec is None:
        available = ", ".join(registry.names())
        return f"Error: unknown subagent '{agent_type}'. Available: {available}"

    print(f"\n[LG 派遣子代理 · {agent_type}]: {task[:80]}")

    try:
        subgraph = get_subagent_graph(llm, registry, agent_type)
        result = subgraph.invoke({
            "input": task,
            "turns_remaining": spec.max_turns,
            "max_turns": spec.max_turns,
            "messages": [],
        })
    except Exception as exc:
        return f"Error: subagent '{agent_type}' raised: {exc}"

    last_text = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            content = msg.content
            last_text = content if isinstance(content, str) else str(content)
            break
    print(f"[LG 子代理汇报]: {last_text[:200]}")
    return last_text or "[子代理未产出任何回复]"
