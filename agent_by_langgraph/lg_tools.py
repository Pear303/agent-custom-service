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
from agent.rag.retriever import rag_search


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
        "rag_search": rag_search,
    }
    return [tool_map[n] for n in spec_tool_names if n in tool_map]


@tool
def dispatch_subagent_lg(agent_name: str, task: str) -> str:
    """派遣子代理独立处理任务（LangGraph StateGraph 实现）。

    当 LLM 在同一帧发出多个 dispatch_subagent_lg 调用时，
    图路由器（route_after_agent）会将它们转为并行 Send，
    由 subagent_worker 节点在同一 superstep 内并行执行，
    而非逐个串行阻塞。

    单个调用时也走 Send 路径，保持一致性。

    agent_name 可用值: quick_helper, web_researcher, doc_analyzer,
    engine_executor, validator, skill_manager, document_processor,
    system_maintainer（详见 `agent.subagents.registry.SubagentRegistry`）

    Args:
        agent_name: 子代理名称
        task: 委派给子代理的具体任务描述

    Returns:
        此函数体不再执行实际逻辑（由 subagent_worker 节点完成），
        保留此 tool 定义是为了让 LLM 知道可以调用它。
    """
    # 实际执行由 subagent_dispatcher + subagent_worker 节点完成。
    # 此函数体不会被调用——route_after_agent 拦截 dispatch_subagent_lg
    # 的 tool_calls，不走 tools 节点，走 subagent_dispatcher。
    # 但作为 fallback（如直接 invoke 工具时），保留同步执行逻辑。
    import warnings
    warnings.warn(
        "dispatch_subagent_lg 走了 fallback 路径（同步执行），"
        "预期应由 route_after_agent 路由到 subagent_dispatcher 并行执行。"
        "请检查路由逻辑是否正常。",
        stacklevel=2,
    )
    from agent.lc_tools import _ctx_llm_ref, _ctx_sub_reg
    from agent_by_langgraph.lg_subagent import get_subagent_graph

    registry = _ctx_sub_reg.get()
    llm = _ctx_llm_ref.get()
    if registry is None:
        return "Error: Subagent registry not initialized"
    if llm is None:
        return "Error: LLM not initialized"

    spec = registry.get(agent_name)
    if spec is None:
        available = ", ".join(registry.names())
        return f"Error: unknown subagent '{agent_name}'. Available: {available}"

    print(f"\n[LG 派遣子代理(fallback) · {agent_name}]: {task[:80]}")

    try:
        subgraph = get_subagent_graph(llm, registry, agent_name)
        sub_input = {
            "input": task,
            "turns_remaining": spec.max_turns,
            "max_turns": spec.max_turns,
            "messages": [],
        }
        if spec.is_rag:
            sub_input.update({
                "rewritten_queries": [],
                "retrieved_docs": [],
                "rag_context": "",
                "needs_web_fallback": False,
            })
        result = subgraph.invoke(sub_input)  # fallback 路径无 checkpointer，不传 config
    except Exception as exc:
        return f"Error: subagent '{agent_name}' raised: {exc}"

    last_text = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            content = msg.content
            last_text = content if isinstance(content, str) else str(content)
            break
    print(f"[LG 子代理汇报]: {last_text[:200]}")
    return last_text or "[子代理未产出任何回复]"
