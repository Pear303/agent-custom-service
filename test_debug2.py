"""Agent 调试脚本 — 精确定位 view 消息完整性问题。"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _check_view_integrity(view: list, label: str) -> bool:
    """检查 view 的 tool_call 完整性，返回是否通过。"""
    # 收集所有 tool_call_ids 和 tool_message_ids
    tool_call_ids = set()
    tool_msg_ids = set()
    for msg in view:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    tool_call_ids.add(tc_id)
        if isinstance(msg, ToolMessage):
            tool_msg_ids.add(msg.tool_call_id)

    missing = tool_call_ids - tool_msg_ids
    orphan = tool_msg_ids - tool_call_ids

    ok = True
    if missing:
        print(f"  [{label}] 错误: {len(missing)} 个 tool_call 缺少 ToolMessage")
        for mid in missing:
            print(f"    - 缺失 ToolMessage for: {mid}")
        ok = False
    if orphan:
        print(f"  [{label}] 错误: {len(orphan)} 个 ToolMessage 无对应 AIMessage(tool_calls)")
        for oid in orphan:
            print(f"    - 孤立 ToolMessage: {oid}")
        ok = False
    if ok:
        print(f"  [{label}] OK: 消息完整性检查通过 ({len(view)} 条消息)")
    return ok


async def test():
    """通过 monkey-patch call_agent 来调试 view 构建。"""
    from agent_by_langgraph.lg_graph import create_agent_graph
    from agent_core.llm import create_deepseek_llm
    from agent_core.tools import (
        _build_workspace, read_file, write_file, edit_file,
        run_command, web_fetch, load_skill, glob_tool, grep_tool,
        update_todos, set_workspace, set_skills_loader, set_subagent_deps,
        set_user_id, set_todo_store,
    )
    from agent_core.memory import MemoryStore
    from agent_core.subagents.registry import SubagentRegistry
    from agent_core.telemetry import TokenTracker
    from agent_core.todo import TodoStore
    from agent_core.context import ContextBuilder
    from agent_by_langgraph.lg_tools import dispatch_subagent_lg
    from agent_core.skills import get_skills_loader
    from agent_core.context_view import ContextView
    from agent_core.in_context_compactor import InContextCompactor
    from agent_core.observation_masker import ObservationMasker
    from pathlib import Path
    from dotenv import load_dotenv
    import os

    load_dotenv()

    root = Path(__file__).parent
    model = "deepseek-v4-flash"
    llm = create_deepseek_llm(model)

    workspace = _build_workspace(root, None, None)
    set_workspace(workspace)

    skills = get_skills_loader(root / "skills")
    set_skills_loader(skills)

    todo_store = TodoStore(user_id=None)
    set_todo_store(todo_store)

    tools = [
        read_file, write_file, edit_file,
        run_command, web_fetch, load_skill,
        glob_tool, grep_tool, update_todos,
        dispatch_subagent_lg,
    ]

    sub_reg = SubagentRegistry(
        root / "templates" / "subagents",
        skills_loader=skills,
    )
    set_subagent_deps(llm=llm, registry=sub_reg)

    memory_store = MemoryStore(
        memory_dir=root / "memory",
        user_file=root / "templates" / "USER.md",
    )

    ctx = ContextBuilder(root / "templates", skills, memory=memory_store)
    system_prompt = ctx.build_system_prompt()

    # Monkey-patch: 在 call_agent 中注入调试
    _context_view = ContextView()
    _in_context_compactor = InContextCompactor()
    _observation_masker = ObservationMasker()

    original_build_view = _context_view.build_view
    call_count = [0]

    def debug_build_view(messages):
        call_count[0] += 1
        print(f"\n=== ContextView.build_view 第 {call_count[0]} 次调用 ===")
        print(f"  输入消息数: {len(messages)}")
        _check_view_integrity(messages, f"build_view 输入 #{call_count[0]}")

        result = original_build_view(messages)
        view, pruned = result
        print(f"  输出消息数: {len(view)} (裁剪了 {len(pruned)} 组)")
        _check_view_integrity(view, f"build_view 输出 #{call_count[0]}")
        return result

    _context_view.build_view = debug_build_view

    original_mask = _observation_masker.mask
    mask_count = [0]

    def debug_mask(messages):
        mask_count[0] += 1
        result = original_mask(messages)
        if len(result) != len(messages):
            print(f"\n=== ObservationMasker.mask 第 {mask_count[0]} 次调用 ===")
            print(f"  输入: {len(messages)} 条 → 输出: {len(result)} 条")
            _check_view_integrity(result, f"mask 输出 #{mask_count[0]}")
        return result

    _observation_masker.mask = debug_mask

    original_compact = _in_context_compactor.compact
    compact_count = [0]

    def debug_compact(messages):
        compact_count[0] += 1
        result = original_compact(messages)
        if len(result) != len(messages):
            print(f"\n=== InContextCompactor.compact 第 {compact_count[0]} 次调用 ===")
            print(f"  输入: {len(messages)} 条 → 输出: {len(result)} 条")
            _check_view_integrity(result, f"compact 输出 #{compact_count[0]}")
        return result

    _in_context_compactor.compact = debug_compact

    # 替换 lg_graph 模块中的实例
    import agent_by_langgraph.lg_graph as lg_graph_mod
    lg_graph_mod._context_view = _context_view  # 这不会影响已编译的闭包

    # 需要直接 patch create_agent_graph 中的闭包
    # 更好的方式：直接在图执行中拦截

    graph = create_agent_graph(llm, tools, system_prompt)

    user_input = "帮我创建一个简单的Python计算器程序，支持加减乘除"
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    config = {
        "callbacks": [],
        "recursion_limit": 210,
        "configurable": {
            "thread_id": "debug-test-2",
            "__has_checkpointer__": False,
        },
    }

    print("=== 开始执行图 ===\n")
    try:
        async for event in graph.astream_events(
            {"messages": initial_messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    print(token, end="", flush=True)
    except Exception as exc:
        print(f"\n\n[ERROR] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(test())
