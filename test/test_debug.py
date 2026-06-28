"""Agent 调试脚本 — 定位 tool_calls 消息完整性问题。"""
from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agent_by_langgraph.lg_agent import LGAgent


def _validate_messages(messages: list, label: str) -> None:
    """验证消息序列的 tool_call 完整性。"""
    print(f"\n--- 消息验证 [{label}] (共 {len(messages)} 条) ---")
    for i, msg in enumerate(messages):
        role = type(msg).__name__
        content_preview = ""
        if isinstance(msg.content, str):
            content_preview = msg.content[:60].replace("\n", "\\n")
        elif isinstance(msg.content, list):
            content_preview = f"[list, {len(msg.content)} items]"

        if isinstance(msg, AIMessage) and msg.tool_calls:
            tc_ids = [tc.get("id", "?")[:12] for tc in msg.tool_calls]
            tc_names = [tc.get("name", "?") for tc in msg.tool_calls]
            print(f"  [{i}] {role}: tool_calls={tc_names}, ids={tc_ids}")
        elif isinstance(msg, ToolMessage):
            print(f"  [{i}] {role}: tool_call_id={msg.tool_call_id[:12] if msg.tool_call_id else 'None'}, name={msg.name}")
        else:
            print(f"  [{i}] {role}: {content_preview}...")

    # 检查完整性
    pending_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id:
                    pending_tool_call_ids.add(tc_id)
        if isinstance(msg, ToolMessage):
            pending_tool_call_ids.discard(msg.tool_call_id)

    if pending_tool_call_ids:
        print(f"  *** 错误: 有 {len(pending_tool_call_ids)} 个 tool_call_id 缺少对应的 ToolMessage!")
        for tc_id in pending_tool_call_ids:
            print(f"    - 缺失: {tc_id}")
    else:
        print(f"  OK: 所有 tool_calls 都有对应的 ToolMessage")


async def test():
    """手动构建图并逐步执行，观察消息变化。"""
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
    from agent_by_langgraph.lg_graph import create_agent_graph
    from agent_by_langgraph.lg_tools import dispatch_subagent_lg
    from agent_core.skills import get_skills_loader
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

    graph = create_agent_graph(llm, tools, system_prompt)

    user_input = "帮我创建一个简单的Python计算器程序，支持加减乘除"
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    _validate_messages(initial_messages, "初始消息")

    # 手动模拟第一步：planner → agent
    from langchain_core.runnables import RunnableConfig
    config: RunnableConfig = {
        "callbacks": [],
        "recursion_limit": 210,
        "configurable": {
            "thread_id": "debug-test",
            "__has_checkpointer__": False,
        },
    }

    # 使用 astream_events 观察每一步
    print("\n=== 开始执行图 ===\n")
    step_count = 0
    try:
        async for event in graph.astream_events(
            {"messages": initial_messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            name = event.get("name", "")

            if kind == "on_chain_start" and name:
                step_count += 1
                print(f"\n[步骤 {step_count}] 开始: {name}")

            elif kind == "on_chain_end" and name:
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "messages" in output:
                    msgs = output["messages"]
                    if msgs:
                        _validate_messages(msgs if isinstance(msgs, list) else [msgs], f"{name} 输出")

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    print(token, end="", flush=True)

    except Exception as exc:
        print(f"\n\n[ERROR] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(test())
