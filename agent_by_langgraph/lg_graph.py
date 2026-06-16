"""LangGraph StateGraph 定义 — 主 Agent 的图结构。

用 StateGraph 替代 LangChain 的 AgentExecutor，显式控制 Agent 循环。

消息拼接策略：
  初始调用时，调用方将 chat_history + input 作为 messages 的一部分传入，
  call_agent 只需在 messages 前插入 SystemMessage，不再重复拼接
  chat_history 和 input，避免多轮循环时 token 浪费。

并行子代理派遣（Send API）：
  当 LLM 在同一帧发出多个 dispatch_subagent_lg 调用时，
  route_after_agent 将它们路由到 subagent_dispatcher 节点，
  由该节点通过 Send 原语生成多个并行 subagent_worker 实例。
  所有 worker 在同一 superstep 内并行执行，结果通过
  Annotated[list[str], add] reducer 安全合并到 subagent_results，
  最后由 aggregate_results 节点汇总为一条 AIMessage 注入主对话流。

安全审批门（interrupt）：
  危险工具（write_file、edit_file、run_command）调用前，
  route_after_agent 将其路由到 interrupt_approval 节点，
  该节点调用 interrupt() 暂停执行，等待人工确认后继续。

兼容性：
  LangGraph >= 1.2 的 compile() 不再接受 callbacks 参数，
  图级回调改为在 invoke/astream 时通过 config.callbacks 传入。
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Annotated, Sequence, TypedDict

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("agent.audit")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Send, interrupt

from agent_by_langgraph.lg_parallel_tools import ParallelToolNode


# 需要人工审批的危险工具
_DANGEROUS_TOOLS = frozenset({
    "write_file", "edit_file", "run_command",
})

# ── 自适应工具分组（P3）────────────────────────────────────────
# gather 阶段：只读工具，信息收集（核心限制：防止未收集信息就修改）
# D5: gather 阶段不包含 run_command，防止 Agent 用 shell 命令（pwd/dir/ls）
# 探测路径而非使用 glob_tool/grep_tool 等专用只读工具。
# 如果 gather 阶段需要运行脚本收集信息，应通过 dispatch_subagent_lg 派遣子代理。
# 注：dispatch_subagent_lg 在 gather 阶段可用，但 LLM 应优先派遣只读子代理
#     （SubagentSpec.read_only=True）。
_GATHER_TOOLS = frozenset({
    "read_file", "grep_tool", "glob_tool", "web_fetch",
    "load_skill", "dispatch_subagent_lg", "update_todos",
})
# modify 阶段：全量工具（修改时需要只读工具辅助参考，无限制必要）
_MODIFY_TOOLS = frozenset({
    "write_file", "edit_file", "run_command",
    "read_file", "grep_tool", "glob_tool", "web_fetch",
    "load_skill", "dispatch_subagent_lg", "update_todos",
})
# verify 阶段：全量工具（验证时可能需要修复，无限制必要）
_VERIFY_TOOLS = frozenset({
    "read_file", "grep_tool", "glob_tool", "web_fetch",
    "write_file", "edit_file", "run_command",
    "load_skill", "dispatch_subagent_lg", "update_todos",
})
# 注：modify/verify 包含全量工具是设计意图，非遗漏。
# P3 的核心价值是 gather 阶段阻止过早修改，后续阶段无需额外限制。

# D14: 使用 UUID 哨兵值，避免与子代理输出内容冲突
import uuid
_SUBAGENT_CLEAR_SENTINEL = f"__SUBAGENT_CLEAR_{uuid.uuid4().hex}__"


def _replace_results(existing: list[str], new: list[str]) -> list[str]:
    """子代理结果 reducer：支持消费后清空。

    约定：
    - new 非空 → 追加到 existing
    - new == [_SUBAGENT_CLEAR_SENTINEL] → 清空（消费信号，由 aggregate_results 发出）
    - new 为空列表 → 保持不变（LangGraph Send 合并时的默认行为）

    替代 operator.add：add reducer 无法清空列表，
    导致 aggregate_results 返回 [] 后 existing 仍保留旧值。

    D14: 使用 UUID 哨兵值，避免与子代理输出内容冲突。
    """
    if new == [_SUBAGENT_CLEAR_SENTINEL]:
        return []  # 清空信号：aggregate_results 消费后发出
    return existing + new


class AgentState(TypedDict):
    """主 Agent 状态。

    Attributes:
        messages: 消息序列，通过 add_messages reducer 自动累积
        subagent_results: 并行子代理结果，通过 _replace_results reducer 合并/清空。
            多个 subagent_worker 在同一 superstep 并行写入，
            aggregate_results 消费后返回 [] 触发清空。
        _approval_next: 审批路由指示，interrupt_approval 设置，
            _route_after_approval 消费。值为 "tools" 或 "agent"
        plan: 当前执行计划，由 planner 节点生成。
            每轮用户请求开始时生成，agent 循环中保持不变。
            新用户请求到来时由 planner 重新生成。
        _phase: 当前执行阶段（P3 自适应工具选择）。
            "gather" = 信息收集（只读工具）
            "modify" = 代码修改（写工具）
            "verify" = 验证总结（只读 + 写工具）
            "all" = 全量工具（兜底）
        _stall_count: 连续停滞次数（P3 兜底机制）。
            agent 在过滤模式下返回空 tool_calls 时 +1，
            连续 2 次停滞回退到 "all" 阶段。
        _pending_tool_calls: 混合调用时暂存的非子代理 tool_calls。
            当 LLM 同时发出子代理调用和普通/危险工具调用时，
            _advance_phase 将非子代理 tool_calls 暂存到此字段，
            _aggregate_results 在汇总后读取并注入消息流，然后清空。
            存储在 state 中而非模块级变量，确保多用户并发安全。
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    subagent_results: Annotated[list[str], _replace_results]
    _approval_next: str
    _pending_route: str
    plan: str
    _phase: str
    _stall_count: int
    _pending_tool_calls: list[dict]
    _compaction_summary: str
    _reflection_result: str
    _reflection_count: int
    _deviation_count: int
    _deviation_reason: str
    _interrupt_repeat_count: int
    _interrupt_last_tool_sig: str


class SubagentWorkerState(TypedDict):
    """单个子代理 worker 的独立状态。

    每个 Send("subagent_worker", ...) 创建一个独立的 worker 实例，
    拥有自己的 state，不与主 AgentState 共享。
    """
    agent_name: str
    task: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    _parent_phase: str  # P3: 主图当前阶段，gather 时强制子代理只读


def _route_after_agent(state: AgentState) -> str | list[Send]:
    """agent 节点后的路由：区分普通工具调用、子代理派遣、危险工具审批、反思、直接结束。

    路由优先级：
    1. 无 tool_calls → 检查是否需要反思
       - verify 阶段 + 有计划 + 反思次数未超限 → reflect
       - 否则 → END
    2. 有 dispatch_subagent_lg 调用 → 直接返回 list[Send] 并行派遣
       - 混合调用时，非子代理的 tool_calls 暂存到 _pending_tool_calls，
         由 aggregate_results 恢复到消息流
    3. 有危险工具调用 → interrupt_approval（安全审批）
    4. 普通工具调用 → tools

    注意：LangGraph 条件边函数支持返回 list[Send]，
    但独立节点不允许。因此子代理派遣逻辑直接在此条件边中完成，
    不再通过独立的 _subagent_dispatcher 节点中转。
    """

    print(f"[路由] 最后消息类型: {type(state['messages'][-1]).__name__}, "
          f"有 tool_calls? {bool(state['messages'][-1].tool_calls)}")

    last = state["messages"][-1]

    if not isinstance(last, AIMessage) or not last.tool_calls:
        # agent 想结束对话 → 检查是否需要反思
        phase = state.get("_phase", "gather") or "gather"
        reflection_count = state.get("_reflection_count", 0) or 0
        plan = state.get("plan", "")

        # 反思触发条件：
        # 1. 有实质计划（非"无需规划"）
        # 2. 计划步骤 > 2（简单任务不需要反思）
        # 3. 反思次数未超限
        _is_simple_plan = (
            not plan
            or plan == "无需规划"
            or len(re.findall(r'^\d+\.\s', plan, re.MULTILINE)) <= 2
        )
        if not _is_simple_plan:
            if phase == "verify" and reflection_count < 2:
                return "reflect"
            if phase == "modify" and reflection_count < 1:
                return "reflect"

        return END

    # 分类 tool_calls
    subagent_calls = [tc for tc in last.tool_calls if tc["name"] == "dispatch_subagent_lg"]
    dangerous_calls = [tc for tc in last.tool_calls if tc["name"] in _DANGEROUS_TOOLS]
    normal_calls = [tc for tc in last.tool_calls if tc["name"] not in _DANGEROUS_TOOLS and tc["name"] != "dispatch_subagent_lg"]

    # T2 优化：只有 update_todos 调用时，走轻量内联节点，跳过 tools 节点的完整循环
    todos_only_calls = [tc for tc in last.tool_calls if tc["name"] == "update_todos"]
    other_normal_calls = [tc for tc in normal_calls if tc["name"] != "update_todos"]
    if (todos_only_calls
        and not other_normal_calls
        and not dangerous_calls
        and not subagent_calls):
        return "todos_inline"

    # 子代理调用优先：直接返回 list[Send] 并行派遣
    if subagent_calls:
        # 混合调用：非子代理的 tool_calls 由 _advance_phase 暂存到 state._pending_tool_calls，
        # _aggregate_results 从 state 读取并恢复到消息流
        if normal_calls:
            logger.info(
                "[D4] 混合调用: %d 个子代理 + %d 个普通工具, "
                "非子代理调用将由 _advance_phase 暂存",
                len(subagent_calls), len(normal_calls),
            )

        # P3 gather 阶段语义约束：将当前阶段传递给 subagent_worker，
        # 由 worker 在执行时强制过滤子代理工具为只读（如果处于 gather 阶段）。
        # 不在此处拒绝派遣，因为条件边函数无法注入拒绝消息到 state，
        # 拒绝后回到 agent 会导致无限循环。
        parent_phase = state.get("_phase", "gather") or "gather"

        sends = []
        for tc in subagent_calls:
            sends.append(Send(
                "subagent_worker",
                {
                    "agent_name": tc["args"]["agent_name"],
                    "task": tc["args"]["task"],
                    "messages": [],
                    "_parent_phase": parent_phase,
                },
            ))
        return sends

    # 危险工具：走 interrupt_approval 安全审批
    if dangerous_calls and not normal_calls:
        return "interrupt_approval"

    # 混合场景（危险 + 普通）：走 interrupt_approval，
    # 审批通过后由 tools 节点统一执行
    if dangerous_calls and normal_calls:
        return "interrupt_approval"

    # 普通工具调用
    return "tools"


async def _todos_inline(state: AgentState, config: RunnableConfig) -> dict:
    """轻量内联执行 update_todos，跳过 tools 节点的完整循环。

    T2 优化：update_todos 是纯本地操作（更新 TodoStore），
    无需经过 ParallelToolNode 的完整工具执行流程。
    直接调用工具函数，返回 ToolMessage，然后回到 agent 继续推理。

    这避免了：tools 节点 → agent 节点 的完整往返（含消息历史重传），
    将 update_todos 的执行合并到当前步中。
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"messages": []}

    from agent_core.tools import update_todos as _update_todos_fn
    results = []
    for tc in last.tool_calls:
        if tc["name"] != "update_todos":
            continue
        try:
            observation = await _update_todos_fn.ainvoke(tc["args"])
            results.append(ToolMessage(
                content=str(observation),
                tool_call_id=tc["id"],
                name="update_todos",
            ))
        except Exception as exc:
            results.append(ToolMessage(
                content=f"Error: {exc}",
                tool_call_id=tc["id"],
                name="update_todos",
                status="error",
            ))

    return {"messages": results}


async def _subagent_worker(state: SubagentWorkerState, config: RunnableConfig) -> dict:
    """执行单个子代理任务，就地压缩后返回结果写入 subagent_results。

    通过 ContextVar 获取 LLM 和子代理注册表，
    调用 get_subagent_graph 创建/获取缓存的子图并执行。

    ContextVar 隔离：在 worker 入口处 snapshot ContextVar，
    执行完毕后 restore，确保并行 worker 之间互不干扰。
    虽然 asyncio Task 会自动 copy_context，但防御性快照/恢复
    可防止子代理内部修改 ContextVar 影响其他 worker。

    压缩策略（P2）：
    - 在 worker 内就地压缩，减少 subagent_results 内存占用
    - 优先提取 ## 结论 段落，否则取最后 3 行
    - 输出结构化格式：类型、摘要、文件、结论
    """
    from agent_core.tools import _ctx_llm_ref, _ctx_sub_reg, _ctx_user_id
    from agent_by_langgraph.lg_subagent import get_subagent_graph
    from agent_by_langgraph.context_var_manager import snapshot as _ctx_snapshot, restore as _ctx_restore

    # 防御性 ContextVar 快照：确保并行 worker 之间互不干扰
    ctx_snap = _ctx_snapshot()

    try:
        registry = _ctx_sub_reg.get()
        llm = _ctx_llm_ref.get()
        agent_name = state["agent_name"]
        task = state["task"]

        if registry is None:
            return {"subagent_results": [f"[{agent_name}] Error: Subagent registry not initialized"]}
        if llm is None:
            return {"subagent_results": [f"[{agent_name}] Error: LLM not initialized"]}

        spec = registry.get(agent_name)
        if spec is None:
            available = ", ".join(registry.names())
            return {"subagent_results": [f"[{agent_name}] Error: unknown subagent. Available: {available}"]}

        print(f"\n[LG 并行子代理 · {agent_name}]: {task[:80]}")

        # P3 gather 阶段语义约束：非只读子代理在 gather 阶段不执行
        # 返回提示消息，让主 agent 知道需要等待 modify 阶段
        parent_phase = state.get("_parent_phase", "all") or "all"
        if parent_phase == "gather" and not spec.read_only:
            logger.warning(
                "[P3] gather 阶段拒绝非只读子代理 %s（含写工具: %s），"
                "请等待进入 modify 阶段后再派遣",
                agent_name,
                [t for t in spec.tool_names if t in _DANGEROUS_TOOLS],
            )
            return {
                "subagent_results": [
                    f"[{agent_name}] P3 约束：当前处于 gather（信息收集）阶段，"
                    f"不允许派遣含写操作工具的子代理。"
                    f"请先完成信息收集，系统会自动进入 modify 阶段后再执行修改操作。"
                    f"你可以派遣只读子代理来收集信息。"
                ]
            }

        # 构造子代理的 checkpointer 和隔离 thread_id
        import time
        user_id = _ctx_user_id.get() or "default"
        sub_thread_id = f"{user_id}:sub:{agent_name}:{int(time.time() * 1000)}"

        # 子代理使用内存 checkpointer（无需持久化，避免 SQLite 锁冲突）
        sub_checkpointer = None
        try:
            from langgraph.checkpoint.memory import MemorySaver
            sub_checkpointer = MemorySaver()
        except ImportError:
            logger.warning("[SubCheckpointer] MemorySaver 不可用，降级为无状态执行")

        try:
            subgraph = get_subagent_graph(llm, registry, agent_name, checkpointer=sub_checkpointer)
            sub_config = {"configurable": {"thread_id": sub_thread_id}}
            # CRAG 子图需要额外的状态字段
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
            result = await subgraph.ainvoke(sub_input, config=sub_config)
        except Exception as exc:
            return {"subagent_results": [f"[{agent_name}] Error: {exc}"]}

        # 提取最后一条 AIMessage 的文本
        last_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                content = msg.content
                last_text = content if isinstance(content, str) else str(content)
                break

        if not last_text:
            return {"subagent_results": [f"[{agent_name}] [子代理未产出任何回复]"]}

        # 就地压缩：提取结论 + 结构化格式
        compressed = _compress_subagent_result(last_text, agent_name=agent_name)
        print(f"[LG 子代理汇报 · {agent_name}]: {compressed[:200]}")
        return {"subagent_results": [compressed]}
    finally:
        # 恢复 ContextVar，防止子代理内部修改影响其他并行 worker
        _ctx_restore(ctx_snap)


def _aggregate_results(state: AgentState) -> dict:
    """将并行子代理的结果合并为一条 AIMessage，注入主对话流。

    消费 subagent_results 后清空，避免下一轮循环重复消费。
    子代理结果已在 _subagent_worker 中就地压缩（结构化格式），
    此处只需拼接，不再二次压缩。

    混合调用恢复：当 LLM 同时发出子代理调用和普通/危险工具调用时，
    _advance_phase 已将非子代理 tool_calls 暂存到 state._pending_tool_calls。
    此函数从 state 精确读取，将其作为新 AIMessage 注入消息流，
    并通过 _pending_route 指示 _route_after_aggregate 直接路由到 tools 节点执行。
    """
    results = state.get("subagent_results", [])
    if not results:
        return {"messages": [], "subagent_results": [], "_pending_route": "agent"}

    # worker 已压缩，直接拼接
    combined = "\n\n".join(results)
    summary_msg = AIMessage(content=f"[子代理汇报]\n{combined}")

    new_messages = [summary_msg]

    # 从 state._pending_tool_calls 精确读取暂存的非子代理 tool_calls
    pending_calls = list(state.get("_pending_tool_calls", []))

    pending_route = "agent"
    if pending_calls:
        # 检查 pending_calls 中是否包含危险工具
        has_dangerous_pending = any(
            tc.get("name") in _DANGEROUS_TOOLS for tc in pending_calls
        )

        # 注入新 AIMessage 包含未执行的 tool_calls
        # 添加 id 字段确保消息唯一性，避免 LangGraph 消息去重异常
        pending_msg = AIMessage(
            content="",
            tool_calls=pending_calls,
            id=f"pending-{uuid.uuid4().hex[:8]}",
        )
        new_messages.append(pending_msg)

        if has_dangerous_pending:
            # 危险工具必须走 interrupt_approval，不能绕过审批门
            pending_route = "interrupt_approval"
            logger.info(
                "[D4] 混合调用恢复: %d 个 pending tool_calls 含危险工具，路由到 interrupt_approval",
                len(pending_calls),
            )
        else:
            # 普通工具直接路由到 tools 节点执行
            pending_route = "tools"
            logger.info(
                "[D4] 混合调用恢复: %d 个 pending tool_calls 将路由到 tools 节点",
                len(pending_calls),
            )
    elif state.get("_pending_tool_calls"):
        # D4 防御性检查：_pending_tool_calls 存在但为空列表，
        # 不应路由到 tools（可能是上一轮残留的空列表）
        logger.warning("[D4] _pending_tool_calls 为空列表，路由到 agent 而非 tools")

    # 清空 subagent_results（通过 UUID 哨兵值触发 _replace_results 清空）
    # 同时清空 _pending_tool_calls（已消费）
    result = {
        "messages": new_messages,
        "subagent_results": [_SUBAGENT_CLEAR_SENTINEL],
        "_pending_route": pending_route,
        "_pending_tool_calls": [],  # 始终清空，避免空列表残留导致下一轮误判
    }
    return result


def _route_after_aggregate(state: AgentState) -> str:
    """aggregate_results 后路由：根据 _pending_route 决定下一节点。

    路由选项：
    - "tools": 仅有普通 pending tool_calls，直接执行
    - "interrupt_approval": pending tool_calls 含危险工具，需审批
    - "agent": 无 pending tool_calls，回到 agent 继续对话
    """
    return state.get("_pending_route", "agent")


# ── 子代理结果压缩 ──────────────────────────────────────────────

_MAX_SUBAGENT_CHARS = 2000
_MAX_SUMMARY_CHARS = 80
_MAX_CONCLUSION_CHARS = 1200


def _compress_subagent_result(
    result: str,
    max_chars: int = _MAX_SUBAGENT_CHARS,
    agent_name: str = "",
) -> str:
    """压缩子代理结果为结构化格式。

    输出格式：
        [子代理名称] 摘要(≤50字)
        - 涉及文件: file1.py:10, file2.ts:42
        - 结论: ...

    策略：
    1. 如果结果 ≤ max_chars，直接加前缀返回
    2. 提取结构化结论（## 结论 标题下的内容）
    3. 退化为取最后 3 行作为结论
    4. 附加文件路径（含行号）
    """
    prefix = f"[{agent_name}] " if agent_name else ""

    # 短结果直接返回
    if len(result) <= max_chars:
        return f"{prefix}{result}"

    # 提取文件路径
    paths = _extract_file_paths(result)
    path_line = ", ".join(paths[:5]) if paths else ""

    # 提取结论
    conclusion = _extract_conclusion(result)

    # 提取摘要（第一行非空文本，或前50字）
    first_line = ""
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    summary = first_line[:_MAX_SUMMARY_CHARS]

    # 组装结构化输出
    parts = [f"{prefix}{summary}"]
    if path_line:
        parts.append(f"- 涉及文件: {path_line}")
    parts.append(f"- 结论: {conclusion[:_MAX_CONCLUSION_CHARS]}")

    structured = "\n".join(parts)
    return structured[:max_chars]


def _extract_conclusion(text: str) -> str:
    """从子代理输出中提取结论。

    优先级：
    1. ## 结论/总结/结果/执行结果/最终答案 标题下的内容
    2. 最后 3 行非空文本
    """
    # 策略1: 提取结构化结论（支持冒号、空格等变体）
    conclusion_match = re.search(
        r'(?:^|\n)##\s*(?:结论|总结|结果|执行结果|最终答案|Conclusion|Summary|Result|Final Answer)'
        r'[:：]?\s*\n(.*?)(?:\n##|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if conclusion_match:
        return conclusion_match.group(1).strip()

    # 策略2: 取最后 3 行非空文本
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[-3:])


def _extract_file_paths(text: str) -> list[str]:
    """从文本中提取文件路径（含行号）。支持 Unix 和 Windows 路径。

    使用文件扩展名白名单避免误匹配（如 "凝气期:1"、"12:30"）。
    """
    # 常见代码/配置文件扩展名白名单
    _EXT_WHITELIST = (
        r'py|pyi|js|jsx|ts|tsx|mjs|cjs|mts|cts|'
        r'json|yaml|yml|toml|ini|cfg|conf|'
        r'html|htm|css|scss|sass|less|'
        r'md|mdx|txt|rst|'
        r'go|rs|java|kt|scala|rb|php|sh|bash|zsh|'
        r'sql|graphql|proto|'
        r'dockerfile|makefile|'
        r'xml|svg|'
        r'env|gitignore|editorconfig|'
        r'lock|log'
    )
    return re.findall(
        rf'(?:[A-Za-z]:)?[\w/.\\-]+\.(?:{_EXT_WHITELIST}):\d+',
        text,
        re.IGNORECASE,
    )


# ── Reflection ─────────────────────────────────────────────────

_REFLECTION_PROMPT = """\
你是一个质量审查员。请评估当前工作是否完整、正确地解决了用户的需求。

评估维度：
1. 完整性：用户需求的所有要点是否都已处理？
2. 正确性：代码/文件修改是否正确？有无语法错误或逻辑问题？
3. 一致性：修改是否与已有代码风格一致？是否引入冲突？

当前执行计划：
{plan}

当前执行阶段：{phase}

请输出评估结果：
- 如果工作已完成且质量合格，输出：[反思: 通过] 然后给出最终总结
- 如果发现问题需要修复，输出：[反思: 回退] 然后说明需要修复什么

注意：不要无理由回退。只有发现明确的遗漏或错误时才回退。
"""

# ── Replanner ──────────────────────────────────────────────────

_REPLANNER_PROMPT = """\
你是一个任务规划修正器。根据执行进度和遇到的问题，修正原有执行计划。

原计划：
{original_plan}

当前遇到的问题/偏差：
{deviation_reason}

最近执行摘要：
{recent_summary}

请基于以上信息，输出修正后的执行计划。规则：
1. 保留仍有效的步骤
2. 修正有问题的步骤
3. 如需新增步骤，追加在末尾
4. 步骤不超过 7 步
5. 最后一行标注当前应进入的阶段：[阶段: gather/modify/verify]

输出格式：
1. [步骤描述] → 执行者: 自己, 工具: xxx
2. ...
"""

# ── Plan-then-Execute ──────────────────────────────────────────

_PLANNER_PROMPT = """\
你是一个任务规划器。根据用户的请求，制定简洁的执行计划。

规则：
1. 步骤不超过 7 步
2. 每步必须指定"执行者"——由你自己做，还是派遣子代理
3. 优先信息收集（read_file, grep_tool, glob_tool），再执行修改
4. 最后一步必须是验证或总结
5. 如果请求很简单（问候、简单问答），输出"无需规划"

阶段选择规则（重要！）：
- 开发/创建/构建类任务（涉及写文件）→ 必须从 gather 开始：
  先用 glob_tool 查看目录结构、read_file 查看已有文件，再动手写代码
- 修复/编辑类任务（已有代码需修改）→ 可从 modify 开始
- 纯查询/验证类任务 → 从 verify 开始

执行者选择规则（重要！）：
判断每一步由谁执行。不要凭用户措辞判断，要凭任务本身的复杂度判断。

派遣决策树（严格按此判断）：

1. 步骤总数 ≤ 2？
   → 自己执行，不派遣

2. 步骤之间有依赖（B 需要 A 的输出才能正确执行）？
   → 有依赖的步骤自己执行，不派遣

3. 存在可独立并行的子任务？
   → 子任务只需只读操作（查资料、校验）？
     → 派遣只读子代理（web_researcher / validator / quick_helper）
   → 子任务需要写操作（创建/修改文件）？
     → 写操作涉及 ≥ 3 个互不依赖的文件？
       → 派遣 engine_executor 处理部分文件
     → 写操作仅涉及 1-2 个文件，或文件间有依赖？
       → 自己执行，不派遣

关键判断原则：
- 派遣子代理的收益是并行节省时间，成本是额外 token 消耗和上下文隔离
- 只读子代理成本低，写操作子代理成本高，需要更慎重
- 子代理看不到主 Agent 的工作上下文，所以有依赖关系的步骤不能拆分
- 不要因为用户说了"同时"或"并行"就派遣，要看任务本身是否真的可独立
- 同一项目的文件（如主程序+测试+文档）通常有依赖关系，默认视为不可并行
- 只有明确互不依赖的文件（如3个独立工具脚本）才适合派遣写操作子代理

子代理能力：
- engine_executor: 可读写执行，适合创建/修改互不依赖的代码文件（成本高，≥3个互不依赖文件才派遣）
- web_researcher: 只读+网络，适合查资料/搜索文档（成本低，有需要就派遣）
- validator: 只读校验，适合验证结果/运行测试（成本低，有需要就派遣）
- quick_helper: 轻量只读，适合简单确认（成本最低）

输出格式：
1. [步骤描述] → 执行者: 自己, 工具: xxx
2. [步骤描述] → 执行者: 派遣 engine_executor, 任务: xxx
3. [步骤描述] → 执行者: 自己, 工具: xxx
...

注意：当某步的执行者是"派遣"时，不再指定具体工具（子代理自己决定用什么工具），
而是给出明确的任务描述。

最后一行必须标注初始执行阶段（三选一）：
- [阶段: gather] — 需要先收集信息（搜索、阅读文件、查看目录等）
- [阶段: modify] — 已有足够信息，直接修改代码
- [阶段: verify] — 只需验证或回答问题

用户请求：
{request}
"""


def _should_plan(state: AgentState) -> str:
    """判断是否需要规划：新请求总是重新规划，已有计划则直接执行。

    路由逻辑：
    - 最新消息是 milestone HumanMessage（新用户请求）→ 总是走 planner
      （planner 会根据任务复杂度决定是否生成计划，同时清除旧状态）
    - plan 非空且非新请求 → "agent"（循环中，按已有计划执行）
    - plan 为空 → "planner"（首轮或 plan 被清空）
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, HumanMessage):
        # milestone 标记的 HumanMessage 是新用户请求
        # 总是走 planner：1) 清除旧状态 2) 生成新计划（简单任务返回"无需规划"）
        if getattr(last_msg, 'metadata', None) and last_msg.metadata.get("milestone"):
            return "planner"
        # 无 milestone 但是首轮（plan 为空）→ 也需要规划
        if not state.get("plan"):
            return "planner"
        # 循环中的 HumanMessage（如 interrupt 恢复后的用户消息）→ 按已有计划执行
        return "agent"

    if state.get("plan"):
        return "agent"
    return "planner"


async def _plan_node(state: AgentState, config: RunnableConfig) -> dict:
    """Plan-then-Execute 的规划节点。

    在 agent 行动前生成执行计划，注入 state.plan。
    使用无工具的 LLM 做纯推理规划，避免工具调用干扰。

    设计要点：
    - 新用户请求（milestone）到来时强制重新规划，清除旧计划和相关状态
    - agent 循环中 plan 保持不变，agent 按计划执行
    - 从计划中提取初始执行阶段（P3 自适应工具选择）
    """
    from agent_core.tools import _ctx_llm_ref

    llm = _ctx_llm_ref.get()
    if llm is None:
        return {"plan": "", "_phase": "gather", "_stall_count": 0,
                "_deviation_count": 0, "_deviation_reason": "",
                "_interrupt_repeat_count": 0, "_interrupt_last_tool_sig": "",
                "_reflection_count": 0, "_compaction_summary": ""}

    # 提取用户最近的请求
    user_request = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            if isinstance(user_request, list):
                user_request = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in user_request
                )
            break

    if not user_request.strip():
        return {"plan": "", "_phase": "gather", "_stall_count": 0,
                "_deviation_count": 0, "_deviation_reason": "",
                "_interrupt_repeat_count": 0, "_interrupt_last_tool_sig": "",
                "_reflection_count": 0, "_compaction_summary": ""}

    # 简单任务快速路径：短消息且无开发/复杂关键词时跳过 LLM 规划
    _SIMPLE_TASK_MAX_LEN = 80
    _COMPLEX_KEYWORDS = ("系统", "架构", "集成", "多个", "数据库", "API", "微服务",
                          "部署", "测试", "重构", "迁移", "平台", "流程")
    _DEV_COMPOUND = re.compile(
        r'(写|创建|开发|构建|实现|编写|生成|制作|修改|编辑|修复|添加|删除)'
        r'.*(脚本|程序|文件|应用|网页|工具|服务|项目|模块|组件|接口|函数|类)'
    )
    is_simple = (
        len(user_request) <= _SIMPLE_TASK_MAX_LEN
        and not any(kw in user_request for kw in _COMPLEX_KEYWORDS)
        and not bool(_DEV_COMPOUND.search(user_request))
    )
    if is_simple:
        logger.info("[规划跳过] 简单任务，无需规划: %s", user_request[:50])
        return {"plan": "无需规划", "_phase": "all", "_stall_count": 0,
                "_deviation_count": 0, "_deviation_reason": "",
                "_interrupt_repeat_count": 0, "_interrupt_last_tool_sig": "",
                "_reflection_count": 0, "_compaction_summary": ""}

    planning_input = _PLANNER_PROMPT.format(request=user_request)
    try:
        # 显式绑定空工具列表，确保 planner 不产生 tool_calls
        planner_llm = llm.bind_tools([])
        response = await planner_llm.ainvoke(
            [HumanMessage(content=planning_input)],
            config=config,
        )
        plan_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        # 规划失败不应阻断流程，返回空计划让 agent 自由行动
        print(f"[Planner] 规划失败: {exc}")
        plan_text = ""

    # 从计划中提取初始阶段
    phase = _extract_phase(plan_text)

    # 新规划时清除所有旧状态，防止旧上下文污染新任务
    return {"plan": plan_text, "_phase": phase, "_stall_count": 0,
            "_deviation_count": 0, "_deviation_reason": "",
            "_interrupt_repeat_count": 0, "_interrupt_last_tool_sig": "",
            "_reflection_count": 0, "_compaction_summary": ""}


def _extract_phase(plan_text: str) -> str:
    """从规划输出中提取执行阶段。

    查找 [阶段: xxx] 标记，未找到时根据内容推断：
    - 简单创建任务（步骤 ≤ 2 且涉及写文件）→ "modify"（直接动手，无需先收集信息）
    - 复杂创建任务（步骤 ≥ 3）→ "gather"（先收集信息再动手）
    - 纯修改/编辑类（已有代码需修改）→ "modify"
    - 搜索/查找/阅读类 → "gather"
    - 默认 → "gather"
    """
    step_count = len(re.findall(r'^\d+\.\s', plan_text, re.MULTILINE))
    has_create_keyword = bool(re.search(
        r'(创建|写|编写|生成|新建).*(文件|脚本|程序|模块|\.\w{1,4}\b)',
        plan_text, re.IGNORECASE
    ))

    match = re.search(r'\[阶段[:：]\s*(gather|modify|verify)\]', plan_text, re.IGNORECASE)
    if match:
        marker_phase = match.group(1).lower()
        # 修正：即使规划器标记为 gather，如果步骤 ≤ 2 且涉及创建/写文件，
        # 应直接推断为 modify，避免 gather 阶段无写工具导致停滞
        if marker_phase == "gather" and step_count <= 2 and has_create_keyword:
            logger.info("[阶段修正] 简单创建任务 gather → modify: 步骤=%d", step_count)
            return "modify"
        return marker_phase

    # 简单创建任务（步骤 ≤ 2 且涉及创建/写文件）→ modify
    if step_count <= 2 and has_create_keyword:
        return "modify"

    # 推断：开发/创建/构建类 → gather（先收集信息再动手）
    if re.search(r'(开发|创建|构建|搭建|新建|实现|设计|编写).*(应用|项目|程序|系统|工具|脚本|网页|网站|服务)', plan_text):
        return "gather"
    # 推断：步骤数 >= 3 → gather（复杂任务先收集信息）
    if step_count >= 3:
        return "gather"
    # 推断：纯修改/编辑（不涉及新建）→ modify
    if re.search(r'(修改|编辑|重写|删除|添加代码|修复)', plan_text):
        return "modify"
    # 推断：搜索关键词 → gather
    if re.search(r'(搜索|查找|阅读|分析|检查)', plan_text):
        return "gather"

    return "gather"


def _bind_tools_for_phase(llm, all_tools, llm_with_all_tools, phase: str, stall_count: int):
    """根据执行阶段动态绑定工具子集（P3 自适应工具选择）。

    Args:
        llm: 原始 LLM 实例（用于 bind_tools）
        all_tools: 全量工具列表
        llm_with_all_tools: 预绑定的全量工具 LLM（性能优化，避免重复 bind）
        phase: 当前执行阶段
        stall_count: 连续停滞次数

    Returns:
        绑定了对应工具子集的 LLM
    """
    # 防御 None（首轮 input_state 可能未初始化新字段）
    phase = phase or "all"
    stall_count = stall_count or 0

    # 兜底：连续 2 次停滞 → 回退到全量工具
    if stall_count >= 2:
        return llm_with_all_tools

    # 阶段 → 允许的工具集
    phase_to_tools = {
        "gather": _GATHER_TOOLS,
        "modify": _MODIFY_TOOLS,
        "verify": _VERIFY_TOOLS,
    }

    allowed = phase_to_tools.get(phase)
    if allowed is None:
        # "all" 或未知阶段 → 全量工具
        return llm_with_all_tools

    # 过滤工具
    filtered = [t for t in all_tools if t.name in allowed]
    if not filtered:
        return llm_with_all_tools

    return llm.bind_tools(filtered)


def _detect_deviation(
    state: AgentState,
    new_stall: int,
    deviation_count: int,
    deviation_reason: str,
) -> tuple[int, str]:
    """偏差检测：分析执行状态，返回 (new_deviation_count, new_deviation_reason)。

    检测信号：
    1. 不可恢复的工具错误 → deviation_count +1
    2. 计划步骤停滞（stall_count >= 2）→ deviation_count +1
    3. 连续 3 轮相同工具调用 → deviation_count +1
    4. 连续 3 轮完全相同的工具调用（死循环）→ deviation_count +2
    5. 无偏差信号 → deviation_count 衰减 -1

    Args:
        state: 当前 Agent 状态
        new_stall: 本轮更新后的停滞计数
        deviation_count: 当前偏差计数
        deviation_reason: 当前偏差原因

    Returns:
        (new_deviation_count, new_deviation_reason)
    """
    _RECOVERABLE_ERROR_PATTERNS = (
        "参数为空", "path 参数", "必须提供", "未找到", "not found",
        "no such file", "does not exist", "missing", "required",
    )
    has_tool_error = False
    is_recoverable_error = False
    tool_error_detail = ""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.status == "error":
                has_tool_error = True
                tool_error_detail = str(msg.content)[:200] if msg.content else ""
                error_lower = tool_error_detail.lower()
                is_recoverable_error = any(
                    pattern.lower() in error_lower for pattern in _RECOVERABLE_ERROR_PATTERNS
                )
            break

    if has_tool_error and not is_recoverable_error:
        deviation_count += 1
        deviation_reason = "工具执行出错"
        logger.info("[偏差检测] 工具错误(不可恢复), deviation_count=%d, detail=%s",
                    deviation_count, tool_error_detail[:80])
    elif has_tool_error and is_recoverable_error:
        logger.info("[偏差检测] 工具错误(可恢复，不计入偏差), detail=%s", tool_error_detail[:80])
    elif new_stall >= 2:
        deviation_count += 1
        deviation_reason = f"计划步骤停滞 (stall_count={new_stall})"
        logger.info("[偏差检测] 计划停滞, deviation_count=%d", deviation_count)
    else:
        # 信号3：连续调用同一工具且参数高度相似
        recent_ai_msgs = [
            m for m in messages[-8:]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        if len(recent_ai_msgs) >= 3:
            recent_tool_sets = [
                frozenset(tc["name"] for tc in m.tool_calls)
                for m in recent_ai_msgs[-3:]
            ]
            if len(recent_tool_sets) == 3 and recent_tool_sets[0] == recent_tool_sets[1] == recent_tool_sets[2]:
                def _tool_call_sig(msg: AIMessage) -> str:
                    parts = []
                    for tc in msg.tool_calls:
                        args_str = str(tc.get("args", {}))[:100]
                        parts.append(f"{tc['name']}:{args_str}")
                    return "|".join(sorted(parts))

                sigs = [_tool_call_sig(m) for m in recent_ai_msgs[-3:]]
                if sigs[0] == sigs[1] == sigs[2]:
                    deviation_count += 2
                    deviation_reason = f"连续 3 轮完全相同的工具调用（死循环）: {recent_tool_sets[0]}"
                    logger.warning(
                        "[偏差检测] 死循环, deviation_count=%d, sig=%s",
                        deviation_count, sigs[0][:80],
                    )
                else:
                    deviation_count += 1
                    deviation_reason = f"连续 3 轮调用相同工具: {recent_tool_sets[0]}"
                    logger.info(
                        "[偏差检测] 连续同工具, deviation_count=%d, tools=%s",
                        deviation_count, recent_tool_sets[0],
                    )
            else:
                deviation_count = max(0, deviation_count - 1)
                if deviation_count == 0:
                    deviation_reason = ""
        else:
            deviation_count = max(0, deviation_count - 1)
            if deviation_count == 0:
                deviation_reason = ""

    return deviation_count, deviation_reason


def _advance_phase(state: AgentState) -> dict:
    """自动推进执行阶段 + 停滞检测 + 偏差检测（P3）。

    阶段推进规则：
    - gather → modify：agent 调用了写工具（write_file, edit_file）
    - modify → verify：agent 返回文本（无 tool_calls），表示修改完成
    - verify → verify：保持不变

    停滞检测：
    - agent 在过滤模式下返回空 tool_calls → stall_count +1
    - 连续 2 次停滞 → 回退到 "all" 阶段

    偏差检测：
    - 连续工具错误（最近 ToolMessage status=error）→ deviation_count +1
    - 计划步骤停滞（stall_count >= 3）→ deviation_count +1
    - deviation_count >= 2 → 触发 replanner 重新规划

    注意：
    - 此节点在 agent 之后、route_after_agent 之前执行，
      不修改 messages，只更新 _phase、_stall_count、_deviation_count 等。
    - 阶段推进是"预判"而非"确认"：如果危险工具被审批拒绝，
      下一轮 agent 会重新调用工具，_advance_phase 会再次评估。
      这不会造成问题，因为：
      a) 拒绝后 agent 回到 agent 节点，再次经过 _advance_phase
      b) 如果 agent 不再调用写工具，phase 不会继续推进
      c) 如果 agent 仍调用写工具，phase 保持 modify，行为一致
    """
    phase = state.get("_phase", "gather") or "gather"
    stall_count = state.get("_stall_count", 0) or 0
    deviation_count = state.get("_deviation_count", 0) or 0
    deviation_reason = state.get("_deviation_reason", "") or ""

    last = state["messages"][-1] if state["messages"] else None
    if not isinstance(last, AIMessage):
        return {"_phase": phase, "_stall_count": stall_count,
                "_deviation_count": deviation_count, "_deviation_reason": deviation_reason}

    # 检测 agent 是否调用了写工具
    # D5: run_command 在 gather 阶段允许用于信息收集（如运行测试），
    # 不应触发 gather→modify 推进。只有 write_file/edit_file 才是真正的修改。
    # D21: 但如果 run_command 中包含 python -c + open('w') 的写操作，
    # 也应触发阶段推进（Agent 试图绕过专用工具写文件）。
    _WRITE_TOOLS = frozenset({"write_file", "edit_file"})
    has_modify_calls = bool(last.tool_calls) and any(
        tc["name"] in _WRITE_TOOLS for tc in last.tool_calls
    )
    # 轻量检测 run_command 中的写操作
    if not has_modify_calls:
        import re as _phase_re
        for tc in last.tool_calls:
            if tc["name"] == "run_command":
                cmd = str(tc.get("args", {}).get("command", ""))
                if cmd.lower().startswith(("python -c", "python3 -c", "py -c")):
                    if _phase_re.search(r'open\s*\([^)]*[\'"]w[\'"]', cmd):
                        has_modify_calls = True
                        logger.info("[D21] run_command 含写操作，触发阶段推进: %s", cmd[:80])
                        break

    # 检测 agent 是否调用了只读工具（非写工具，含 run_command）
    has_read_only_calls = bool(last.tool_calls) and not has_modify_calls

    # 检测 agent 是否返回文本（无 tool_calls）
    has_text_response = not last.tool_calls and bool(last.content)

    new_phase = phase
    new_stall = stall_count
    new_deviation_count = deviation_count
    new_deviation_reason = deviation_reason

    if phase == "gather":
        if has_modify_calls:
            # gather → modify：agent 开始修改
            new_phase = "modify"
            new_stall = 0
        elif has_read_only_calls:
            # gather 阶段正常调用只读工具，重置停滞计数
            new_stall = 0
        elif has_text_response:
            # gather 阶段直接返回文本，可能是简单回答
            new_stall = stall_count + 1
            if new_stall >= 2:
                new_phase = "all"
                new_stall = 0

    elif phase == "modify":
        if has_text_response:
            # modify → verify：修改完成，进入验证
            new_phase = "verify"
            new_stall = 0
        elif has_modify_calls:
            # 仍在修改
            new_stall = 0

    elif phase == "verify":
        if has_text_response:
            # verify 阶段返回文本，可能已完成
            new_stall = stall_count + 1
            if new_stall >= 2:
                new_phase = "all"
                new_stall = 0
        elif has_read_only_calls or has_modify_calls:
            # verify 阶段仍在调用工具，重置停滞
            new_stall = 0

    # "all" 阶段不做推进

    # ── 偏差检测 ──────────────────────────────────────────────
    new_deviation_count, new_deviation_reason = _detect_deviation(state, new_stall, deviation_count, deviation_reason)

    # 混合调用暂存：当 LLM 同时发出子代理调用和普通/危险工具调用时，
    # 将非子代理 tool_calls 暂存到 state._pending_tool_calls，
    # 由 _aggregate_results 读取并恢复到消息流。
    # 存储在 state 中而非模块级变量，确保多用户并发安全。
    pending_calls: list[dict] = []
    if isinstance(last, AIMessage) and last.tool_calls:
        subagent_calls = [tc for tc in last.tool_calls if tc["name"] == "dispatch_subagent_lg"]
        if subagent_calls:
            # 混合调用：暂存非子代理的 tool_calls
            other_calls = [
                tc for tc in last.tool_calls if tc["name"] != "dispatch_subagent_lg"
            ]
            pending_calls = other_calls
            if other_calls:
                logger.info(
                    "[D4] 混合调用暂存: %d 个子代理 + %d 个普通/危险工具, "
                    "非子代理调用已暂存到 _pending_tool_calls",
                    len(subagent_calls), len(other_calls),
                )

    result = {
        "_phase": new_phase,
        "_stall_count": new_stall,
        "_deviation_count": new_deviation_count,
        "_deviation_reason": new_deviation_reason,
    }
    if pending_calls:
        result["_pending_tool_calls"] = pending_calls
    elif state.get("_pending_tool_calls"):
        # 无新 pending_calls 但 state 中有残留值，清空避免误判
        result["_pending_tool_calls"] = []
    return result


def _interrupt_approval(state: AgentState, config: RunnableConfig) -> dict:
    """危险操作审批门：暂停执行，等待人工确认。

    当 LLM 发出危险工具调用（write_file、edit_file、run_command）时，
    interrupt() 暂停图执行，返回审批请求给调用方。
    调用方通过 Command(resume=...) 恢复执行：
    - "approve": 放行，交给 tools 节点执行
    - "reject": 拒绝，注入拒绝消息

    返回值包含 _approval_next 字段，指示下一个节点：
    - "tools": 审批通过，执行工具
    - "agent": 审批拒绝，回到 agent

    降级处理：interrupt() 需要 checkpointer 才能工作（状态需持久化才能恢复），
    无 checkpointer 时自动放行并记录警告，避免运行时崩溃。
    """
    last = state["messages"][-1]
    dangerous_calls = [
        tc for tc in last.tool_calls if tc["name"] in _DANGEROUS_TOOLS
    ]

    # 检查 checkpointer 是否可用
    # interrupt() 需要 checkpointer 才能暂停和恢复执行。
    # 不能用 thread_id 存在来判断——调用方始终设置 thread_id，
    # 即使 checkpointer=None。因此由 create_agent_graph 在编译时
    # 将 checkpointer 可用性写入 graph 自定义属性，调用方在
    # config.configurable.__has_checkpointer__ 中显式传递。
    has_checkpointer = config.get("configurable", {}).get("__has_checkpointer__", False)

    if not has_checkpointer:
        # 无 checkpointer：interrupt() 无法工作
        tool_names = ", ".join(tc["name"] for tc in dangerous_calls)
        # D19: 默认放行而非拒绝。无 checkpointer 时拒绝会导致 Agent 完全无法工作
        # （write_file/edit_file/run_command 都是危险工具）。
        # 设置 AUTO_APPROVE_WITHOUT_CHECKPOINTER=false 可改为拒绝（严格模式）。
        auto_approve = os.environ.get("AUTO_APPROVE_WITHOUT_CHECKPOINTER", "true").lower() in ("true", "1", "yes")
        if auto_approve:
            logger.warning(
                "[D7] 无 checkpointer，危险工具自动放行: %s。"
                "启用 checkpointer 可获得安全审批能力。"
                "设置 AUTO_APPROVE_WITHOUT_CHECKPOINTER=false 可改为拒绝。",
                tool_names,
            )
            # 结构化审计日志：记录自动放行的详细信息，便于事后追溯
            audit_logger.warning(
                "AUTO_APPROVE | tools=%s | reason=no_checkpointer | args_summary=%s",
                tool_names,
                {tc["name"]: str(tc.get("args", ""))[:100] for tc in dangerous_calls},
            )
            return {"messages": [], "_approval_next": "tools"}
        else:
            logger.error(
                "[D7] 无 checkpointer 且 AUTO_APPROVE_WITHOUT_CHECKPOINTER=false，"
                "拒绝危险工具调用: %s",
                tool_names,
            )
            reject_msg = HumanMessage(content=f"[安全拒绝] 无 checkpointer，危险工具调用被拒绝: {tool_names}。请启用 checkpointer 或设置 AUTO_APPROVE_WITHOUT_CHECKPOINTER=true。")
            return {"messages": [reject_msg], "_approval_next": "agent"}

    # 构建审批请求
    approval_request = {
        "type": "dangerous_tool_approval",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
            for tc in dangerous_calls
        ],
    }

    # 暂停执行，返回审批请求给调用方
    decision = interrupt(approval_request)

    if decision == "approve":
        # 放行：tools 节点会正常执行这些 tool_calls
        audit_logger.info(
            "APPROVED | tools=%s | decision=approve",
            ", ".join(tc["name"] for tc in dangerous_calls),
        )

        # D20: Interrupt 循环检测 — 检测真正的死循环（同一工具+相似参数反复调用）
        # 而非简单计数（正常连续写操作不应触发）
        prev_count = state.get("_interrupt_repeat_count", 0)
        prev_tool_sig = state.get("_interrupt_last_tool_sig", "")
        # 生成当前调用的签名：工具名+参数摘要（截断避免过长）
        current_sig = "|".join(
            f"{tc['name']}:{str(tc.get('args', ''))[:80]}"
            for tc in dangerous_calls
        )
        # 只有与上次签名高度相似时才增加计数（真正的循环）
        # 简单启发式：工具名集合相同 + 参数有重叠
        current_tool_names = frozenset(tc["name"] for tc in dangerous_calls)
        _sig_overlap = False
        if prev_tool_sig:
            prev_names = frozenset(part.split(":")[0] for part in prev_tool_sig.split("|"))
            if current_tool_names == prev_names:
                _sig_overlap = True
        repeat_count = prev_count + 1 if _sig_overlap else 1
        updates: dict = {
            "_approval_next": "tools",
            "_interrupt_repeat_count": repeat_count,
            "_interrupt_last_tool_sig": current_sig,
        }

        _WARN_THRESHOLD = 5
        _FORCE_PHASE_THRESHOLD = 8

        if repeat_count >= _FORCE_PHASE_THRESHOLD:
            # 连续批准过多：强制解除工具限制 + 注入强警告
            logger.warning(
                "[D20] Interrupt 循环检测: 连续批准 %d 次，强制解除工具限制",
                repeat_count,
            )
            warn_msg = SystemMessage(
                content=(
                    f"[系统警告] 你已经连续 {repeat_count} 次请求审批执行操作，"
                    "可能陷入了无效循环。请重新审视你的策略：\n"
                    "1. 你是否在反复尝试同一种失败的方法？\n"
                    "2. 是否应该换一种完全不同的方式来完成任务？\n"
                    "3. 如果当前方法不可行，请直接向用户报告困难。\n"
                    "工具限制已解除，你可以自由选择任何工具。"
                )
            )
            updates["messages"] = [warn_msg]
            updates["_phase"] = "all"
            updates["_stall_count"] = 0
        elif repeat_count >= _WARN_THRESHOLD:
            # 连续批准较多：注入温和警告
            logger.info(
                "[D20] Interrupt 循环检测: 连续批准 %d 次，注入警告",
                repeat_count,
            )
            warn_msg = SystemMessage(
                content=(
                    f"[系统提示] 你已经连续 {repeat_count} 次请求审批执行操作。"
                    "如果你发现当前策略不奏效，请尝试换一种方式。"
                )
            )
            updates["messages"] = [warn_msg]

        return updates
    else:
        # 拒绝：注入拒绝消息，并回退 phase 到 gather（审批拒绝说明不应修改）
        tool_names = ", ".join(tc["name"] for tc in dangerous_calls)
        audit_logger.warning(
            "REJECTED | tools=%s | decision=reject",
            tool_names,
        )
        # 为每个被拒绝的 tool_call 生成 ToolMessage(status="error")，
        # 确保 AIMessage(tool_calls) 的完整性——API 要求每个 tool_call_id
        # 都有对应的 ToolMessage，否则后续请求会报 400 错误。
        reject_tool_messages = [
            ToolMessage(
                content=f"[人工审批] 操作已被拒绝: {tc['name']}",
                tool_call_id=tc["id"],
                name=tc["name"],
                status="error",
            )
            for tc in dangerous_calls
        ]
        current_phase = state.get("_phase", "gather")
        rollback_phase = "gather" if current_phase in ("modify", "verify") else current_phase
        return {"messages": reject_tool_messages, "_approval_next": "agent", "_phase": rollback_phase, "_stall_count": 0}


def _route_after_approval(state: AgentState) -> str:
    """interrupt_approval 后路由：根据审批结果决定下一步。"""
    return state.get("_approval_next", "agent")


# ── Reflection 节点 ────────────────────────────────────────────

async def _reflect_node(state: AgentState, config: RunnableConfig) -> dict:
    """反思节点：评估当前工作质量，决定是否需要继续。

    触发条件：agent 在 verify 阶段返回文本（无 tool_calls），
    且反思次数未超限。

    行为：
    - 用无工具 LLM 评估工作质量
    - 通过 → 结束对话
    - 回退 → 修改 _phase 为 modify，注入反思结论，agent 继续工作
    """
    from agent_core.tools import _ctx_llm_ref

    llm = _ctx_llm_ref.get()
    reflection_count = state.get("_reflection_count", 0) or 0

    if llm is None:
        return {"_reflection_result": "[反思: 通过]", "_reflection_count": reflection_count + 1}

    plan = state.get("plan", "无计划")
    phase = state.get("_phase", "gather")

    # 取最近 6 条消息作为反思输入（避免 token 浪费）
    recent_messages = list(state["messages"][-6:])

    reflection_input = _REFLECTION_PROMPT.format(plan=plan, phase=phase)

    try:
        planner_llm = llm.bind_tools([])
        response = await planner_llm.ainvoke(
            [SystemMessage(content=reflection_input)] + recent_messages,
            config=config,
        )
        result_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.warning("[Reflection] 反思失败: %s，默认通过", exc)
        result_text = "[反思: 通过]"

    # 判断反思结论
    needs_rollback = "[反思: 回退]" in result_text

    updates: dict = {
        "_reflection_result": result_text,
        "_reflection_count": reflection_count + 1,
    }

    if needs_rollback:
        # 回退到 modify 阶段，注入反思结论让 agent 知道需要修复什么
        reflection_msg = SystemMessage(
            content=f"[反思评估 - 需要修复]\n{result_text}\n\n"
            "请根据以上反思结论修复问题，修复完成后重新进入验证阶段。"
        )
        updates["messages"] = [reflection_msg]
        updates["_phase"] = "modify"
        updates["_stall_count"] = 0
        updates["_deviation_count"] = 0
        updates["_deviation_reason"] = ""
        logger.info("[Reflection] 反思回退: %s", result_text[:100])
    else:
        # 通过，检查是否需要补充 JSON 摘要（开发阶段常见问题）
        needs_json_summary = False
        json_prompt = ""
        if phase in ("verify", "modify"):
            # 检查最近消息中是否包含 JSON 摘要
            last_ai_content = ""
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    last_ai_content = str(msg.content)
                    break
            if "project_name" not in last_ai_content or "files_created" not in last_ai_content:
                needs_json_summary = True
                json_prompt = (
                    "\n\n⚠️ 你的最终输出中缺少 JSON 摘要。"
                    "请在回复末尾补充如下格式的 JSON 摘要（放在 ```json 代码块中）：\n"
                    '```json\n{"project_name": "项目名", "files_created": ["文件1路径"], '
                    '"tech_stack": "技术栈", "setup_instructions": "安装运行步骤"}\n```'
                )

        reflection_msg = SystemMessage(
            content=f"[反思评估 - 通过]\n{result_text}{json_prompt}"
        )
        updates["messages"] = [reflection_msg]
        logger.info("[Reflection] 反思通过%s", " (已追加JSON摘要提示)" if needs_json_summary else "")

    return updates


def _route_after_reflect(state: AgentState) -> str:
    """反思后路由：根据反思结论决定继续还是结束。"""
    result = state.get("_reflection_result", "")
    if "[反思: 回退]" in result:
        return "agent"
    return END


# ── Replanner 节点 ─────────────────────────────────────────────

def _extract_recent_summary(messages: list[BaseMessage], max_messages: int = 4) -> str:
    """从最近消息中提取执行摘要，供 replanner 使用。"""
    recent = messages[-max_messages:] if len(messages) > max_messages else list(messages)
    parts: list[str] = []
    for msg in recent:
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content:
                parts.append(f"[AI] {content[:200]}")
            if msg.tool_calls:
                tool_names = ", ".join(tc["name"] for tc in msg.tool_calls)
                parts.append(f"[AI 调用工具] {tool_names}")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            status = " (错误)" if msg.status == "error" else ""
            parts.append(f"[工具结果{status}] {content[:150]}")
        elif isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[用户] {content[:100]}")
    return "\n".join(parts) if parts else "无最近执行记录"


async def _replanner_node(state: AgentState, config: RunnableConfig) -> dict:
    """计划修正节点：基于执行偏差重新规划。

    触发条件：_deviation_count >= 2（连续工具错误或计划步骤停滞）。

    行为：
    - 用无工具 LLM 修正计划
    - 重置偏差计数和停滞计数
    - 注入修正通知到消息流
    """
    from agent_core.tools import _ctx_llm_ref

    llm = _ctx_llm_ref.get()

    if llm is None:
        return {"_deviation_count": 0, "_deviation_reason": ""}

    original_plan = state.get("plan", "无计划")
    deviation_reason = state.get("_deviation_reason", "执行偏离原计划")
    recent_summary = _extract_recent_summary(list(state["messages"]))

    replanner_input = _REPLANNER_PROMPT.format(
        original_plan=original_plan,
        deviation_reason=deviation_reason,
        recent_summary=recent_summary,
    )

    try:
        planner_llm = llm.bind_tools([])
        response = await planner_llm.ainvoke(
            [HumanMessage(content=replanner_input)],
            config=config,
        )
        new_plan = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.warning("[Replanner] 重规划失败: %s，保持原计划", exc)
        new_plan = original_plan

    # 从修正后的计划中提取新阶段
    new_phase = "gather"
    phase_match = re.search(r'\[阶段:\s*(gather|modify|verify)\]', new_plan)
    if phase_match:
        new_phase = phase_match.group(1)

    # 注入重规划通知
    replan_msg = SystemMessage(
        content=f"[计划修正]\n原因: {deviation_reason}\n修正后计划:\n{new_plan}"
    )

    logger.info("[Replanner] 计划修正: phase=%s, reason=%s", new_phase, deviation_reason[:80])

    return {
        "plan": new_plan,
        "_phase": new_phase,
        "_deviation_count": 0,
        "_deviation_reason": "",
        "_stall_count": 0,
        "messages": [replan_msg],
    }


def _route_after_advance(state: AgentState) -> str | list[Send]:
    """advance_phase 后路由：先检查偏差，再走原有路由逻辑。

    偏差检测优先级高于反思路由，因为计划层面的问题应先修正，
    再让 agent 按修正后的计划执行。
    """
    deviation_count = state.get("_deviation_count", 0) or 0

    # 偏差阈值：连续 2 次偏差触发重规划
    if deviation_count >= 2:
        return "replanner"

    # 无偏差或偏差未达阈值 → 走原有路由
    return _route_after_agent(state)


def create_agent_graph(
    llm, tools, system_prompt: str,
    llm_callbacks: list | None = None,
    checkpointer=None,
):
    """构造并编译主 Agent 的 StateGraph。

    图结构：
        START → _should_plan
                    ├── plan 非空 → agent（已有计划，直接执行）
                    └── plan 为空 → planner → agent（新请求，先规划再执行）
                        agent → advance_phase → _route_after_advance
                            ├── deviation_count >= 2 → replanner → agent
                            └── _route_after_agent
                                ├── 无 tool_calls + verify阶段 → reflect → _route_after_reflect
                                │     ├── 通过 → END
                                │     └── 回退 → agent (phase回退到modify)
                                ├── 无 tool_calls + 非verify → END
                                ├── dispatch_subagent_lg → subagent_dispatcher
                                │     └── [Send × N] → subagent_worker → aggregate_results → agent
                                ├── 危险工具 → interrupt_approval
                                │     ├── approve → tools → agent
                                │     └── reject → agent
                                └── 普通工具 → tools → agent

    Args:
        llm: 绑定了 DeepSeek API 的 ChatOpenAI 实例（支持 reasoning_content）
        tools: 工具列表
        system_prompt: 构建好的系统提示词
        llm_callbacks: 图级回调（始终生效，如 token 追踪）。
            LangGraph >= 1.2 不再支持 compile(callbacks=...)，
            改为在 invoke 时通过 config.callbacks 传入。
            此参数保留用于向后兼容，实际回调在 invoke 时合并。
        checkpointer: 可选的 LangGraph Checkpointer（如 SqliteSaver），
            启用后支持状态持久化、时间旅行调试和断点续跑

    Returns:
        CompiledGraph

    Note:
        - 回调通过 invoke(input, config={"callbacks": [...]}) 传入，
          在节点内透传给 LLM，使 ReasoningCollector 等能捕获 reasoning_content
        - 调用方应传入 {"messages": [SystemMessage, ...chat_history, HumanMessage(input)]}
          而非分开传 chat_history 和 input，避免多轮循环时重复拼接
        - interrupt() 需要 checkpointer 才能工作（状态需持久化才能恢复）
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ParallelToolNode(tools)

    # 上下文视图裁剪器：在注入 LLM 前裁剪消息，state 保持完整
    from agent_core.context_view import ContextView, _ensure_tool_call_integrity
    from agent_core.in_context_compactor import InContextCompactor
    from agent_core.decision_summary import DecisionSummaryExtractor, merge_summaries
    from agent_core.observation_masker import ObservationMasker
    _context_view = ContextView()
    _in_context_compactor = InContextCompactor()
    _summary_extractor = DecisionSummaryExtractor()
    _observation_masker = ObservationMasker()

    def _inject_system_messages(
        view: list[BaseMessage],
        injections: list[tuple[int, SystemMessage]],
    ) -> list[BaseMessage]:
        """按优先级注入 SystemMessage。

        Args:
            view: 当前消息列表
            injections: [(priority, SystemMessage)] 列表，
                priority 越小越靠前（越接近第一条 SystemMessage）

        优先级约定：
            10 = 决策摘要（最基础，始终存在）
            20 = 执行计划（指导当前任务）
            30 = 批量调用提示（效率优化，可忽略）
        """
        if not injections:
            return view
        sorted_msgs = [msg for _, msg in sorted(injections, key=lambda x: x[0])]
        if view and isinstance(view[0], SystemMessage):
            return [view[0]] + sorted_msgs + view[1:]
        return sorted_msgs + view

    async def call_agent(state: AgentState, config: RunnableConfig) -> dict:
        # 构建裁剪视图：LLM 只看到裁剪后的消息，state 保持完整
        # 这样 Checkpointer 保存完整序列，interrupt 恢复时状态一致
        full_messages = state["messages"]

        # 步骤1: ContextView 裁剪 — 决定哪些消息保留 + 收集被裁剪组
        view, pruned_groups = _context_view.build_view(list(full_messages))

        # 步骤2: 决策摘要注入 — 从被裁剪组提取关键信息（T3）
        new_summary = ""
        if pruned_groups:
            new_summary = _summary_extractor.extract(pruned_groups)

        # 合并新旧摘要（跨压缩保持决策链）
        old_summary = state.get("_compaction_summary", "")
        if new_summary and old_summary:
            merged = merge_summaries(old_summary, new_summary)
        elif new_summary:
            merged = new_summary
        else:
            merged = old_summary

        # 收集待注入的 SystemMessage（按优先级排序，一次性注入）
        injections: list[tuple[int, SystemMessage]] = []

        # 优先级 10: 决策摘要
        if merged:
            injections.append((10, SystemMessage(content=merged)))

        # 优先级 20: 执行计划
        plan = state.get("plan", "")
        if plan and plan != "无需规划":
            plan_instruction = (
                f"[执行计划]\n{plan}\n\n"
                "请严格按计划执行，特别注意每步的执行者选择：\n"
                "- 执行者=自己 → 你直接调用对应工具\n"
                "- 执行者=派遣 xxx → 你调用 dispatch_subagent_lg(agent_name='xxx', task='...')\n"
                "不要跳过计划中标注的派遣步骤，也不要把派遣步骤替换为自己执行。"
            )
            injections.append((20, SystemMessage(content=plan_instruction)))

        # 优先级 30: 批量调用提示
        phase = state.get("_phase", "gather")
        stall_count = state.get("_stall_count", 0)
        if phase in ("modify", "verify") and len(view) > 10:
            injections.append((30, SystemMessage(
                content="[效率提示] 如果需要连续修改多个文件或执行多个操作，"
                "请在同一轮中批量发出所有 tool_calls，而非逐个调用。"
                "例如：同时发出多个 edit_file 调用，而非每次只发一个。"
            )))

        view = _inject_system_messages(view, injections)

        # 步骤3: 观察遮蔽 — 对只读工具的大体积输出做结构保留式压缩（T3）
        view = _observation_masker.mask(view)

        # 步骤4: InContextCompactor 压缩 — 截断旧的大体积 ToolMessage
        view = _in_context_compactor.compact(view)

        # 步骤4.5: 安全网 — 最终完整性校验，确保 view 中无悬空的 tool_calls
        # 这是对 ContextView / ObservationMasker / InContextCompactor 的防御性兜底
        view = _ensure_tool_call_integrity(view)

        # 步骤5: 自适应工具选择（P3）— 根据 _phase 过滤工具
        llm_bound = _bind_tools_for_phase(llm, tools, llm_with_tools, phase, stall_count)

        response = await llm_bound.ainvoke(view, config=config)

        # 返回摘要到 state（跨压缩保持）
        updates: dict = {"messages": [response]}
        if merged != old_summary:
            updates["_compaction_summary"] = merged
        # D20: agent 节点正常返回时重置 interrupt 循环计数
        # （非 interrupt 路径说明 Agent 在正常推进，不应累积计数）
        if not response.tool_calls:
            updates["_interrupt_repeat_count"] = 0
            updates["_interrupt_last_tool_sig"] = ""
        return updates

    builder = StateGraph(AgentState)

    # 节点
    builder.add_node("planner", _plan_node)
    builder.add_node("agent", call_agent)
    builder.add_node("advance_phase", _advance_phase)
    builder.add_node("tools", tool_node)
    builder.add_node("todos_inline", _todos_inline)
    builder.add_node("subagent_worker", _subagent_worker)
    builder.add_node("aggregate_results", _aggregate_results)
    builder.add_node("interrupt_approval", _interrupt_approval)
    builder.add_node("reflect", _reflect_node)
    builder.add_node("replanner", _replanner_node)

    # 边
    # START → 条件路由：有计划直接进 agent，无计划先规划
    builder.add_conditional_edges(START, _should_plan)
    builder.add_edge("planner", "agent")
    # agent → advance_phase → _route_after_advance（偏差检测优先，再走原有路由）
    builder.add_edge("agent", "advance_phase")
    builder.add_conditional_edges("advance_phase", _route_after_advance)
    # replanner 修正计划后回到 agent
    builder.add_edge("replanner", "agent")
    # reflect 反思后路由：通过 → END，回退 → agent
    builder.add_conditional_edges("reflect", _route_after_reflect)
    builder.add_edge("tools", "agent")
    builder.add_edge("todos_inline", "agent")
    builder.add_edge("subagent_worker", "aggregate_results")
    builder.add_conditional_edges("aggregate_results", _route_after_aggregate)
    builder.add_conditional_edges("interrupt_approval", _route_after_approval)

    compiled = builder.compile(checkpointer=checkpointer)
    # 保存 llm_callbacks 供调用方在 invoke 时合并到 config.callbacks
    compiled._lg_llm_callbacks = list(llm_callbacks) if llm_callbacks else []
    # 保存 checkpointer 可用性，供调用方在 config.configurable 中传递
    compiled._lg_has_checkpointer = checkpointer is not None
    return compiled
