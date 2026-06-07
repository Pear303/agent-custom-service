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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
# D5: 允许 run_command，因为只读命令（如 python test.py、ls）也是信息收集
# 注：dispatch_subagent_lg 在 gather 阶段可用，但 LLM 应优先派遣只读子代理
#     （SubagentSpec.read_only=True）。非只读子代理在 gather 阶段可能执行写操作，
#     违反 gather 只读语义，但保留此工具以支持灵活的信息收集场景。
_GATHER_TOOLS = frozenset({
    "read_file", "grep_tool", "glob_tool", "web_fetch",
    "load_skill", "dispatch_subagent_lg", "update_todos",
    "run_command",
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

# 子代理 checkpointer 缓存（线程安全）
_sub_checkpointer_cache: OrderedDict[str, tuple] = OrderedDict()
_sub_checkpointer_lock = threading.Lock()
_MAX_SUB_CHECKPOINTER_CACHE_SIZE = 50

# D14: 使用 UUID 哨兵值，避免与子代理输出内容冲突
import uuid
_SUBAGENT_CLEAR_SENTINEL = f"__SUBAGENT_CLEAR_{uuid.uuid4().hex}__"




async def _invoke_with_retry(subgraph, sub_input: dict, sub_config: dict, agent_name: str, max_retries: int = 3):
    """带重试的子图调用，处理 SQLite 'database is locked' 错误。

    高并发场景下多个子代理同时写入同一 SQLite 数据库，
    即使启用 WAL 模式仍可能短暂锁冲突。重试机制确保瞬态错误不导致任务失败。
    """
    import asyncio
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await subgraph.ainvoke(sub_input, config=sub_config)
        except Exception as exc:
            last_exc = exc
            err_msg = str(exc).lower()
            if "database is locked" in err_msg or "locked" in err_msg:
                wait = 0.5 * (2 ** attempt)  # 指数退避: 0.5s, 1s, 2s
                logger.warning(
                    "[SubCheckpointer] %s 数据库锁定，第 %d/%d 次重试（等待 %.1fs）",
                    agent_name, attempt + 1, max_retries, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise last_exc


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
    """agent 节点后的路由：区分普通工具调用、子代理派遣、危险工具审批、直接结束。

    路由优先级：
    1. 无 tool_calls → END
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
        return END

    # 分类 tool_calls
    subagent_calls = [tc for tc in last.tool_calls if tc["name"] == "dispatch_subagent_lg"]
    dangerous_calls = [tc for tc in last.tool_calls if tc["name"] in _DANGEROUS_TOOLS]
    normal_calls = [tc for tc in last.tool_calls if tc["name"] not in _DANGEROUS_TOOLS and tc["name"] != "dispatch_subagent_lg"]

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
    from agent.lc_tools import _ctx_llm_ref, _ctx_sub_reg, _ctx_user_id
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

        # 尝试从主图的 checkpointer 复用（同类型子代理共享 checkpointer 实例）
        # _subagent_worker 是 async 节点，已在事件循环中，
        # 可直接 await 初始化 AsyncSqliteSaver，无需 asyncio.run()
        sub_checkpointer = None
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from pathlib import Path
            db_path = Path("data") / "users" / user_id / "subagent_checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # 线程安全地获取或创建 checkpointer
            with _sub_checkpointer_lock:
                # LRU: 访问时移到末尾
                if user_id in _sub_checkpointer_cache:
                    _sub_checkpointer_cache.move_to_end(user_id)
                else:
                    # 超容量时淘汰最久未用的
                    while len(_sub_checkpointer_cache) >= _MAX_SUB_CHECKPOINTER_CACHE_SIZE:
                        oldest_key, (old_ctx, _) = _sub_checkpointer_cache.popitem(last=False)
                        try:
                            await old_ctx.__aexit__(None, None, None)
                            logger.info("[SubCheckpointer] LRU 淘汰: user_id=%s", oldest_key)
                        except Exception:
                            pass
                    ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
                    # 直接 await：_subagent_worker 是 async 节点，已在事件循环中
                    checkpointer_instance = await ctx.__aenter__()
                    # 启用 WAL 模式：子代理并行写入时不互相阻塞
                    try:
                        await checkpointer_instance.db.execute("PRAGMA journal_mode=WAL")
                        await checkpointer_instance.db.execute("PRAGMA busy_timeout=5000")
                    except Exception:
                        pass
                    _sub_checkpointer_cache[user_id] = (ctx, checkpointer_instance)
                sub_checkpointer = _sub_checkpointer_cache[user_id][1]
        except Exception as exc:
            logger.warning("[SubCheckpointer] 初始化失败，降级为无状态执行: %s", exc)

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
            result = await _invoke_with_retry(subgraph, sub_input, sub_config, agent_name)
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
    1. ## 结论/总结/结果 标题下的内容
    2. 最后 3 行非空文本
    """
    # 策略1: 提取结构化结论
    conclusion_match = re.search(
        r'(?:^|\n)##\s*(?:结论|总结|结果|Conclusion|Summary|Result)\s*\n(.*?)(?:\n##|\Z)',
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


# ── Plan-then-Execute ──────────────────────────────────────────

_PLANNER_PROMPT = """\
你是一个任务规划器。根据用户的请求，制定简洁的执行计划。

规则：
1. 步骤不超过 7 步
2. 每步明确说明要做什么和用什么工具
3. 优先信息收集（read_file, grep_tool, glob_tool），再执行修改
4. 最后一步必须是验证或总结
5. 如果请求很简单（问候、简单问答），输出"无需规划"

输出格式：
1. [步骤描述] → 工具: xxx
2. [步骤描述] → 工具: xxx
...

最后一行必须标注初始执行阶段（三选一）：
- [阶段: gather] — 需要先收集信息（搜索、阅读文件等）
- [阶段: modify] — 已有足够信息，直接修改代码
- [阶段: verify] — 只需验证或回答问题

用户请求：
{request}
"""


def _should_plan(state: AgentState) -> str:
    """判断是否需要规划：新请求总是重新规划，已有计划则直接执行。

    路由逻辑：
    - 最新消息是 milestone HumanMessage（新用户请求）→ "planner"
      即使旧 plan 残留也必须重新规划，避免旧计划劫持新请求
    - plan 非空且非新请求 → "agent"（循环中，按已有计划执行）
    - plan 为空 → "planner"（首轮或 plan 被清空）
    """
    last_msg = state["messages"][-1] if state["messages"] else None
    if isinstance(last_msg, HumanMessage):
        # milestone 标记的 HumanMessage 是新用户请求，必须重新规划
        if last_msg.metadata and last_msg.metadata.get("milestone"):
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
    - 只在 plan 为空时触发（新用户请求）
    - agent 循环中 plan 保持不变，agent 按计划执行
    - 新用户请求到来时 planner 重新生成计划
    - 从计划中提取初始执行阶段（P3 自适应工具选择）
    """
    from agent.lc_tools import _ctx_llm_ref

    llm = _ctx_llm_ref.get()
    if llm is None:
        return {"plan": "", "_phase": "gather", "_stall_count": 0}

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
        return {"plan": "", "_phase": "gather", "_stall_count": 0}

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

    return {"plan": plan_text, "_phase": phase, "_stall_count": 0}


def _extract_phase(plan_text: str) -> str:
    """从规划输出中提取执行阶段。

    查找 [阶段: xxx] 标记，未找到时根据内容推断：
    - 包含修改/编辑/写入关键词 → "modify"
    - 包含搜索/查找/阅读关键词 → "gather"
    - 默认 → "gather"（任何新任务都应先收集信息再修改）

    注：默认值从 "all" 改为 "gather"，确保 P3 自适应工具选择
    在大多数场景下生效。"all" 只在停滞兜底时使用。
    """
    match = re.search(r'\[阶段[:：]\s*(gather|modify|verify)\]', plan_text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 推断：包含修改关键词 → modify
    if re.search(r'(修改|编辑|写入|重写|删除|添加代码)', plan_text):
        return "modify"
    # 推断：包含搜索关键词 → gather
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


def _advance_phase(state: AgentState) -> dict:
    """自动推进执行阶段 + 停滞检测（P3）。

    阶段推进规则：
    - gather → modify：agent 调用了写工具（write_file, edit_file, run_command）
    - modify → verify：agent 返回文本（无 tool_calls），表示修改完成
    - verify → verify：保持不变

    停滞检测：
    - agent 在过滤模式下返回空 tool_calls → stall_count +1
    - 连续 2 次停滞 → 回退到 "all" 阶段

    注意：
    - 此节点在 agent 之后、route_after_agent 之前执行，
      不修改 messages，只更新 _phase 和 _stall_count。
    - 阶段推进是"预判"而非"确认"：如果危险工具被审批拒绝，
      下一轮 agent 会重新调用工具，_advance_phase 会再次评估。
      这不会造成问题，因为：
      a) 拒绝后 agent 回到 agent 节点，再次经过 _advance_phase
      b) 如果 agent 不再调用写工具，phase 不会继续推进
      c) 如果 agent 仍调用写工具，phase 保持 modify，行为一致
    """
    phase = state.get("_phase", "gather") or "gather"
    stall_count = state.get("_stall_count", 0) or 0

    last = state["messages"][-1] if state["messages"] else None
    if not isinstance(last, AIMessage):
        return {"_phase": phase, "_stall_count": stall_count}

    # 检测 agent 是否调用了写工具
    # D5: run_command 在 gather 阶段允许用于信息收集（如运行测试），
    # 不应触发 gather→modify 推进。只有 write_file/edit_file 才是真正的修改。
    _WRITE_TOOLS = frozenset({"write_file", "edit_file"})
    has_modify_calls = bool(last.tool_calls) and any(
        tc["name"] in _WRITE_TOOLS for tc in last.tool_calls
    )

    # 检测 agent 是否调用了只读工具（非写工具，含 run_command）
    has_read_only_calls = bool(last.tool_calls) and not has_modify_calls

    # 检测 agent 是否返回文本（无 tool_calls）
    has_text_response = not last.tool_calls and bool(last.content)

    new_phase = phase
    new_stall = stall_count

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

    result = {"_phase": new_phase, "_stall_count": new_stall}
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
        auto_approve = os.environ.get("AUTO_APPROVE_WITHOUT_CHECKPOINTER", "false").lower() in ("true", "1", "yes")
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
        return {"messages": [], "_approval_next": "tools"}
    else:
        # 拒绝：注入拒绝消息，并回退 phase 到 gather（审批拒绝说明不应修改）
        tool_names = ", ".join(tc["name"] for tc in dangerous_calls)
        audit_logger.warning(
            "REJECTED | tools=%s | decision=reject",
            tool_names,
        )
        reject_msg = AIMessage(
            content=f"[人工审批] 以下操作已被拒绝: {tool_names}"
        )
        current_phase = state.get("_phase", "gather")
        rollback_phase = "gather" if current_phase in ("modify", "verify") else current_phase
        return {"messages": [reject_msg], "_approval_next": "agent", "_phase": rollback_phase, "_stall_count": 0}


def _route_after_approval(state: AgentState) -> str:
    """interrupt_approval 后路由：根据审批结果决定下一步。"""
    return state.get("_approval_next", "agent")


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
                        agent → advance_phase → route_after_agent
                            ├── 无 tool_calls → END
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
    from agent.context_view import ContextView
    from agent.in_context_compactor import InContextCompactor
    _context_view = ContextView()
    _in_context_compactor = InContextCompactor()

    async def call_agent(state: AgentState, config: RunnableConfig) -> dict:
        # 构建裁剪视图：LLM 只看到裁剪后的消息，state 保持完整
        # 这样 Checkpointer 保存完整序列，interrupt 恢复时状态一致
        full_messages = state["messages"]
        # 步骤1: ContextView 裁剪 — 决定哪些消息保留
        view = _context_view.build_view(list(full_messages))
        # 步骤2: InContextCompactor 压缩 — 截断旧的大体积 ToolMessage
        view = _in_context_compactor.compact(view)

        # 步骤3: 注入执行计划（Plan-then-Execute）
        plan = state.get("plan", "")
        if plan and plan != "无需规划":
            plan_msg = SystemMessage(
                content=f"[执行计划]\n{plan}\n\n请按计划逐步执行。"
            )
            # 插入到第一条 SystemMessage 之后
            if view and isinstance(view[0], SystemMessage):
                view = [view[0], plan_msg] + view[1:]
            else:
                view = [plan_msg] + view

        # 步骤4: 自适应工具选择（P3）— 根据 _phase 过滤工具
        phase = state.get("_phase", "gather")
        stall_count = state.get("_stall_count", 0)
        llm_bound = _bind_tools_for_phase(llm, tools, llm_with_tools, phase, stall_count)

        response = await llm_bound.ainvoke(view, config=config)
        return {"messages": [response]}

    builder = StateGraph(AgentState)

    # 节点
    builder.add_node("planner", _plan_node)
    builder.add_node("agent", call_agent)
    builder.add_node("advance_phase", _advance_phase)
    builder.add_node("tools", tool_node)
    builder.add_node("subagent_worker", _subagent_worker)
    builder.add_node("aggregate_results", _aggregate_results)
    builder.add_node("interrupt_approval", _interrupt_approval)

    # 边
    # START → 条件路由：有计划直接进 agent，无计划先规划
    builder.add_conditional_edges(START, _should_plan)
    builder.add_edge("planner", "agent")
    # agent → advance_phase → route_after_agent（P3：先推进阶段，再路由）
    builder.add_edge("agent", "advance_phase")
    builder.add_conditional_edges("advance_phase", _route_after_agent)
    builder.add_edge("tools", "agent")
    builder.add_edge("subagent_worker", "aggregate_results")
    builder.add_conditional_edges("aggregate_results", _route_after_aggregate)
    builder.add_conditional_edges("interrupt_approval", _route_after_approval)

    compiled = builder.compile(checkpointer=checkpointer)
    # 保存 llm_callbacks 供调用方在 invoke 时合并到 config.callbacks
    compiled._lg_llm_callbacks = list(llm_callbacks) if llm_callbacks else []
    # 保存 checkpointer 可用性，供调用方在 config.configurable 中传递
    compiled._lg_has_checkpointer = checkpointer is not None
    return compiled
