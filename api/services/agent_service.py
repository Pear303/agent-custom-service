"""Agent 服务：Dify 路由 + 需求分析"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from .session_manager import SessionManager, Session
from ..clients.dify import DifyChatflowClient
from ..core.config import settings
from ..utils.file_manager import _clean_output_path
from agent.factory import create_agent
from agent.lc_tools import set_workspace, set_skills_loader, set_todo_store, set_subagent_deps, set_user_id, set_ticket_id, clear_context, _build_workspace

# LangGraph Agent 支持（可选，需安装 langgraph-checkpoint-sqlite）
_USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "false").lower() in ("true", "1", "yes")

if _USE_LANGGRAPH:
    try:
        from agent_by_langgraph.factory import create_lg_agent
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        logger.warning("LangGraph Agent 不可用，回退到 LCAgent")
        _USE_LANGGRAPH = False

# 启动时检测 Checkpointer 依赖
if _USE_LANGGRAPH:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        logger.info("[启动检测] langgraph-checkpoint-sqlite 已安装，Checkpointer 功能可用")
    except ImportError:
        logger.warning(
            "[启动检测] langgraph-checkpoint-sqlite 未安装！"
            "Agent 将运行但无状态持久化/增量更新能力。"
            "请运行: pip install langgraph-checkpoint-sqlite>=2.0.0"
        )

logger = logging.getLogger(__name__)

# 启动时醒目打印当前 Agent 引擎选择
_AGENT_ENGINE = "LangGraph (StateGraph)" if _USE_LANGGRAPH else "LangChain (AgentExecutor)"
logger.info("=" * 60)
logger.info("  Agent 引擎: %s", _AGENT_ENGINE)
logger.info("  切换方式: 设置 USE_LANGGRAPH=%s", "false" if _USE_LANGGRAPH else "true")
logger.info("=" * 60)

_PROJECT_ROOT = Path(__file__).parent.parent.parent

_agent_semaphore = asyncio.Semaphore(settings.agent_concurrency_limit)

_MAX_RETRIES = settings.agent_max_retries
_BASE_DELAY = settings.agent_base_delay


def _normalize_cjk_quotes(content: str) -> str:
    """将中文弯引号（\u201c \u201d \u300c \u300d）替换为转义 ASCII 引号，
    防止 LLM 在 JSON 字符串值内使用中文引号导致解析截断。"""
    return content.replace('\u201c', '\\"').replace('\u201d', '\\"').replace('\u300c', '\\"').replace('\u300d', '\\"')


def _fix_unescaped_inner_quotes(content: str) -> str:
    """修复 LLM 在 JSON 字符串值内使用未转义 ASCII 双引号的问题。

    核心思路：逐字符状态机遍历，当在字符串内部遇到 " 时，
    检查其后是否紧跟合法的 JSON 续接符（, : ] } 或空白后接这些），
    若否则判定为未转义的内部引号，自动转义为 \"。
    """
    result: list[str] = []
    i = 0
    in_string = False
    n = len(content)

    while i < n:
        ch = content[i]

        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
        else:
            if ch == '\\' and i + 1 < n:
                # 已有转义，原样保留
                result.append(ch)
                result.append(content[i + 1])
                i += 1
            elif ch == '"':
                # 判断：字符串结束符 vs 未转义的内部引号
                rest = content[i + 1:]
                stripped = rest.lstrip()
                if not stripped or stripped[0] in ',:]}\r\n':
                    # 后面是合法 JSON 续接 → 字符串结束
                    result.append('"')
                    in_string = False
                else:
                    # 后面不是合法续接 → 未转义内部引号
                    result.append('\\"')
            else:
                result.append(ch)

        i += 1

    return ''.join(result)


def _parse_json_safe(content: str, _debug_label: str = "") -> dict | None:
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()

    strategies = [
        lambda c: json.loads(c),
        lambda c: json.loads(_fix_unescaped_inner_quotes(c)),
        lambda c: json.loads(_normalize_cjk_quotes(c)),
        lambda c: json.loads(re.sub(r',\s*([}\]])', r'\1', c)),
        lambda c: json.loads(re.sub(r',\s*([}\]])', r'\1', _fix_unescaped_inner_quotes(c))),
        lambda c: json.loads(re.sub(r',\s*([}\]])', r'\1', _normalize_cjk_quotes(c))),
        lambda c: _try_truncated_json(c),
        lambda c: _try_truncated_json(_fix_unescaped_inner_quotes(c)),
        lambda c: _try_truncated_json(_normalize_cjk_quotes(c)),
        lambda c: json.loads(re.sub(r'(?m)^```(?:json)?\s*\n?|^\s*```\s*$', '', c).strip()),
        lambda c: json.loads(re.sub(r"(?<!\\)'", '"', c)),
        lambda c: _try_truncated_json(re.sub(r"(?<!\\)'", '"', c)),
    ]
    for _ in strategies:
        try:
            return _(content)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    _dump_debug_json(content, _debug_label)
    return None


def _dump_debug_json(content: str, label: str = "") -> None:
    debug_dir = _PROJECT_ROOT / "data" / "_json_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = re.sub(r"[^\w一-龥]", "_", label)[:40] if label else "unknown"
    filepath = debug_dir / f"parse_fail_{ts}_{safe_label}.json"
    filepath.write_text(content, encoding="utf-8")
    logger.warning("废弃 JSON 已保存到 %s (%d 字节)", filepath, len(content))


def _try_truncated_json(content: str) -> dict:
    start = content.find('{')
    if start == -1:
        start = content.find('[')
        close_char = ']'
        open_char = '['
    else:
        close_char = '}'
        open_char = '{'
    bracket_open = '['
    bracket_close = ']'
    if start == -1:
        raise json.JSONDecodeError("No opening brace/bracket found", content, 0)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if e.pos > 0 and "Extra data" in str(e):
            try:
                return json.loads(content[:e.pos])
            except json.JSONDecodeError:
                pass

    for end in range(len(content) - 1, start, -1):
        if content[end] == close_char:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                continue

    depth = {close_char: 0, bracket_close: 0}
    for ch in content[start:]:
        if ch == open_char:
            depth[close_char] += 1
        elif ch == close_char:
            depth[close_char] = max(0, depth[close_char] - 1)
        elif ch == bracket_open:
            depth[bracket_close] += 1
        elif ch == bracket_close:
            depth[bracket_close] = max(0, depth[bracket_close] - 1)
    suffix = bracket_close * depth[bracket_close] + close_char * depth[close_char]
    if suffix:
        return json.loads(content[start:] + suffix)

    raise json.JSONDecodeError("No valid JSON found in truncated content", content, 0)


def _invoke_in_thread(workspace, user_id, ticket_id, executor, input_data, output_subdir=None,
                       skills_loader=None, todo_store=None, sub_reg=None):
    """在线程中设置上下文变量 (ContextVar) 后执行调用。

    asyncio.to_thread 不会自动复制上下文变量，因此必须在新线程中显式设置。
    workspace 参数代表项目根目录，此函数会将其重置为完整的目标路径。
    output_subdir: 非空时使用 _build 临时目录作为 workspace，
                   避免 LLM 的 write_file 与最终输出目录冲突导致路径嵌套。
    """
    effective_ws = _build_workspace(workspace, user_id, ticket_id)
    logger.debug("[LangChain 执行] user_id=%s, ticket_id=%s", user_id, ticket_id)
    if output_subdir:
        # 使用 _build 临时目录：LLM 在此自由操作，不会污染最终输出目录
        effective_ws = effective_ws / "_build"
    effective_ws.mkdir(parents=True, exist_ok=True)
    set_workspace(effective_ws)
    set_user_id(user_id)
    set_ticket_id(ticket_id)
    if skills_loader is not None:
        set_skills_loader(skills_loader)
    if todo_store is not None:
        set_todo_store(todo_store)
    if sub_reg is not None:
        # sub_reg 是一个元组 (llm, registry) 或 registry 对象
        if isinstance(sub_reg, tuple) and len(sub_reg) == 2:
            set_subagent_deps(sub_reg[0], sub_reg[1])
        else:
            set_subagent_deps(None, sub_reg)
    try:
        return _invoke_with_retry(executor, input_data)
    finally:
        clear_context()


def _invoke_with_retry(executor, input_data, max_retries: int = _MAX_RETRIES):
    """带有指数退避重试机制的同步执行器调用。

    退避公式：delay = base_delay * (2 ^ attempt) + jitter
    - attempt=0: ~1s
    - attempt=1: ~2s
    - attempt=2: ~4s

    Args:
        executor: AgentExecutor 实例
        input_data: 输入数据字典
        max_retries: 最大重试次数

    Returns:
        执行结果字典

    Raises:
        若所有重试均失败，则抛出最后一次捕获的异常
    """
    import random
    import time
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return executor.invoke(input_data)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("第 %d 次尝试失败，%.1f 秒后重试: %s", attempt + 1, delay, exc)
                time.sleep(delay)
    raise last_exc


async def _invoke_lg_async(agent, prompt: str, output_subdir=None):
    """异步调用 LangGraph Agent（原生 async，无需 asyncio.run 嵌套）。

    替代旧的 _invoke_lg_in_thread（通过 asyncio.to_thread + asyncio.run 调用），
    直接在当前事件循环中 await agent.graph.ainvoke，消除线程切换开销。

    ContextVar 在 async 上下文中自动传播，无需手动快照/恢复。

    Checkpointer 利用策略：
    - 首轮（agent._first_turn=True）：传入完整上下文 [SystemMessage, ...chat_history, HumanMessage]
    - 后续轮次（agent._first_turn=False）：只传入 [HumanMessage]，
      checkpointer 自动恢复之前的 state（含 SystemMessage + 历史消息），
      add_messages reducer 将新 HumanMessage 追加到已有消息序列末尾。
    - 若 checkpointer 不可用（未安装 langgraph-checkpoint-sqlite），则始终传完整上下文。

    Args:
        agent: LangGraphAgent 实例
        prompt: 用户输入
        output_subdir: 非空时使用 _build 临时目录作为 workspace，
                       避免 LLM 的 write_file 与最终输出目录冲突导致路径嵌套。

    Returns:
        dict: {"output": 回复文本} 或 {"output": ..., "interrupt": ...}（需要审批时）
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from agent_by_langgraph.lg_agent import ReasoningCollector

    # 如果指定了 output_subdir，将 workspace 切换到 _build 临时目录
    # 避免 LLM 的 write_file 直接操作在 ticket 目录下导致路径嵌套
    original_workspace = None
    if output_subdir:
        original_workspace = _ctx_workspace.get()
        build_ws = original_workspace / "_build" if original_workspace else None
        if build_ws:
            build_ws.mkdir(parents=True, exist_ok=True)
            set_workspace(build_ws)
            logger.info("[LG Invoke] 使用 _build 隔离目录: %s", build_ws)

    try:
        return await _invoke_lg_async_inner(agent, prompt)
    finally:
        # 恢复原始 workspace
        if original_workspace is not None:
            set_workspace(original_workspace)


async def _invoke_lg_async_inner(agent, prompt: str):
    """_invoke_lg_async 的核心逻辑（workspace 已切换后执行）。"""
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from agent_by_langgraph.lg_agent import ReasoningCollector

    logger.debug(
        "[LangGraph 执行] user_id=%s, first_turn=%s, checkpointer=%s",
        agent.user_id, agent._first_turn, agent.graph.checkpointer is not None,
    )

    has_checkpointer = agent.graph.checkpointer is not None
    async with agent._async_invoke_lock:
        is_first_turn = agent._first_turn
        if is_first_turn and has_checkpointer:
            agent._first_turn = False

    # 构造输入消息：根据 checkpointer 状态和 _first_turn 决定是否传完整上下文
    if is_first_turn or not has_checkpointer:
        # 首轮或无 checkpointer：传入完整上下文
        initial_messages = [SystemMessage(content=agent._system_prompt)]
        initial_messages.extend(agent.memory_store.messages)
        initial_messages.append(HumanMessage(content=prompt))
        input_state = {"messages": initial_messages}
        msg_count = len(initial_messages)
        logger.info(
            "[LG Invoke] 完整上下文模式: first_turn=%s, checkpointer=%s, "
            "messages=%d (system=1 + history=%d + input=1)",
            is_first_turn, has_checkpointer, msg_count, msg_count - 2,
        )
    else:
        # 后续轮次 + checkpointer 可用：只传 HumanMessage，checkpointer 自动恢复
        input_state = {"messages": [HumanMessage(content=prompt)]}
        logger.info(
            "[LG Invoke] 增量模式: 只传 HumanMessage, checkpointer 自动恢复历史状态",
        )

    collector = ReasoningCollector()
    # 合并图级回调（TokenTracker）和 per-invoke 回调（ReasoningCollector）
    all_callbacks = list(getattr(agent.graph, '_lg_llm_callbacks', []))
    all_callbacks.append(collector)
    config = {
        "callbacks": all_callbacks,
        "recursion_limit": agent.max_iterations * 2 + 5,
        "configurable": {
            "thread_id": agent.user_id or "default",
            "__has_checkpointer__": getattr(agent.graph, '_lg_has_checkpointer', agent.graph.checkpointer is not None),
        },
    }
    try:
        # 原生 async 调用，无需 asyncio.run 嵌套
        # _async_invoke_lock 保护防止同一 user_id 的并发请求导致 checkpointer 状态混乱
        async with agent._async_invoke_lock:
            result = await agent.graph.ainvoke(input_state, config=config)
    except Exception:
        # 仅在异常时清理上下文，正常流程由调用方在 finally 中清理
        clear_context()
        raise

    # 检查是否有中断（interrupt_approval 节点暂停）
    # __interrupt__ 是 LangGraph 在 interrupt() 后自动添加到结果中的字段
    interrupts = result.get("__interrupt__")
    if interrupts:
        logger.info("[LG Invoke] 执行被中断，等待人工审批: %s", interrupts)
        return {
            "output": "",
            "interrupt": {
                "type": "dangerous_tool_approval",
                "data": interrupts,
                "thread_id": config["configurable"]["thread_id"],
            },
        }

    # 优先使用 ReasoningCollector 收集的完整 AIMessage（含 reasoning_content）
    collector_msg = collector.last
    if collector_msg is not None and collector_msg.content:
        content = collector_msg.content
        reply = content if isinstance(content, str) else str(content)
    else:
        messages = result.get("messages", [])
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                reply = content if isinstance(content, str) else str(content)
                break

    # 持久化到 MemoryStore（与 REPL run() 行为一致）
    agent.memory_store.append_history("user", prompt)
    if collector_msg is not None and (reply or collector_msg.additional_kwargs):
        agent.memory_store.append_history(
            "assistant", reply,
            additional_kwargs=collector_msg.additional_kwargs or None,
        )
    elif reply:
        agent.memory_store.append_history("assistant", reply)

    # 压缩检查
    if agent.token_tracker.should_compact(max_context=200_000, threshold=0.5):
        agent.compactor.compact_store()

    logger.info("[LG Invoke] 完成, reply 长度=%d", len(reply))
    return {"output": reply}


async def resume_lg_approval(agent, decision: str, thread_id: str) -> dict:
    """恢复被 interrupt() 暂停的 LangGraph Agent 执行。

    当 interrupt_approval 节点暂停执行后，调用方通过此函数恢复：
    - decision="approve": 放行危险工具调用
    - decision="reject": 拒绝危险工具调用

    Args:
        agent: LangGraphAgent 实例
        decision: 审批决定 ("approve" 或 "reject")
        thread_id: 被中断的线程 ID（用于恢复正确的 checkpoint）

    Returns:
        dict: {"output": 回复文本}
    """
    from langgraph.types import Command

    config = {
        "recursion_limit": agent.max_iterations * 2 + 5,
        "configurable": {"thread_id": thread_id},
    }

    # async 上下文中 ContextVar 自动传播，无需手动恢复

    try:
        result = await agent.graph.ainvoke(Command(resume=decision), config=config)
    finally:
        clear_context()

    messages = result.get("messages", [])
    reply = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            reply = content if isinstance(content, str) else str(content)
            break

    logger.info("[LG Resume] 审批恢复完成, decision=%s, reply 长度=%d", decision, len(reply))
    return {"output": reply}


REQUIREMENT_ANALYST_PROMPT = """你是需求分析师，负责将客户模糊的原始需求转化为结构化的需求简报。

收到客户需求后，按以下维度分析：

1. **项目概述**：一句话概括项目本质
2. **目标用户**：谁会使用这个产品
3. **核心功能**：3-5 个最关键的功能点
4. **非功能需求**：性能、安全、兼容性等
5. **约束条件**：预算、时间、技术限制
6. **风险点**：可能的技术或业务风险
7. **待澄清问题**：需要客户补充的信息

最后输出结构化需求简报（JSON 格式），并给出复杂度评估（简单/中等/复杂）。"""

PRODUCT_MANAGER_PROMPT = """你是产品经理，负责将需求分析转化为完整的产品需求文档（PRD）。

收到需求分析后，按以下维度设计：

1. **产品定位**：一句话描述产品价值和差异化
2. **功能清单**：按优先级排序（P0 核心/P1 重要/P2 锦上添花）
3. **用户故事**：3-5 个核心用户场景的完整描述
4. **信息架构**：主要页面和导航结构
5. **数据模型**：核心数据实体和关系
6. **验收标准**：每个 P0 功能的完成定义

最后输出 PRD（JSON 格式），包含功能总数、核心场景数和技术复杂度评估。"""

COST_ESTIMATOR_PROMPT = """你是成本估算师，负责根据需求分析和 PRD 计算开发成本和报价。

收到 PRD 后，按以下维度估算：

1. **人力成本**：前端/后端/UI/测试/项目管理的工时 × 单价
2. **基础设施成本**：服务器/云服务/第三方 API
3. **风险缓冲**：15% 应急预算
4. **利润空间**：25% 合理利润

最后输出报价单（JSON 格式），包含：
- 总报价（元）
- 分项明细
- 付款节点（如 3-4-3）
- 交付周期（周）
- 售后支持期限（月）"""


class AgentService:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self._dify: DifyChatflowClient | None = None

    async def _get_dify(self) -> DifyChatflowClient:
        if self._dify is None:
            self._dify = DifyChatflowClient()
        return self._dify

    async def chat(self, user_id: str, message: str) -> dict:
        session = await self.session_manager.get_or_create_async(user_id)
        dify = await self._get_dify()

        resp = await dify.chat(
            query=message,
            user_id=user_id,
            conversation_id=session.conversation_id,
        )
        session.conversation_id = resp.get("conversation_id")
        answer = resp.get("answer", "")

        session.history.append({"role": "user", "content": message, "timestamp": int(time.time())})
        session.history.append({"role": "assistant", "content": answer, "source": "dify", "timestamp": int(time.time())})
        await self.session_manager._save_session(session)
        return {
            "user_id": user_id,
            "answer": answer,
            "conversation_id": session.conversation_id,
            "source": "dify",
        }

    async def chat_stream(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        session = await self.session_manager.get_or_create_async(user_id)
        dify = await self._get_dify()
        full_answer = ""

        async for chunk in dify.chat_stream(
            query=message,
            user_id=user_id,
            conversation_id=session.conversation_id,
        ):
            event = chunk.get("event")
            if event == "message":
                chunk_text = chunk.get("answer", "")
                full_answer += chunk_text
                yield json.dumps({
                    "event": "message",
                    "answer": chunk_text,
                    "source": "dify",
                }) + "\n"
            elif event == "message_end":
                session.conversation_id = chunk.get("conversation_id")
                session.history.append({"role": "user", "content": message, "timestamp": int(time.time())})
                session.history.append({"role": "assistant", "content": full_answer, "source": "dify", "timestamp": int(time.time())})
                await self.session_manager._save_session(session)
                yield json.dumps({
                    "event": "message_end",
                    "conversation_id": chunk.get("conversation_id"),
                }) + "\n"
                return
            elif event == "error":
                raise RuntimeError(chunk.get("message", "Dify stream error"))

    async def analyze_requirement(self, user_id: str, requirement: dict, ticket_id: str | None = None) -> dict:
        """需求分析：将客户需求转化为结构化需求简报"""
        async with _agent_semaphore:
            if _USE_LANGGRAPH:
                logger.debug("[引擎路由] 需求分析 → LangGraph | user_id=%s", user_id)
                return await self._analyze_requirement_lg(user_id, requirement, ticket_id)
            logger.debug("[引擎路由] 需求分析 → LangChain | user_id=%s", user_id)
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            prompt = f"""基于以下客户需求，按要求输出 JSON 格式的需求分析结果：

{json.dumps(requirement, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{REQUIREMENT_ANALYST_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("需求分析响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "需求分析响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("需求分析 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"需求分析格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("需求分析失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    async def design_prd(self, user_id: str, analysis: dict, ticket_id: str | None = None) -> dict:
        async with _agent_semaphore:
            if _USE_LANGGRAPH:
                logger.debug("[引擎路由] PRD设计 → LangGraph | user_id=%s", user_id)
                return await self._design_prd_lg(user_id, analysis, ticket_id)
            logger.debug("[引擎路由] PRD设计 → LangChain | user_id=%s", user_id)
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            prompt = f"""基于以下需求分析结果，按要求输出 JSON 格式的 PRD：

{json.dumps(analysis, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{PRODUCT_MANAGER_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容（处理 LLM 可能返回的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("PRD 响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "PRD 响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("PRD JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"PRD 格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("PRD 设计失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    async def estimate_cost(self, user_id: str, prd: dict, analysis: dict, ticket_id: str | None = None) -> dict:
        async with _agent_semaphore:
            if _USE_LANGGRAPH:
                logger.debug("[引擎路由] 成本估算 → LangGraph | user_id=%s", user_id)
                return await self._estimate_cost_lg(user_id, prd, analysis, ticket_id)
            logger.debug("[引擎路由] 成本估算 → LangChain | user_id=%s", user_id)
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            combined = {**prd, **analysis}
            prompt = f"""基于以下 PRD 和需求分析，按要求输出 JSON 格式的成本估算：

{json.dumps(combined, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{COST_ESTIMATOR_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容（处理 LLM 可能返回的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("成本估算响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "成本估算响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("成本估算 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"成本估算格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("成本估算失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    DEVELOPER_PROMPT = """你是全栈开发工程师，负责根据产品需求文档（PRD）生成完整的项目代码。

收到 PRD 后，按以下维度生成代码：

1. **项目结构**：合理的目录组织
2. **核心代码**：实现所有 P0/P1 功能
3. **配置文件**：package.json / requirements.txt 等
4. **README**：项目说明和运行指南

⚠️ 重要约束：在本任务中，禁止使用 write_file 工具写文件。
你必须将所有代码内容包含在最终 JSON 输出的 files 字段中，
由系统后端统一写入文件系统。

最后输出开发结果（JSON 格式），包含：
- project_structure: 项目目录结构（树形文本）
- files: 生成的文件列表 [{path: "相对路径", content: "文件完整内容"}, ...]
- tech_stack: 使用的技术栈
- setup_instructions: 安装和运行步骤

输出要求：只输出 JSON，不要输出任何其他解释性文字。"""

    def _recover_from_build(self, user_id: str, ticket_id: str) -> dict | None:
        _build_dir = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "_build"
        if not _build_dir.exists():
            return None
        scanned_files: list[dict] = []
        blacklist_prefixes = (
            str(_build_dir / "data"),
            str(_build_dir / "-p"),
        )
        for fp in _build_dir.rglob("*"):
            if not fp.is_file():
                continue
            fp_str = str(fp)
            if any(fp_str.startswith(prefix) for prefix in blacklist_prefixes):
                continue
            try:
                rel = fp.relative_to(_build_dir)
            except ValueError:
                continue
            try:
                fc = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            if not fc.strip():
                continue
            scanned_files.append({"path": str(rel.as_posix()), "content": fc})
        if not scanned_files:
            return None
        scanned_files.sort(key=lambda entry: entry["path"])
        logger.info("从 _build 恢复 %d 个文件", len(scanned_files))
        return {
            "project_structure": "从 _build 目录恢复",
            "files": scanned_files,
            "tech_stack": "从 _build 恢复（未知）",
            "setup_instructions": "请参考项目 README",
            "_recovered_from_build": True,
        }

    async def develop_project(self, user_id: str, project_data: dict, ticket_id: str | None = None) -> dict:
        if _USE_LANGGRAPH:
            logger.debug("[引擎路由] 项目开发 → LangGraph | user_id=%s", user_id)
            return await self._develop_project_lg(user_id, project_data, ticket_id)
        logger.debug("[引擎路由] 项目开发 → LangChain | user_id=%s", user_id)

        async with _agent_semaphore:
            agent = create_agent(user_id=user_id, ticket_id=ticket_id, max_iterations=80)

            prompt = f"""基于以下项目数据，按要求输出 JSON 格式的开发结果：

{json.dumps(project_data, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{self.DEVELOPER_PROMPT}"""

            try:
                # 使用 asyncio.to_thread 避免阻塞事件循环
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    "成品",
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]

                # 检测 Agent 迭代耗尽
                if "max iterations" in content.lower() or "Agent stopped" in content:
                    logger.error("Agent 迭代次数耗尽，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "Agent 迭代次数耗尽，项目过于复杂未能完成。可重试或简化需求。"}
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                    data = _parse_json_safe(content)
                    if data is None:
                        raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                else:
                    # fallback: 从 _build 目录扫描 LLM 实际写出的文件
                    logger.warning("开发响应中未找到 JSON 内容，尝试从 _build 恢复")
                    data = self._recover_from_build(user_id, ticket_id)
                    if data is None:
                        logger.error("开发响应中未找到 JSON 内容，原始输出: %s", content[:200])
                        return {"status": "failed", "error": "开发响应格式无效"}

                # 将 LLM 生成的代码文件写入 data/users/{user_id}/{ticket_id}/成品/
                files = data.get("files")
                if files and isinstance(files, list) and ticket_id:
                    output_root = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "成品"
                    saved_count = 0
                    for entry in files:
                        if not isinstance(entry, dict):
                            continue
                        file_path = entry.get("path") or entry.get("file") or entry.get("name")
                        file_content = entry.get("content") or entry.get("code") or ""
                        if not file_path or not file_content:
                            continue
                        safe_path = _clean_output_path(file_path, user_id, ticket_id)
                        if safe_path is None:
                            continue
                        target = output_root / safe_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(file_content, encoding="utf-8")
                        saved_count += 1
                    if saved_count > 0:
                        logger.info("开发成品已保存 %d 个文件到 %s", saved_count, output_root)
                        data["_output_dir"] = str(output_root)
                        data["_file_count"] = saved_count

                # 开发完成后清理 _build 临时目录
                _build_dir = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "_build"
                if ticket_id and _build_dir.exists():
                    import shutil
                    try:
                        shutil.rmtree(_build_dir)
                        logger.info("已清理 _build 临时目录: %s", _build_dir)
                    except Exception as e:
                        logger.warning("清理 _build 目录失败: %s", e)

                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("开发 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"开发格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("开发失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    # ── LangGraph Agent 方法 ──────────────────────────────────────

    async def _analyze_requirement_lg(self, user_id: str, requirement: dict, ticket_id: str | None = None) -> dict:
        """需求分析（LangGraph Agent 版本）"""
        agent = create_lg_agent(user_id=user_id, ticket_id=ticket_id)

        prompt = f"""基于以下客户需求，按要求输出 JSON 格式的需求分析结果：

{json.dumps(requirement, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{REQUIREMENT_ANALYST_PROMPT}"""

        try:
            result = await _invoke_lg_async(agent, prompt)
            content = result["output"]
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                return {"status": "failed", "error": "需求分析响应格式无效"}
            data = _parse_json_safe(content)
            if data is None:
                raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
            return {"status": "completed", "data": data}
        except json.JSONDecodeError as exc:
            logger.error("LG 需求分析 JSON 解析失败: %s", exc)
            return {"status": "failed", "error": f"需求分析格式错误: {str(exc)}"}
        except Exception as exc:
            logger.error("LG 需求分析失败: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            clear_context()

    async def _design_prd_lg(self, user_id: str, analysis: dict, ticket_id: str | None = None) -> dict:
        """PRD 设计（LangGraph Agent 版本）"""
        agent = create_lg_agent(user_id=user_id, ticket_id=ticket_id)

        prompt = f"""基于以下需求分析结果，按要求输出 JSON 格式的 PRD：

{json.dumps(analysis, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{PRODUCT_MANAGER_PROMPT}"""

        try:
            result = await _invoke_lg_async(agent, prompt)
            content = result["output"]
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                return {"status": "failed", "error": "PRD 响应格式无效"}
            data = _parse_json_safe(content)
            if data is None:
                raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
            return {"status": "completed", "data": data}
        except json.JSONDecodeError as exc:
            logger.error("LG PRD JSON 解析失败: %s", exc)
            return {"status": "failed", "error": f"PRD 格式错误: {str(exc)}"}
        except Exception as exc:
            logger.error("LG PRD 设计失败: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            clear_context()

    async def _estimate_cost_lg(self, user_id: str, prd: dict, analysis: dict, ticket_id: str | None = None) -> dict:
        """成本估算（LangGraph Agent 版本）"""
        agent = create_lg_agent(user_id=user_id, ticket_id=ticket_id)

        combined = {**prd, **analysis}
        prompt = f"""基于以下 PRD 和需求分析，按要求输出 JSON 格式的成本估算：

{json.dumps(combined, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{COST_ESTIMATOR_PROMPT}"""

        try:
            result = await _invoke_lg_async(agent, prompt)
            content = result["output"]
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                return {"status": "failed", "error": "成本估算响应格式无效"}
            data = _parse_json_safe(content)
            if data is None:
                raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
            return {"status": "completed", "data": data}
        except json.JSONDecodeError as exc:
            logger.error("LG 成本估算 JSON 解析失败: %s", exc)
            return {"status": "failed", "error": f"成本估算格式错误: {str(exc)}"}
        except Exception as exc:
            logger.error("LG 成本估算失败: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            clear_context()

    async def _develop_project_lg(self, user_id: str, project_data: dict, ticket_id: str | None = None) -> dict:
        """项目开发（LangGraph Agent 版本）"""
        agent = create_lg_agent(user_id=user_id, ticket_id=ticket_id, max_iterations=80)

        prompt = f"""基于以下项目数据，按要求输出 JSON 格式的开发结果：

{json.dumps(project_data, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{self.DEVELOPER_PROMPT}"""

        try:
            result = await _invoke_lg_async(agent, prompt, output_subdir="成品")
            content = result["output"]

            # 检测 Agent 迭代耗尽
            if "max iterations" in content.lower() or "Agent stopped" in content:
                logger.error("LG Agent 迭代次数耗尽，原始输出: %s", content[:200])
                return {"status": "failed", "error": "Agent 迭代次数耗尽，项目过于复杂未能完成。可重试或简化需求。"}
            if "```" in content:
                content = content.split("```", 1)[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

            # 提取 JSON 内容
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                logger.error("LG 开发响应中未找到 JSON 内容，原始输出: %s", content[:200])
                return {"status": "failed", "error": "开发响应格式无效"}

            data = _parse_json_safe(content)
            if data is None:
                raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)

            # 将 LLM 生成的代码文件写入 data/users/{user_id}/{ticket_id}/成品/
            files = data.get("files")
            if files and isinstance(files, list) and ticket_id:
                output_root = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "成品"
                saved_count = 0
                for entry in files:
                    if not isinstance(entry, dict):
                        continue
                    file_path = entry.get("path") or entry.get("file") or entry.get("name")
                    file_content = entry.get("content") or entry.get("code") or ""
                    if not file_path or not file_content:
                        continue
                    safe_path = _clean_output_path(file_path, user_id, ticket_id)
                    if safe_path is None:
                        continue
                    target = output_root / safe_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(file_content, encoding="utf-8")
                    saved_count += 1
                if saved_count > 0:
                    logger.info("LG 开发成品已保存 %d 个文件到 %s", saved_count, output_root)
                    data["_output_dir"] = str(output_root)
                    data["_file_count"] = saved_count

            # 开发完成后清理 _build 临时目录
            _build_dir = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "_build"
            if ticket_id and _build_dir.exists():
                import shutil
                try:
                    shutil.rmtree(_build_dir)
                    logger.info("已清理 _build 临时目录: %s", _build_dir)
                except Exception as e:
                    logger.warning("清理 _build 目录失败: %s", e)

            return {"status": "completed", "data": data}
        except json.JSONDecodeError as exc:
            logger.error("LG 开发 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
            return {"status": "failed", "error": f"开发格式错误: {str(exc)}"}
        except Exception as exc:
            logger.error("LG 开发失败: %s", exc)
            return {"status": "failed", "error": str(exc)}
        finally:
            clear_context()


