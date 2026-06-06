"""LangGraph Agent —— 基于 StateGraph 的智能代理，替代 LCAgent + AgentExecutor。"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from openai import OpenAI as OpenAIClient

from agent.compactor import Compactor
from agent.context import ContextBuilder
from agent.lc_agent import DeepSeekChatOpenAI, create_deepseek_llm
from agent.lc_tools import (
    _build_workspace,
    edit_file,
    glob_tool,
    grep_tool,
    load_skill,
    read_file,
    run_command,
    set_subagent_deps,
    set_skills_loader,
    set_ticket_id,
    set_todo_store,
    set_user_id,
    set_workspace,
    update_todos,
    web_fetch,
    write_file,
)
from agent.memory import MemoryStore
from agent.subagents.registry import SubagentRegistry
from agent.telemetry import TokenTracker
from agent.todo import TodoStore
from agent_by_langgraph.lg_graph import create_agent_graph
from agent_by_langgraph.lg_tools import dispatch_subagent_lg

logger = logging.getLogger(__name__)


class TokenTrackerCallback(BaseCallbackHandler):
    """把 LLM token 用量记录到 TokenTracker。"""
    def __init__(self, tracker: TokenTracker, model_name: str):
        self._tracker = tracker
        self._model = model_name

    def on_llm_end(self, response, **kwargs):
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
        if not usage:
            try:
                usage = response.generations[0][0].message.usage_metadata
            except Exception:
                return
        if not usage:
            return
        input_tokens = getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0) or usage.get("total_tokens", 0)
        self._tracker.record_raw(self._model, input_tokens, output_tokens, total_tokens)


class ReasoningCollector(BaseCallbackHandler):
    """收集 LLM 调用中产生的完整 AIMessage（含 reasoning_content 等附加字段）。

    DeepSeek thinking mode 必须在多轮请求中原样回传 `reasoning_content`，
    否则下一轮 API 报错。LangGraph StateGraph 的 invoke() 返回值只包含
    最终 reply 文本，会丢 reasoning_content；通过 callback 在最底层拦截
    才能拿到完整 AIMessage。
    """
    def __init__(self):
        self.ai_messages: list[AIMessage] = []

    def on_llm_end(self, response, **kwargs):
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    if isinstance(gen.message, AIMessage):
                        self.ai_messages.append(gen.message)
        except Exception:
            pass

    @property
    def last(self) -> AIMessage | None:
        return self.ai_messages[-1] if self.ai_messages else None


class LangGraphAgent:
    """基于 LangGraph StateGraph 的智能代理类。

    功能等价于 LCAgent，使用 StateGraph + ToolNode 替代 AgentExecutor。
    本类不接生产路径，仅作 LangGraph 重写验证。
    """

    def __init__(
        self,
        user_id: str | None = None,
        ticket_id: str | None = None,
        root: Path | None = None,
        model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations: int = 50,
        llm: DeepSeekChatOpenAI | None = None,
        skills_loader: Any | None = None,
        sub_reg: Any | None = None,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        self.root = root or Path(__file__).parent.parent
        self.user_id = user_id
        self.ticket_id = ticket_id
        self.model = model
        self.max_iterations = max_iterations
        self._first_turn = True
        self._invoke_lock = threading.Lock()  # 保护 _first_turn 和 graph.ainvoke 的串行化（同步路径）
        self._async_invoke_lock = asyncio.Lock()  # 保护 async 路径的串行化

        self.llm = llm or create_deepseek_llm(model)

        workspace = _build_workspace(self.root, user_id, ticket_id)
        set_workspace(workspace)
        if user_id:
            set_user_id(user_id)
        if ticket_id:
            set_ticket_id(ticket_id)

        from agent.skills import get_skills_loader
        self.skills = skills_loader or get_skills_loader(self.root / "skills")
        set_skills_loader(self.skills)

        self.todo_store = TodoStore(user_id=user_id)
        set_todo_store(self.todo_store)

        # 工具列表：与 LCAgent 相同，但 dispatch_subagent 替换为 LangGraph 版
        self.tools = [
            read_file, write_file, edit_file,
            run_command, web_fetch, load_skill,
            glob_tool, grep_tool, update_todos,
            dispatch_subagent_lg,
        ]

        self.sub_reg = sub_reg or SubagentRegistry(
            self.root / "templates" / "subagents",
            skills_loader=self.skills,
        )
        set_subagent_deps(llm=self.llm, registry=self.sub_reg)

        # 保存 ContextVar 快照，供同步路径（REPL run）在新线程中恢复
        # async 路径（_invoke_lg_async）中 ContextVar 自动传播，无需快照
        from agent_by_langgraph.context_var_manager import snapshot
        self._ctx_snapshot = snapshot()

        if user_id:
            self.memory_store = MemoryStore(user_id=user_id)
            token_log_path = self.root / "data" / "users" / user_id / "tokens.jsonl"
        else:
            self.memory_store = MemoryStore(
                memory_dir=self.root / "memory",
                user_file=self.root / "templates" / "USER.md",
            )
            token_log_path = self.root / "memory" / "tokens.jsonl"

        self.token_tracker = TokenTracker(log_file=token_log_path)
        openai_client = OpenAIClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.compactor = Compactor(openai_client, model, self.memory_store)

        unarchived = self.memory_store.load_unarchived_history()
        if len(unarchived) >= 2:
            print(f"[Startup: found {len(unarchived)} unarchived turns, compacting...]")
            try:
                self.compactor.compact_startup(unarchived)
            except Exception as exc:
                print(f"[warning] startup compaction failed: {exc}", file=sys.stderr)

        ctx = ContextBuilder(
            self.root / "templates",
            self.skills,
            memory=self.memory_store,
        )
        system_prompt = ctx.build_system_prompt()

        # 初始化 LangGraph Checkpointer（SQLite 持久化）
        # 启用后支持：状态持久化、时间旅行调试、断点续跑
        checkpointer = self._init_checkpointer()

        self.graph = create_agent_graph(
            self.llm, self.tools, system_prompt,
            llm_callbacks=[TokenTrackerCallback(self.token_tracker, model)],
            checkpointer=checkpointer,
        )
        self._system_prompt = system_prompt

    def close(self) -> None:
        """释放资源：关闭 Checkpointer 的 SQLite 连接。

        应在 agent 不再使用时调用（如 LRU 淘汰、会话结束），
        确保 SQLite 连接正确关闭，避免资源泄漏。
        """
        self._close_checkpointer_ctx("_checkpointer_ctx", "主 Checkpointer")

        # 清理子代理 checkpointer 缓存
        from agent_by_langgraph.lg_graph import _sub_checkpointer_cache, _sub_checkpointer_lock
        with _sub_checkpointer_lock:
            if self.user_id and self.user_id in _sub_checkpointer_cache:
                sub_ctx, _ = _sub_checkpointer_cache.pop(self.user_id)
                self._close_checkpointer_ctx_obj(sub_ctx, f"子代理 Checkpointer (user_id={self.user_id})")

    @staticmethod
    def _close_checkpointer_ctx_obj(ctx, label: str) -> None:
        """可靠关闭一个 AsyncSqliteSaver 的 async context manager。"""
        if ctx is None:
            return
        try:
            import asyncio
            import concurrent.futures

            async def _do_close():
                await ctx.__aexit__(None, None, None)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 已在事件循环中：用新线程同步等待关闭完成
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _do_close())
                    future.result(timeout=5)
            else:
                # 无事件循环：直接运行
                asyncio.run(_do_close())

            logger.info("[Checkpointer] %s SQLite 连接已关闭", label)
        except Exception as exc:
            logger.warning("[Checkpointer] 关闭 %s SQLite 连接时出错: %s", label, exc)

    def _close_checkpointer_ctx(self, attr_name: str, label: str) -> None:
        """关闭实例上的 checkpointer context manager 属性。"""
        ctx = getattr(self, attr_name, None)
        if ctx is not None:
            setattr(self, attr_name, None)
            self._close_checkpointer_ctx_obj(ctx, label)

    def __del__(self) -> None:
        """析构时确保 Checkpointer 资源释放。"""
        self.close()

    def _init_checkpointer(self):
        """初始化 LangGraph Checkpointer。

        按 user_id 隔离 SQLite 数据库文件，支持：
        - 状态持久化：服务重启后可恢复对话状态
        - 时间旅行：可回溯到任意历史检查点
        - 断点续跑：中断后从最后检查点继续执行

        注意：langgraph-checkpoint-sqlite >= 3.0 的 from_conn_string()
        返回 context manager，需手动 __enter__() 保持长生命周期。

        由于图节点均为 async def，必须使用 AsyncSqliteSaver（而非同步 SqliteSaver），
        否则 ainvoke / astream 时会抛出 NotImplementedError。
        """
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            if self.user_id:
                db_path = self.root / "data" / "users" / self.user_id / "checkpoints.db"
            else:
                db_path = self.root / "data" / "checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # AsyncSqliteSaver.from_conn_string 返回 async context manager，
            # 必须在最终使用的事件循环中初始化（aiosqlite 内部线程与事件循环绑定）。
            # 此处只保存 db_path，实际初始化延迟到首次 ainvoke 时在正确的事件循环中完成。
            # 参见 _ensure_checkpointer() 方法。
            self._checkpointer_db_path = str(db_path)
            self._checkpointer_initialized = False

            logger.info("[Checkpointer] AsyncSqliteSaver 延迟初始化: %s", db_path)
            return None  # 延迟初始化，首次 ainvoke 时完成
        except ImportError:
            logger.warning(
                "[Checkpointer] langgraph-checkpoint-sqlite 未安装，"
                "状态持久化/时间旅行/断点续跑不可用。"
                "请运行: pip install langgraph-checkpoint-sqlite>=2.0.0"
            )
            return None
        except Exception as exc:
            logger.warning("[Checkpointer] 初始化失败: %s，checkpointer 降级为 None", exc)
            return None

    def run(self) -> None:
        """REPL 交互循环。每次输入调用编译后的 StateGraph。

        Checkpointer 利用策略：
        - 首轮（_first_turn=True）：传入完整消息 [SystemMessage, ...chat_history, HumanMessage]
        - 后续轮次（_first_turn=False）：只传入 [HumanMessage]，
          checkpointer 自动恢复之前的 state（含 SystemMessage + 历史消息），
          add_messages reducer 将新 HumanMessage 追加到已有消息序列末尾。

        注意：由于节点函数已改为 async def，需要用 asyncio.run() 包装 ainvoke。
        """
        while True:
            try:
                user_input = input("You🫅 : ")
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            # 流式输出回调：逐 token 打印（与 LCAgent 行为一致）
            class StreamHandler(BaseCallbackHandler):
                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    print(token, end="", flush=True)

            self._reasoning_collector = ReasoningCollector()
            stream_handler = StreamHandler()
            # 合并图级回调（TokenTracker）和 per-invoke 回调（ReasoningCollector, StreamHandler）
            all_callbacks = list(getattr(self.graph, '_lg_llm_callbacks', []))
            all_callbacks.extend([self._reasoning_collector, stream_handler])
            # 注意：checkpointer 可能延迟初始化，此时 graph.checkpointer 为 None
            # 但 _checkpointer_db_path 存在，意味着 checkpointer 将在 ainvoke 前初始化
            _will_have_checkpointer = (
                self.graph.checkpointer is not None
                or getattr(self, '_checkpointer_db_path', None) is not None
            )
            config: RunnableConfig = {
                "callbacks": all_callbacks,
                "recursion_limit": self.max_iterations * 4 + 10,
                "configurable": {
                    "thread_id": self.user_id or "default",
                    "__has_checkpointer__": _will_have_checkpointer,
                },
            }

            with self._invoke_lock:
                is_first_turn = self._first_turn
                if is_first_turn and _will_have_checkpointer:
                    self._first_turn = False

            if is_first_turn or not _will_have_checkpointer:
                # 首轮或无 checkpointer：传入完整上下文（system + chat_history + user input）
                initial_messages = [SystemMessage(content=self._system_prompt)]
                initial_messages.extend(self.memory_store.messages)
                # 标记用户原始请求为 milestone（ContextView 会优先保留）
                user_msg = HumanMessage(content=user_input)
                user_msg.metadata = {"milestone": True}
                initial_messages.append(user_msg)
                input_state = {"messages": initial_messages}
            else:
                # 后续轮次：只传入新的 HumanMessage，
                # checkpointer 自动恢复之前的 state
                user_msg = HumanMessage(content=user_input)
                user_msg.metadata = {"milestone": True}
                input_state = {"messages": [user_msg]}

            # 节点函数为 async def，需要用 asyncio.run 包装 ainvoke
            # REPL 场景下不会有正在运行的事件循环，直接用 asyncio.run()
            # 如果在已有事件循环中（不应出现在 REPL），使用新线程隔离
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            async def _invoke_with_checkpointer():
                """执行 ainvoke，自动处理 interrupt 恢复。

                interrupt() 暂停图后 ainvoke 返回当前 state，
                需要用 Command(resume="approve") 恢复执行。
                此循环自动批准所有 interrupt，直到图正常结束。
                """
                from langgraph.types import Command
                await self._ensure_checkpointer()
                # _ensure_checkpointer 可能重新编译图，需更新 config 中的回调
                config["callbacks"] = list(getattr(self.graph, '_lg_llm_callbacks', []))
                config["callbacks"].extend([self._reasoning_collector, stream_handler])
                config["configurable"]["__has_checkpointer__"] = self.graph.checkpointer is not None
                current_input = input_state
                max_interrupt_retries = 30
                for _ in range(max_interrupt_retries):
                    result = await self.graph.ainvoke(current_input, config=config)
                    # 用 aget_state 检查是否有 pending interrupt（比消息内容检测更可靠）
                    has_interrupt = False
                    if _will_have_checkpointer:
                        try:
                            snapshot = await self.graph.aget_state(config)
                            if snapshot and snapshot.next:
                                for task in snapshot.tasks:
                                    if hasattr(task, "interrupts") and task.interrupts:
                                        has_interrupt = True
                                        logger.info(
                                            "[Interrupt] 自动批准: %s",
                                            task.interrupts,
                                        )
                                        current_input = Command(resume="approve")
                                        break
                        except Exception as exc:
                            logger.warning("[Interrupt] aget_state 检查失败，降级为消息检测: %s", exc)
                            # 降级：用消息内容检测
                            messages = result.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                    dangerous_calls = [
                                        tc for tc in last_msg.tool_calls
                                        if tc["name"] in {"write_file", "edit_file", "run_command"}
                                    ]
                                    if dangerous_calls:
                                        has_interrupt = True
                                        current_input = Command(resume="approve")
                    if not has_interrupt:
                        break
                return result

            if loop and loop.is_running():
                # 已在事件循环中（不应出现在 REPL，但防御性处理）
                # 使用新线程运行 asyncio.run()，避免嵌套事件循环
                # 同时在新线程中恢复 ContextVar 快照
                from agent_by_langgraph.context_var_manager import restore as _restore_ctx
                ctx_snap = self._ctx_snapshot

                def _run_in_thread():
                    _restore_ctx(ctx_snap)
                    return asyncio.run(_invoke_with_checkpointer())

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(_run_in_thread).result()
            else:
                result = asyncio.run(_invoke_with_checkpointer())
            print()  # 流式结束换行

            messages = result["messages"]
            final_msg: AIMessage | None = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    if msg.content or msg.additional_kwargs:
                        final_msg = msg
                        break

            collector_msg = self._reasoning_collector.last
            if collector_msg is not None:
                final_msg = collector_msg

            reply = final_msg.content if final_msg else ""
            if isinstance(reply, list):
                reply = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in reply
                )

            self.memory_store.append_history("user", user_input)
            if final_msg is not None and (reply or final_msg.additional_kwargs):
                self.memory_store.append_history(
                    "assistant",
                    reply,
                    additional_kwargs=final_msg.additional_kwargs or None,
                )
            elif reply:
                self.memory_store.append_history("assistant", reply)

            self._maybe_compact()

    def _maybe_compact(self) -> None:
        if self.token_tracker.should_compact(max_context=200_000, threshold=0.5):
            self.compactor.compact_store()

    async def _ensure_checkpointer(self) -> None:
        """延迟初始化 AsyncSqliteSaver，确保在正确的事件循环中创建。

        AsyncSqliteSaver 内部使用 aiosqlite，其工作线程与创建时的事件循环绑定。
        如果在 __init__ 中创建（不同事件循环），后续 ainvoke 会失败。
        因此延迟到首次 ainvoke 时，在正确的事件循环中初始化。

        注意：此方法会重新编译图（注入 checkpointer），调用方应在
        _ensure_checkpointer() 后重新获取 graph._lg_llm_callbacks。
        """
        if self._checkpointer_initialized:
            return
        if not hasattr(self, '_checkpointer_db_path') or not self._checkpointer_db_path:
            return

        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            ctx = AsyncSqliteSaver.from_conn_string(self._checkpointer_db_path)
            checkpointer = await ctx.__aenter__()
            self._checkpointer_ctx = ctx

            # 保存旧图的回调列表，确保重新编译后不丢失
            old_callbacks = list(getattr(self.graph, '_lg_llm_callbacks', []))

            # 重新编译图以注入 checkpointer
            self.graph = create_agent_graph(
                self.llm, self.tools, self._system_prompt,
                llm_callbacks=[TokenTrackerCallback(self.token_tracker, self.model)],
                checkpointer=checkpointer,
            )

            # 合并旧回调到新图（避免丢失 run() 中添加的 per-invoke 回调引用）
            new_callbacks = list(getattr(self.graph, '_lg_llm_callbacks', []))
            # 保留旧图中非 TokenTrackerCallback 的回调（如 ReasoningCollector、StreamHandler）
            for cb in old_callbacks:
                if not isinstance(cb, TokenTrackerCallback) and cb not in new_callbacks:
                    new_callbacks.append(cb)
            self.graph._lg_llm_callbacks = new_callbacks

            self._checkpointer_initialized = True
            logger.info("[Checkpointer] AsyncSqliteSaver 延迟初始化成功: %s", self._checkpointer_db_path)
        except Exception as exc:
            logger.warning("[Checkpointer] 延迟初始化失败: %s，checkpointer 降级为 None", exc)
            self._checkpointer_initialized = True  # 避免反复尝试

    async def arun_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """异步流式运行 Agent，使用 astream_events 逐 token 输出。

        适用于 Web API（SSE）等异步场景，替代 callback 式流式输出。
        每次 yield 一个 token 片段，调用方负责拼接和发送。

        Checkpointer 利用策略与 run() 一致：
        - 首轮传入完整上下文，后续轮次只传 HumanMessage。

        Args:
            user_input: 用户输入文本

        Yields:
            LLM 输出的 token 片段字符串
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        _will_have_checkpointer = (
            self.graph.checkpointer is not None
            or getattr(self, '_checkpointer_db_path', None) is not None
        )
        with self._invoke_lock:
            is_first_turn = self._first_turn
            if is_first_turn and _will_have_checkpointer:
                self._first_turn = False

        if is_first_turn or not _will_have_checkpointer:
            initial_messages = [SystemMessage(content=self._system_prompt)]
            initial_messages.extend(self.memory_store.messages)
            # 标记用户原始请求为 milestone（ContextView 会优先保留）
            user_msg = HumanMessage(content=user_input)
            user_msg.metadata = {"milestone": True}
            initial_messages.append(user_msg)
            input_state = {"messages": initial_messages}
        else:
            user_msg = HumanMessage(content=user_input)
            user_msg.metadata = {"milestone": True}
            input_state = {"messages": [user_msg]}

        config: RunnableConfig = {
            "callbacks": list(getattr(self.graph, '_lg_llm_callbacks', [])),
            "recursion_limit": self.max_iterations * 2 + 5,
            "configurable": {
                "thread_id": self.user_id or "default",
                "__has_checkpointer__": _will_have_checkpointer,
            },
        }

        full_reply = ""
        # 收集最后一次 LLM 调用的完整 AIMessage（含 reasoning_content）
        # DeepSeek thinking mode 必须在后续请求中原样回传 reasoning_content
        last_ai_msg: AIMessage | None = None

        # 延迟初始化 checkpointer（确保在正确的事件循环中）
        await self._ensure_checkpointer()
        # _ensure_checkpointer 可能重新编译图，需更新 config 中的回调
        config["callbacks"] = list(getattr(self.graph, '_lg_llm_callbacks', []))
        config["configurable"]["__has_checkpointer__"] = self.graph.checkpointer is not None

        # 使用 astream_events 流式输出，自动处理 interrupt 恢复
        from langgraph.types import Command
        current_input = input_state
        max_interrupt_retries = 30
        for _interrupt_retry in range(max_interrupt_retries):
            found_interrupt = False
            async for event in self.graph.astream_events(
                current_input,
                config=config,
                version="v2",
            ):
                kind = event.get("event")
                # on_chat_model_stream: LLM 逐 token 输出
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        full_reply += token
                        yield token
                # on_chat_model_end: 收集完整 AIMessage（含 reasoning_content）
                elif kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, AIMessage):
                        last_ai_msg = output

            # 检查是否有 pending interrupt 需要恢复
            # interrupt 后 astream_events 结束，需要检查最终状态
            if _will_have_checkpointer:
                # 获取当前 state 检查是否有未处理的 interrupt
                try:
                    state_snapshot = await self.graph.aget_state(config)
                    if state_snapshot and state_snapshot.next:
                        # 有待执行的节点，可能是 interrupt 暂停
                        # 检查 tasks 中是否有 interrupt 信息
                        tasks = getattr(state_snapshot, "tasks", [])
                        for task in tasks:
                            if hasattr(task, "interrupts") and task.interrupts:
                                found_interrupt = True
                                logger.info("[Interrupt] arun_stream 自动批准: %s", task.interrupts)
                                current_input = Command(resume="approve")
                                break
                except Exception:
                    pass

            if not found_interrupt:
                break

        # 持久化到 MemoryStore
        self.memory_store.append_history("user", user_input)
        if last_ai_msg is not None and (full_reply or last_ai_msg.additional_kwargs):
            self.memory_store.append_history(
                "assistant",
                full_reply,
                additional_kwargs=last_ai_msg.additional_kwargs or None,
            )
        elif full_reply:
            self.memory_store.append_history("assistant", full_reply)

        self._maybe_compact()
