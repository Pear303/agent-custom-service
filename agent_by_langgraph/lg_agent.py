"""LangGraph Agent —— 基于 StateGraph 的智能代理，替代 LCAgent + AgentExecutor。"""
from __future__ import annotations

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
        self._turn_lock = threading.Lock()  # 保护 _first_turn 的并发读写

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

        # 保存 ContextVar 快照，供 _invoke_lg_in_thread 在新线程中恢复
        # asyncio.to_thread 不会自动传播 ContextVar，需要显式快照
        from agent.lc_tools import (
            _ctx_workspace, _ctx_skills_loader, _ctx_todo_store,
            _ctx_subagent_registry, _ctx_llm_ref, _ctx_user_id, _ctx_ticket_id,
        )
        self._ctx_snapshot = {
            "workspace": _ctx_workspace.get(),
            "skills_loader": _ctx_skills_loader.get(),
            "todo_store": _ctx_todo_store.get(),
            "subagent_registry": _ctx_subagent_registry.get(),
            "llm_ref": _ctx_llm_ref.get(),
            "user_id": _ctx_user_id.get(),
            "ticket_id": _ctx_ticket_id.get(),
        }

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
        ctx = getattr(self, "_checkpointer_ctx", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
                logger.info("[Checkpointer] SQLite 连接已关闭")
            except Exception as exc:
                logger.warning("[Checkpointer] 关闭 SQLite 连接时出错: %s", exc)
            finally:
                self._checkpointer_ctx = None

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
        """
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            if self.user_id:
                db_path = self.root / "data" / "users" / self.user_id / "checkpoints.db"
            else:
                db_path = self.root / "data" / "checkpoints.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # from_conn_string 返回 context manager，手动进入以保持长生命周期
            ctx = SqliteSaver.from_conn_string(str(db_path))
            checkpointer = ctx.__enter__()
            # 保存 context manager 引用，避免被 GC 回收导致数据库连接关闭
            self._checkpointer_ctx = ctx

            logger.info("[Checkpointer] SqliteSaver 初始化成功: %s", db_path)
            return checkpointer
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
            config: RunnableConfig = {
                "callbacks": all_callbacks,
                "recursion_limit": self.max_iterations * 2 + 5,
                "configurable": {
                    "thread_id": self.user_id or "default",
                },
            }

            has_checkpointer = self.graph.checkpointer is not None
            with self._turn_lock:
                is_first_turn = self._first_turn
                if is_first_turn and has_checkpointer:
                    self._first_turn = False

            if is_first_turn or not has_checkpointer:
                # 首轮或无 checkpointer：传入完整上下文（system + chat_history + user input）
                initial_messages = [SystemMessage(content=self._system_prompt)]
                initial_messages.extend(self.memory_store.messages)
                initial_messages.append(HumanMessage(content=user_input))
                input_state = {"messages": initial_messages}
            else:
                # 后续轮次：只传入新的 HumanMessage，
                # checkpointer 自动恢复之前的 state
                input_state = {"messages": [HumanMessage(content=user_input)]}

            # 节点函数为 async def，需要用 asyncio.run 包装 ainvoke
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 已在事件循环中（不应出现在 REPL，但防御性处理）
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self.graph.ainvoke(input_state, config=config)
                    ).result()
            else:
                result = asyncio.run(self.graph.ainvoke(input_state, config=config))
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

        has_checkpointer = self.graph.checkpointer is not None
        with self._turn_lock:
            is_first_turn = self._first_turn
            if is_first_turn and has_checkpointer:
                self._first_turn = False

        if is_first_turn or not has_checkpointer:
            initial_messages = [SystemMessage(content=self._system_prompt)]
            initial_messages.extend(self.memory_store.messages)
            initial_messages.append(HumanMessage(content=user_input))
            input_state = {"messages": initial_messages}
        else:
            input_state = {"messages": [HumanMessage(content=user_input)]}

        config: RunnableConfig = {
            "callbacks": list(getattr(self.graph, '_lg_llm_callbacks', [])),
            "recursion_limit": self.max_iterations * 2 + 5,
            "configurable": {"thread_id": self.user_id or "default"},
        }

        full_reply = ""
        # 收集最后一次 LLM 调用的完整 AIMessage（含 reasoning_content）
        # DeepSeek thinking mode 必须在后续请求中原样回传 reasoning_content
        last_ai_msg: AIMessage | None = None

        async for event in self.graph.astream_events(
            input_state,
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
