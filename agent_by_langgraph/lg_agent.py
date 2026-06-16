"""LangGraph Agent —— 基于 StateGraph 的智能代理，替代 LCAgent + AgentExecutor。"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, AsyncGenerator

# 注意：Windows 控制台 UTF-8 编码修复已统一到 agent_lg.py 入口文件，
# 此处不再重复处理，避免多次调用 chcp 65001

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from openai import OpenAI as OpenAIClient

from agent_core.compactor import Compactor
from agent_core.context import ContextBuilder
from agent_core.llm import DeepSeekChatOpenAI, create_deepseek_llm
from agent_core.tools import (
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
from agent_core.memory import MemoryStore
from agent_core.subagents.registry import SubagentRegistry
from agent_core.telemetry import TokenTracker
from agent_core.todo import TodoStore
from agent_by_langgraph.lg_graph import create_agent_graph
from agent_by_langgraph.lg_tools import dispatch_subagent_lg
from agent_by_langgraph.level_router import LevelRouter, TaskLevel

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


class LGAgent:
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

        # LangSmith 追踪初始化：设置环境变量后 LangChain/LangGraph 自动上报
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            if not os.getenv("LANGCHAIN_API_KEY"):
                logger.warning("[LangSmith] LANGCHAIN_TRACING_V2=true 但未设置 LANGCHAIN_API_KEY，追踪将不生效")
            else:
                os.environ.setdefault("LANGCHAIN_PROJECT", "agent-custom-service")
                logger.info(
                    "[LangSmith] 追踪已启用, project=%s",
                    os.getenv("LANGCHAIN_PROJECT", "agent-custom-service"),
                )

        self.llm = llm or create_deepseek_llm(model)

        workspace = _build_workspace(self.root, user_id, ticket_id)
        set_workspace(workspace)
        self._workspace = workspace  # D2: 保存工作目录引用，供外部查询
        if user_id:
            set_user_id(user_id)
        if ticket_id:
            set_ticket_id(ticket_id)

        from agent_core.skills import get_skills_loader
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

        # D1: 注入当前工作目录信息 + 工具约束（合并精简，减少重复）
        workspace = _build_workspace(self.root, user_id, ticket_id)
        cwd_and_constraints = (
            f"\n\n---\n\n# 工作目录与工具约束\n\n"
            f"## 工作目录\n"
            f"CWD: `{workspace}`\n"
            f"- 文件操作路径相对于此目录解析（如 `read_file('src/app.py')` → `{workspace}/src/app.py`）\n"
            f"- run_command 自动在 CWD 下执行，无需手动 cd\n\n"
            f"## 禁止行为\n"
            f"- 禁止使用 `pwd`/`cd`/`dir`/`ls` 探测目录 — 你已知 CWD，用 `glob_tool` 查看内容\n"
            f"- 禁止使用 `copy`/`move`/`mkdir`/`del`/`rm` 等 shell 命令操作文件 — 用专用工具\n"
            f"- 禁止使用绝对路径或 `../` 跳出工作目录\n"
            f"- 禁止通过 `python -c \"open(...,'w')...\"` 在 run_command 中写文件 — 用 write_file/edit_file\n\n"
            f"## 工具使用要点\n"
            f"- **update_todos**: 只在关键里程碑更新，不要每改一个文件就更新\n"
            f"- **read_file**: 优先用 offset/limit 读取大文件关键部分\n"
            f"- **dispatch_subagent_lg**: gather 阶段只能派遣只读子代理\n"
            f"- **写后必审**: write_file 后用 read_file 检查内容，不要用 run_command 重写同一文件\n"
            f"- **跑后必修**: run_command 失败后用 edit_file 精准修复，不要重写整个文件\n"
            f"- **复杂任务分治**: 3+ 文件时考虑派遣子代理并行处理\n"
            f"- **gather 不跳过**: 先用 glob_tool/read_file 了解上下文再写代码"
        )
        system_prompt += cwd_and_constraints

        # 初始化 LangGraph Checkpointer（SQLite 持久化）
        # 启用后支持：状态持久化、时间旅行调试、断点续跑
        checkpointer = self._init_checkpointer()

        self.graph = create_agent_graph(
            self.llm, self.tools, system_prompt,
            llm_callbacks=[TokenTrackerCallback(self.token_tracker, model)],
            checkpointer=checkpointer,
        )
        self._system_prompt = system_prompt

        # 级别路由器：根据用户输入判断任务级别
        self._level_router = LevelRouter(llm=self.llm)
        self._current_level: TaskLevel | None = None

    @property
    def will_have_checkpointer(self) -> bool:
        """判断当前 Agent 是否将拥有（或已拥有）checkpointer。

        统一判断逻辑，避免 run() 和 arun_stream() 中重复推断。
        覆盖三种场景：
        1. checkpointer 已初始化：_checkpointer_initialized 为 True
        2. checkpointer 已注入：graph.checkpointer is not None
        3. checkpointer 延迟初始化：_checkpointer_db_path 存在但尚未初始化

        ⚠️ 重要：调用方应使用此属性或 checkpointer_ready 判断 checkpointer 状态，
        而非直接检查 graph.checkpointer is not None（延迟初始化期间为 None）。
        直接检查 graph.checkpointer 会在 _ensure_checkpointer() 调用前误判为无 checkpointer。
        """
        return (
            getattr(self, '_checkpointer_initialized', False)
            or self.graph.checkpointer is not None
            or getattr(self, '_checkpointer_db_path', None) is not None
        )

    def _ensure_repl_loop(self):
        """确保 REPL 持久事件循环已创建并就绪（线程安全）。

        持久事件循环用于在 REPL 的同步上下文中执行异步 ainvoke，
        避免 asyncio.run() 每次创建新事件循环导致 aiosqlite 连接失效。
        """
        import asyncio
        import threading

        if hasattr(self, '_repl_loop') and self._repl_loop is not None and not self._repl_loop.is_closed():
            return  # 已创建且运行中

        with self._invoke_lock:
            # 双重检查
            if hasattr(self, '_repl_loop') and self._repl_loop is not None and not self._repl_loop.is_closed():
                return

            self._repl_loop_ready = threading.Event()
            self._repl_loop = asyncio.new_event_loop()

            def _loop_runner(loop, ready_event):
                asyncio.set_event_loop(loop)
                ready_event.set()  # 通知主线程事件循环已就绪
                loop.run_forever()

            self._repl_thread = threading.Thread(
                target=_loop_runner,
                args=(self._repl_loop, self._repl_loop_ready),
                daemon=True,
                name="lg-agent-repl-loop",
            )
            self._repl_thread.start()
            # 等待事件循环就绪（最多 5 秒）
            if not self._repl_loop_ready.wait(timeout=5):
                raise RuntimeError("REPL 持久事件循环启动超时")

    def close(self) -> None:
        """释放资源：关闭 Checkpointer 的 SQLite 连接。

        应在 agent 不再使用时调用（如 LRU 淘汰、会话结束），
        确保 SQLite 连接正确关闭，避免资源泄漏。
        """
        try:
            self._close_checkpointer_ctx("_checkpointer_ctx", "主 Checkpointer")
            # 子代理已改用 MemorySaver，无需清理 SQLite 缓存
        except (ImportError, AttributeError):
            # Python 关闭期间 sys.meta_path 可能为 None，import 会失败；
            # 属性可能已被 GC 回收。静默跳过即可。
            pass

        # 关闭 REPL 持久事件循环
        if hasattr(self, '_repl_loop') and self._repl_loop is not None and not self._repl_loop.is_closed():
            try:
                self._repl_loop.call_soon_threadsafe(self._repl_loop.stop)
                if hasattr(self, '_repl_thread') and self._repl_thread.is_alive():
                    self._repl_thread.join(timeout=5)
                self._repl_loop.close()
            except RuntimeError:
                pass  # 事件循环可能正在运行中，忽略关闭错误

    @property
    def checkpointer_ready(self) -> bool:
        """D3: Checkpointer 是否已初始化完成。

        延迟初始化期间 graph.checkpointer 为 None，
        但 _ensure_checkpointer 尚未被调用。
        调用方应检查此属性而非直接检查 graph.checkpointer。
        """
        return getattr(self, '_checkpointer_initialized', False)

    @staticmethod
    def _close_checkpointer_ctx_obj(ctx, label: str) -> None:
        """可靠关闭一个 AsyncSqliteSaver 的 async context manager。

        优先使用 asyncio 关闭；若 Python 解释器关闭阶段 asyncio 不可用，
        则回退到直接访问底层 sqlite3 连接进行同步关闭。
        """
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
        except (ImportError, AttributeError):
            # Python 关闭期间 asyncio 可能已被 GC 回收（sys.modules 中为 None）
            # 回退到同步关闭：直接访问底层 sqlite3 连接
            try:
                saver = getattr(ctx, '_obj', None) or getattr(ctx, 'obj', None)
                if saver is None:
                    # 尝试从 __wrapped__ 或其他包装属性获取
                    for attr in ('_obj', 'obj', '__wrapped__', 'saver'):
                        saver = getattr(ctx, attr, None)
                        if saver is not None:
                            break
                if saver is not None:
                    conn = getattr(saver, 'conn', None) or getattr(saver, '_conn', None)
                    if conn is not None:
                        conn.close()
                        logger.info("[Checkpointer] %s SQLite 连接已同步关闭（回退）", label)
                        return
                logger.debug("[Checkpointer] %s 无法获取底层连接，跳过关闭", label)
            except Exception:
                pass
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
            self._checkpointer_initialized = False
            self._checkpointer_db_path = None
            logger.warning(
                "[Checkpointer] langgraph-checkpoint-sqlite 未安装，"
                "状态持久化/时间旅行/断点续跑不可用。"
                "请运行: pip install langgraph-checkpoint-sqlite>=2.0.0"
            )
            return None
        except Exception as exc:
            self._checkpointer_initialized = False
            self._checkpointer_db_path = None
            logger.warning("[Checkpointer] 初始化失败: %s，checkpointer 降级为 None", exc)
            return None

    def run(self) -> None:
        """REPL 交互循环。每次输入调用编译后的 StateGraph。

        Checkpointer 利用策略：
        - 首轮（_first_turn=True）：传入完整消息 [SystemMessage, ...chat_history, HumanMessage]
        - 后续轮次（_first_turn=False）：只传入 [HumanMessage]，
          checkpointer 自动恢复之前的 state（含 SystemMessage + 历史消息），
          add_messages reducer 将新 HumanMessage 追加到已有消息序列末尾。

        级别路由：
        - 首轮输入时，LevelRouter 判断任务级别
        - 根据级别调整系统提示词（追加级别约束）
        - 未实现的级别（5/6）会给出提示

        注意：由于节点函数已改为 async def，需要用 asyncio.run() 包装 ainvoke。
        """
        # 预初始化 checkpointer：延迟到首次 ainvoke 时在同一个 asyncio.run() 中完成。
        # 不在此处调用 asyncio.run()，因为：
        # 1. AsyncSqliteSaver 内部的 aiosqlite 连接线程与创建时的事件循环绑定
        # 2. asyncio.run() 每次创建新事件循环，第二次调用时旧连接线程已死
        # 3. 会导致 RuntimeError: threads can only be started once
        # 改为在 _invoke_with_checkpointer 内部首次调用时初始化，确保同一事件循环。

        # Windows 终端 emoji 兼容：如果 stdout 编码不支持 emoji，替换提示符
        _repl_prompt = "You🫅 : "
        try:
            _repl_prompt.encode(sys.stdout.encoding or "utf-8")
        except UnicodeEncodeError:
            _repl_prompt = "You> "

        while True:
            try:
                user_input = input(_repl_prompt)
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input.strip():
                continue

            # ── 级别路由（仅首轮） ──────────────────────────────
            if self._first_turn and self._current_level is None:
                level_config = self._level_router.route(user_input)
                self._current_level = level_config.level

                if not LevelRouter.is_implemented(level_config.level):
                    print(f"\n⚠️  {LevelRouter.get_unimplemented_message(level_config.level)}")
                    print(f"   当前支持的级别：1-文章撰写 | 2-静态网页 | 3-自动化脚本 | 4-网站开发")
                    print(f"   请重新描述您的需求，或选择已支持的级别。\n")
                    self._current_level = None
                    continue

                # 追加级别约束到系统提示词
                if level_config.extra_prompt:
                    self._system_prompt += level_config.extra_prompt

                # 调整最大迭代数
                if level_config.max_iterations != self.max_iterations:
                    self.max_iterations = level_config.max_iterations

                print(f"[级别路由] → 级别 {level_config.level}: {level_config.label}")

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
            _will_have_checkpointer = self.will_have_checkpointer
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

            # 使用持久事件循环执行 ainvoke，避免 AsyncSqliteSaver 的 aiosqlite
            # 连接线程因事件循环销毁而失效（RuntimeError: threads can only be started once）。
            # 首次创建事件循环后保持运行，后续 REPL 轮次复用同一循环。
            self._ensure_repl_loop()

            import asyncio
            from agent_by_langgraph.context_var_manager import restore as _restore_ctx
            ctx_snap = self._ctx_snapshot

            async def _invoke_with_checkpointer():
                """执行 ainvoke，自动处理 interrupt 恢复。

                interrupt() 暂停图后 ainvoke 返回当前 state，
                需要用 Command(resume="approve") 恢复执行。
                此循环自动批准所有 interrupt，直到图正常结束。
                """
                # 在持久事件循环中恢复 ContextVar
                _restore_ctx(ctx_snap)
                from langgraph.types import Command
                await self._ensure_checkpointer()
                # _ensure_checkpointer 可能重新编译图，需更新 config 中的回调
                config["callbacks"] = list(getattr(self.graph, '_lg_llm_callbacks', []))
                config["callbacks"].extend([self._reasoning_collector, stream_handler])
                config["configurable"]["__has_checkpointer__"] = self.checkpointer_ready
                current_input = input_state
                max_interrupt_retries = 30
                _TASK_TIMEOUT = 180.0  # 单次任务总超时（秒）
                for _interrupt_idx in range(max_interrupt_retries):
                    try:
                        result = await asyncio.wait_for(
                            self.graph.ainvoke(current_input, config=config),
                            timeout=_TASK_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[超时] 任务执行超过 %.0f 秒，强制停止", _TASK_TIMEOUT)
                        result = {"messages": [AIMessage(content="[超时] 任务执行时间过长，已自动停止。请简化任务或分步执行。")]}
                        break
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

            # 在持久事件循环中提交协程并等待结果
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(_invoke_with_checkpointer(), self._repl_loop)
            result = future.result(timeout=300)  # 总超时 5 分钟
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
        if getattr(self, '_checkpointer_failed', False):
            return
        if not hasattr(self, '_checkpointer_db_path') or not self._checkpointer_db_path:
            return

        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            ctx = AsyncSqliteSaver.from_conn_string(self._checkpointer_db_path)
            checkpointer = await ctx.__aenter__()
            self._checkpointer_ctx = ctx

            # 启用 WAL 模式：写操作不阻塞读，高并发下性能显著提升
            try:
                db = getattr(checkpointer, 'db', None) or getattr(checkpointer, '_db', None)
                if db is not None:
                    await db.execute("PRAGMA journal_mode=WAL")
                    await db.execute("PRAGMA busy_timeout=5000")
                    logger.info("[Checkpointer] WAL 模式已启用: %s", self._checkpointer_db_path)
                else:
                    logger.info("[Checkpointer] 无法获取数据库连接，跳过 WAL 模式设置")
            except Exception as wal_exc:
                logger.info("[Checkpointer] WAL 模式设置失败（不影响功能）: %s", wal_exc)

            # 重新编译图以注入 checkpointer
            # 新图已自带 TokenTrackerCallback，无需合并旧的
            # per-invoke 回调（StreamHandler, ReasoningCollector）在每轮 ainvoke 时
            # 由 _invoke_with_checkpointer() 重新设置到 config["callbacks"]，
            # 不应存储到 _lg_llm_callbacks 中（避免过期引用和内存泄漏）
            self.graph = create_agent_graph(
                self.llm, self.tools, self._system_prompt,
                llm_callbacks=[TokenTrackerCallback(self.token_tracker, self.model)],
                checkpointer=checkpointer,
            )

            self._checkpointer_initialized = True
            logger.info("[Checkpointer] AsyncSqliteSaver 延迟初始化成功: %s", self._checkpointer_db_path)
        except Exception as exc:
            logger.warning("[Checkpointer] 延迟初始化失败: %s，checkpointer 降级为 None", exc)
            self._checkpointer_initialized = False
            self._checkpointer_failed = True  # 标记失败，避免反复尝试
            self._checkpointer_db_path = None

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

        _will_have_checkpointer = self.will_have_checkpointer
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
            "recursion_limit": self.max_iterations * 4 + 10,
            "configurable": {
                "thread_id": self.user_id or "default",
                "__has_checkpointer__": _will_have_checkpointer,
            },
        }

        full_reply = ""
        # 收集最后一次 LLM 调用的完整 AIMessage（含 reasoning_content）
        # DeepSeek thinking mode 必须在后续请求中原样回传 reasoning_content
        last_ai_msg: AIMessage | None = None

        # 检查 checkpointer 是否在另一个事件循环中初始化
        # 如果是，需要在新事件循环中重新初始化（关闭旧的，创建新的）
        if self.checkpointer_ready:
            import asyncio
            current_loop = asyncio.get_running_loop()
            repl_loop = getattr(self, '_repl_loop', None)
            if repl_loop is not None and repl_loop is not current_loop:
                # checkpointer 在持久事件循环中初始化，但当前在不同事件循环
                # 需要关闭旧 checkpointer 并重新初始化
                logger.warning(
                    "[arun_stream] 事件循环不匹配，重新初始化 checkpointer"
                )
                self._close_checkpointer_ctx("_checkpointer_ctx", "主 Checkpointer")
                self._checkpointer_initialized = False
                self._checkpointer_failed = False

        # 延迟初始化 checkpointer（确保在正确的事件循环中）
        await self._ensure_checkpointer()
        # _ensure_checkpointer 可能重新编译图，需更新 config 中的回调
        config["callbacks"] = list(getattr(self.graph, '_lg_llm_callbacks', []))
        config["configurable"]["__has_checkpointer__"] = self.checkpointer_ready

        # 使用 astream_events 流式输出，自动处理 interrupt 恢复
        from langgraph.types import Command
        import time as _time
        current_input = input_state
        max_interrupt_retries = 30
        _TASK_TIMEOUT = 180.0  # 单次任务总超时（秒）
        _start_time = _time.monotonic()
        for _interrupt_retry in range(max_interrupt_retries):
            # 超时检查
            if _time.monotonic() - _start_time > _TASK_TIMEOUT:
                logger.warning("[超时] arun_stream 任务执行超过 %.0f 秒，强制停止", _TASK_TIMEOUT)
                yield "\n\n[超时] 任务执行时间过长，已自动停止。请简化任务或分步执行。"
                break

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
                # on_tool_start: 工具开始执行，输出进度提示（不计入 full_reply）
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    if tool_name:
                        yield f"\n[执行: {tool_name}]\n"

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
