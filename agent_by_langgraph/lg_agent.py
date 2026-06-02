"""LangGraph Agent —— 基于 StateGraph 的智能代理，替代 LCAgent + AgentExecutor。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
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


class TokenTrackerCallback(BaseCallbackHandler):
    """把 LLM token 用量记录到 TokenTracker。"""
    def __init__(self, tracker: TokenTracker, model_name: str):
        self._tracker = tracker
        self._model = model_name

    def on_llm_end(self, response, **kwargs):
        usage = getattr(response, "llm_output", {}).get("token_usage", {})
        if not usage:
            try:
                usage = response.generations[0][0].message.usage_metadata
            except Exception:
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

        self.graph = create_agent_graph(
            self.llm, self.tools, system_prompt,
            llm_callbacks=[TokenTrackerCallback(self.token_tracker, model)],
        )

    def run(self) -> None:
        """REPL 交互循环。每次输入调用编译后的 StateGraph。"""
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
            config: RunnableConfig = {
                "callbacks": [self._reasoning_collector, stream_handler],
            }

            result = self.graph.invoke(
                {
                    "input": user_input,
                    "chat_history": self.memory_store.messages,
                },
                config=config,
            )
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
