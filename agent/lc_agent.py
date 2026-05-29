"""LangChain Agent —— 基于 LangChain 框架实现的智能代理，替代手搓的 AgentLoop + AgentRunner。

主要功能：
- 集成 DeepSeek LLM
- 管理工具集（文件操作、命令执行、网络搜索等）
- 支持子代理调度
- 自动记忆压缩和 Token 追踪
- 提供 REPL 交互界面
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI as OpenAIClient
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .compactor import Compactor
from .context import ContextBuilder
from .lc_tools import (
    read_file, write_file, edit_file,
    run_command, web_fetch, load_skill,
    glob_tool, grep_tool, update_todos,
    dispatch_subagent,
    set_workspace, set_skills_loader, set_todo_store,
    set_subagent_deps, set_user_id, set_ticket_id,
    _build_workspace,
)
from .memory import MemoryStore
from .telemetry import TokenTracker
from .todo import TodoStore
from .subagents.registry import SubagentRegistry


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类，保留 DeepSeek thinking mode 的 reasoning_content。
    
    LangChain 原生 _create_chat_result / _convert_delta_to_message_chunk 
    只处理 parsed / refusal / tool_calls 扩展字段，丢弃了 DeepSeek 特有的 
    reasoning_content。下一轮对话时缺少此字段会导致 API 报错：
    "The reasoning_content in the thinking mode must be passed back to the API."
    
    本子类在三处拦截：
    1. _create_chat_result — 处理非流式调用，从原始响应注入 reasoning_content
    2. _convert_chunk_to_generation_chunk — 处理流式调用，逐 chunk 注入
    3. _get_request_payload — 将 AIMessage.additional_kwargs 中的 reasoning_content
       输出到 API 请求 dict，解决 LangChain _convert_message_to_dict 不输出它的问题
    """

    # 非流式：从原始响应中提取 reasoning_content
    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        chat_result = super()._create_chat_result(response, generation_info)
        if isinstance(response, openai.BaseModel) and getattr(response, "choices", None):
            message = response.choices[0].message
            # 如果 message 对象有 reasoning_content 属性，且该属性不为空
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                if chat_result.generations:
                    chat_result.generations[0].message.additional_kwargs[
                        "reasoning_content"
                    ] = message.reasoning_content
        return chat_result

    # 流式：逐 chunk 提取 reasoning_content
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is not None:
            choices = (
                chunk.get("choices", [])
                or chunk.get("chunk", {}).get("choices", [])
            )
            if choices:
                delta = choices[0].get("delta") or {}
                rc = delta.get("reasoning_content")
                if rc and isinstance(gen_chunk.message, AIMessageChunk):
                    gen_chunk.message.additional_kwargs["reasoning_content"] = rc
        return gen_chunk

    # 发送时将 additional_kwargs 中的 reasoning_content
    # 写回 API 请求的 dict 中
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        ai_idx = 0
        for msg_dict in payload.get("messages", []):
            if msg_dict.get("role") == "assistant":
                while ai_idx < len(messages):
                    m = messages[ai_idx]
                    ai_idx += 1
                    if isinstance(m, AIMessage):
                        rc = m.additional_kwargs.get("reasoning_content")
                        if rc:
                            msg_dict["reasoning_content"] = rc
                        break
        return payload


def create_deepseek_llm(model: str = "deepseek-v4-flash") -> DeepSeekChatOpenAI:
    """创建连接 DeepSeek API 的 ChatOpenAI 实例。

    Args:
        model: DeepSeek 模型名称，默认为 "deepseek-v4-flash"

    Returns:
        配置好的 DeepSeekChatOpenAI 实例
    """
    load_dotenv()
    return DeepSeekChatOpenAI(
        model=model,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        streaming=True,
    )


class LCAgent:
    """基于 LangChain 的智能代理类，整合了工具调用、记忆管理、上下文构建等功能。
    
    核心组件：
    - LLM: DeepSeek 语言模型
    - Tools: 文件系统、命令执行、网络搜索、技能加载等工具
    - Memory: 长期记忆存储和自动压缩
    - Context: 系统提示词动态构建
    - Telemetry: Token 使用追踪
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
        """初始化 LCAgent。
        
        Args:
            user_id: 用户唯一标识（提供时状态按用户隔离）
            ticket_id: 工单唯一标识（提供时文件保存到对应工单目录）
            root: 项目根目录路径，默认为当前文件的上两级目录
            model: 使用的 LLM 模型名称
            max_iterations: Agent 单次任务的最大迭代次数
            llm: 可选的共享 LLM 实例（工厂模式提供）
            skills_loader: 可选的共享 SkillsLoader 实例（工厂模式提供）
            sub_reg: 可选的共享 SubagentRegistry 实例（工厂模式提供）
        """
        load_dotenv()
        self.root = root or Path(__file__).parent.parent
        self.user_id = user_id
        self.ticket_id = ticket_id
        self.model = model

        # ── 初始化 LLM 客户端（优先使用共享实例）──
        self.llm = llm or create_deepseek_llm(model)
        openai_client = OpenAIClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

        # ── 初始化工具系统 ──
        workspace = _build_workspace(self.root, user_id, ticket_id)
        set_workspace(workspace)
        if user_id:
            set_user_id(user_id)
        if ticket_id:
            set_ticket_id(ticket_id)

        # 加载技能系统（优先使用共享实例）
        from agent.skills import get_skills_loader
        self.skills = skills_loader or get_skills_loader(self.root / "skills")
        set_skills_loader(self.skills)

        # 初始化待办事项存储（按 user_id 隔离）
        self.todo_store = TodoStore(user_id=user_id)
        set_todo_store(self.todo_store)

        # 注册所有可用工具
        self.tools = [
            read_file, write_file, edit_file,
            run_command, web_fetch, load_skill,
            glob_tool, grep_tool, update_todos,
            dispatch_subagent,
        ]

        # ── 配置子代理上下文（优先使用共享实例）──
        self.sub_reg = sub_reg or SubagentRegistry(
            self.root / "templates" / "subagents",
            skills_loader=self.skills,
        )
        set_subagent_deps(
            llm=self.llm,
            registry=self.sub_reg,
        )

        # ── 初始化记忆系统和压缩器（按 user_id 隔离）──
        if user_id:
            self.memory_store = MemoryStore(user_id=user_id)
            token_log_path = Path(__file__).parent.parent / "data" / "users" / user_id / "tokens.jsonl"
        else:
            self.memory_store = MemoryStore(
                memory_dir=self.root / "memory",
                user_file=self.root / "templates" / "USER.md",
            )
            token_log_path = self.root / "memory" / "tokens.jsonl"
        
        self.token_tracker = TokenTracker(log_file=token_log_path)
        self.compactor = Compactor(openai_client, model, self.memory_store)

        # 启动时处理未归档的历史对话
        unarchived = self.memory_store.load_unarchived_history()
        if len(unarchived) >= 2:
            print(f"[Startup: found {len(unarchived)} unarchived turns, compacting...]")
            try:
                self.compactor.compact_startup(unarchived)
            except Exception as exc:
                print(f"[warning] startup compaction failed: {exc}", file=sys.stderr)

        # ── 构建系统提示词 ──
        ctx = ContextBuilder(
            self.root / "templates",
            self.skills,
            memory=self.memory_store,
        )
        system_prompt = ctx.build_system_prompt()

        # ── 创建聊天提示模板 ──
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # ── 创建 Agent 和执行器 ──
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        from langchain_core.callbacks import BaseCallbackHandler

        class TokenCallback(BaseCallbackHandler):
            def __init__(self, tracker: TokenTracker, model_name: str):
                self._tracker = tracker
                self._model = model_name

            def on_llm_end(self, response, **kwargs):
                usage = getattr(response, 'llm_output', {}).get('token_usage', {})
                if not usage:
                    try:
                        usage = response.generations[0][0].message.usage_metadata
                    except Exception:
                        return
                input_tokens = getattr(usage, 'input_tokens', 0) or usage.get('input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0) or usage.get('output_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0) or usage.get('total_tokens', 0)
                self._tracker.record_raw(self._model, input_tokens, output_tokens, total_tokens)

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            callbacks=[TokenCallback(self.token_tracker, model)],
        )
    

    def run(self) -> None:
        """启动 REPL 交互循环。
        
        使用 MemoryStore 作为单一数据源
        每次交互后自动检查是否需要压缩历史
        """
        from langchain_core.callbacks import BaseCallbackHandler

        class ReasoningCollector(BaseCallbackHandler):
            """收集 LLM 调用中产生的完整 AIMessage（含 reasoning_content 等额外字段）。
            
            DeepSeek thinking mode 下 AIMessage.additional_kwargs 包含 reasoning_content，
            后续请求必须原样传回，否则 API 报错。
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

        while True:
            try:
                user_input = input("You🫅 : ")
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            # 流式输出回调：逐 token 打印
            class StreamHandler(BaseCallbackHandler):
                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    print(token, end="", flush=True)

            # 从 MemoryStore 获取已持久化的未归档历史（不含当前用户输入）
            collector = ReasoningCollector()
            stream_handler = StreamHandler()
            result = self.executor.invoke(
                {
                    "input": user_input,
                    "chat_history": self.memory_store.messages,
                },
                {"callbacks": [collector, stream_handler]},
            )

            print()  # 流式结束换行

            # 提取助手回复并保存到记忆
            reply = result["output"]
            self.memory_store.append_history("user", user_input)

            # 优先使用收集到的完整 AIMessage（含 reasoning_content）
            final_msg = collector.last
            if final_msg is not None and final_msg.additional_kwargs:
                self.memory_store.append_history(
                    "assistant", final_msg.content,
                    additional_kwargs=final_msg.additional_kwargs,
                )
            else:
                self.memory_store.append_history("assistant", reply)
            
            # 检查是否需要压缩历史
            self._maybe_compact()

    def _maybe_compact(self) -> None:
        """根据 Token 使用情况判断是否需要压缩历史。
        
        当上下文 token 数超过阈值的 50% 时触发压缩。
        """
        if self.token_tracker.should_compact(max_context=200_000, threshold=0.5):
            self.compactor.compact_store()
            
            
"""
┌─────────────────────────────────────────────────────────────────┐
│ AgentExecutor.invoke({input, chat_history})                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第1步: 准备 Prompt 输入                                          │
│   把 {chat_history} 和 {input} 填入 ChatPromptTemplate           │
│   agent_scratchpad 初始为空                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第2步: 调用 LLM                                                  │
│   agent.invoke({"input": ..., "chat_history": [...],             │
│                 "agent_scratchpad": [...]})                       │
│                                                                  │
│   LLM 返回 → AIMessage                                           │
│     可能包含: tool_calls[]（决定调用工具）                         │
│     或纯文本（最终回答）                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
              有 tool_calls       纯文本回复
                    │                 │
                    ▼                 ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ 第3步: 格式检查   │  │ 第5步: 返回结果   │
        │ 验证 tool_calls   │  │ {"output": "..."} │
        │ 是否合法          │  └──────────────────┘
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ 第4步: 执行工具   │
        │ 按顺序/并发执行    │
        │ 收集 tool_result  │
        │                   │
        │ 追加到             │
        │ agent_scratchpad  │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ 回到第1步(下一轮)  │
        │ agent_scratchpad  │
        │ 包含了上一轮的     │
        │ 工具调用+结果      │
        └──────────────────┘

"""