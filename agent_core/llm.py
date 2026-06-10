"""LLM 封装：DeepSeekChatOpenAI + create_deepseek_llm 工厂函数。

从 agent.lc_agent 提取，供 agent_by_langgraph 和其他模块共用。
"""
from __future__ import annotations

import os

import openai
from openai import OpenAI as OpenAIClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


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
