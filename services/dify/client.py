import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from .config import DifyConfig

logger = logging.getLogger(__name__)


class DifyChatflowClient:
    """Dify Chatflow API 客户端。

    用法:
        client = DifyChatflowClient()
        response = await client.chat("你好", user_id="user_001")
        async for chunk in client.chat_stream("你好", user_id="user_001"):
            print(chunk)
    """

    def __init__(self, config: Optional[DifyConfig] = None):
        self.config = config or DifyConfig.from_env()
        # user_id -> conversation_id (语义映射，非纯类型可表达)
        self._sessions: dict[str, str] = {}

    async def chat(
        self,
        query: str,
        user_id: str,
        inputs: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        timeout: float = 60.0,
    ) -> dict:
        merged_inputs = dict(inputs or {})
        merged_inputs.setdefault("user_id", user_id)
        payload = {
            "inputs": merged_inputs,
            "query": query,
            "response_mode": "blocking",
            "user": user_id,
        }

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        inputs: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        timeout: float = 120.0,
    ) -> AsyncGenerator[dict, None]:
        merged_inputs = dict(inputs or {})
        merged_inputs.setdefault("user_id", user_id)
        payload = {
            "inputs": merged_inputs,
            "query": query,
            "response_mode": "streaming",
            "user": user_id,
        }
        if conversation_id or self._sessions.get(user_id):
            payload["conversation_id"] = conversation_id or self._sessions[user_id]

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=False) as client:
            resp = await client.post(
                self._build_url("/v1/chat-messages"),
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if cid := data.get("conversation_id"):
            self._sessions[user_id] = cid

        return data

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        inputs: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        timeout: float = 120.0,
    ) -> AsyncGenerator[dict, None]:
        # 自动注入 user_id 到 inputs，Dify 工作流可通过 {{user_id}} 引用
        merged_inputs = dict(inputs or {})
        merged_inputs.setdefault("user_id", user_id)
        payload = {
            "inputs": merged_inputs,
            "query": query,
            "response_mode": "streaming",
            "user": user_id,
        }
        if conversation_id or self._sessions.get(user_id):
            payload["conversation_id"] = conversation_id or self._sessions[user_id]

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, trust_env=False) as client:
            async with client.stream(
                "POST",
                self._build_url("/v1/chat-messages"),
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Dify SSE 解析失败: %s", data_str[:100])
                        continue

                    if cid := event.get("conversation_id"):
                        self._sessions[user_id] = cid

                    yield event

    def get_conversation_id(self, user_id: str) -> Optional[str]:
        return self._sessions.get(user_id)

    def reset_session(self, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    def _build_url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}{path}"

    @staticmethod
    def _headers() -> dict[str, str]:
        from .config import DifyConfig

        cfg = DifyConfig.from_env()
        return {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
