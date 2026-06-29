import asyncio
import json
import logging
import time
from enum import Enum
from typing import AsyncGenerator, Optional

import httpx

from .config import DifyConfig

logger = logging.getLogger(__name__)


def _get_settings():
    """延迟加载 Settings，避免循环导入（api.core.__init__ → lifespan → dify → core.config）。"""
    from ...core.config import settings
    return settings


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """熔断器：保护下游服务，避免雪崩。

    状态机：
    - CLOSED（正常）：请求正常通过，失败时累计计数
    - OPEN（熔断）：直接拒绝请求，等待冷却期后进入半开
    - HALF_OPEN（半开）：放行一个探测请求，成功则恢复 CLOSED，失败则回到 OPEN

    熔断触发条件：在 window_seconds 窗口内，失败次数 >= failure_threshold
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self._failure_times: list[float] = []
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("熔断器进入 HALF_OPEN 状态，放行探测请求")
                    return True
                return False
            return True

    async def record_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self._failure_times.clear()
                logger.info("熔断器恢复 CLOSED 状态")

    async def record_failure(self):
        async with self._lock:
            now = time.monotonic()
            self._failure_times.append(now)
            self._last_failure_time = now
            self._failure_times = [
                t for t in self._failure_times if now - t < self.window_seconds
            ]
            if len(self._failure_times) >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        "熔断器进入 OPEN 状态（%d 秒内 %d 次失败）",
                        self.window_seconds,
                        len(self._failure_times),
                    )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("探测请求失败，熔断器回到 OPEN 状态")


class DifyChatflowClient:
    """Dify Chatflow API 客户端，内置指数退避重试 + 熔断器 + 连接池复用。

    重试策略：指数退避 + 抖动
      第 1 次重试等待 ~1 秒
      第 2 次重试等待 ~2 秒
      第 3 次重试等待 ~4 秒
      ...

    熔断策略：
      60 秒内 5 次失败 → 熔断 30 秒
      30 秒后放行 1 个探测请求，成功则恢复

    连接池：
      复用 httpx.AsyncClient 实例，避免每次请求创建/销毁 TCP 连接。
      通过 close() 方法在应用关闭时清理。
    """

    def __init__(self, config: Optional[DifyConfig] = None):
        self.config = config or DifyConfig.from_env()
        s = _get_settings()
        self._circuit = CircuitBreaker(
            failure_threshold=s.dify_circuit_failure_threshold,
            window_seconds=s.dify_circuit_window_seconds,
            recovery_timeout=s.dify_circuit_recovery_timeout,
        )
        self._max_retries = s.dify_max_retries
        self._base_delay = s.dify_base_delay
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self, timeout: float = 60.0) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=10.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
                follow_redirects=True,
                trust_env=False,
            )
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _retry_request(self, request_fn, label: str = ""):
        """带指数退避重试的请求执行器。

        退避公式：delay = base_delay * (2 ^ attempt) + random_jitter
        - attempt=0: ~1s
        - attempt=1: ~2s
        - attempt=2: ~4s

        仅对可重试的错误（网络超时、5xx）进行重试，
        4xx 客户端错误直接抛出。
        """
        last_exc = None
        for attempt in range(self._max_retries + 1):
            if not await self._circuit.allow_request():
                raise RuntimeError(
                    "Dify 服务熔断中，请稍后重试"
                )
            try:
                result = await request_fn()
                await self._circuit.record_success()
                return result
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "Dify 请求超时 [%s] (第 %d/%d 次): %s",
                    label, attempt + 1, self._max_retries + 1, exc,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    await self._circuit.record_failure()
                    raise
                last_exc = exc
                logger.warning(
                    "Dify 服务端错误 [%s] (第 %d/%d 次): %s",
                    label, attempt + 1, self._max_retries + 1, exc,
                )
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
                last_exc = exc
                logger.warning(
                    "Dify 连接错误 [%s] (第 %d/%d 次): %s",
                    label, attempt + 1, self._max_retries + 1, exc,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Dify 未知错误 [%s] (第 %d/%d 次): %s",
                    label, attempt + 1, self._max_retries + 1, exc,
                )

            await self._circuit.record_failure()

            if attempt < self._max_retries:
                import random
                delay = self._base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.info("等待 %.1f 秒后重试...", delay)
                await asyncio.sleep(delay)

        raise last_exc

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
        if conversation_id:
            payload["conversation_id"] = conversation_id

        client = await self._get_http_client(timeout)

        async def _do_request():
            resp = await client.post(
                self._build_url("/v1/chat-messages"),
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

        data = await self._retry_request(_do_request, label="chat-blocking")

        return data

    async def chat_stream(
        self,
        query: str,
        user_id: str,
        inputs: Optional[dict] = None,
        conversation_id: Optional[str] = None,
        timeout: float = 120.0,
    ) -> AsyncGenerator[dict, None]:
        if not await self._circuit.allow_request():
            raise RuntimeError("Dify 服务熔断中，请稍后重试")

        merged_inputs = dict(inputs or {})
        merged_inputs.setdefault("user_id", user_id)
        payload = {
            "inputs": merged_inputs,
            "query": query,
            "response_mode": "streaming",
            "user": user_id,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        client = await self._get_http_client(timeout)

        try:
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
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning("Dify SSE 解析失败: %s", data_str[:100])
                        continue

                    yield event

            await self._circuit.record_success()
        except Exception as exc:
            await self._circuit.record_failure()
            raise

    def _build_url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        return f"{base}{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
