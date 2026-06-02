"""简易令牌桶限流中间件"""
from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于 IP 的令牌桶限流。

    每个客户端 IP 独立计数，超出限制返回 429。
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst: int = 10,
    ):
        super().__init__(app)
        self.rate = requests_per_minute / 60.0
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (float(burst), time.monotonic()))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        tokens, last_time = self._buckets[client_ip]
        now = time.monotonic()
        elapsed = now - last_time
        tokens = min(self.burst, tokens + elapsed * self.rate)

        if tokens < 1.0:
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
            )

        self._buckets[client_ip] = (tokens - 1.0, now)
        return await call_next(request)
