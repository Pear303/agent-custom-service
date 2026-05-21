"""配置管理 —— 从环境变量读取服务配置"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    service_port: int = int(os.getenv("SERVICE_PORT", "8000"))
    agent_api_key: str = os.getenv("AGENT_API_KEY", "change-me")
    cs_mode: bool = os.getenv("CS_MODE", "true").lower() == "true"
    session_timeout_minutes: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "1000"))
    product_data_path: str | None = os.getenv("PRODUCT_DATA_PATH")


settings = Settings()
