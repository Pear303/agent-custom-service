"""配置管理 —— 从环境变量读取服务配置"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    service_port: int = int(os.getenv("SERVICE_PORT", "8000"))
    agent_api_key: str = os.getenv("AGENT_API_KEY", "change-me")
    cs_mode: bool = os.getenv("CS_MODE", "true").lower() == "true"
    session_timeout_minutes: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    max_sessions: int = int(os.getenv("MAX_SESSIONS", "1000"))
    product_data_path: str | None = os.getenv("PRODUCT_DATA_PATH")

    agent_concurrency_limit: int = int(os.getenv("AGENT_CONCURRENCY_LIMIT", "5"))
    task_queue_size: int = int(os.getenv("TASK_QUEUE_SIZE", "100"))
    task_workers: int = int(os.getenv("TASK_WORKERS", "3"))
    agent_max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
    agent_base_delay: float = float(os.getenv("AGENT_BASE_DELAY", "1.0"))

    dify_max_retries: int = int(os.getenv("DIFY_MAX_RETRIES", "3"))
    dify_base_delay: float = float(os.getenv("DIFY_BASE_DELAY", "1.0"))
    dify_circuit_failure_threshold: int = int(os.getenv("DIFY_CIRCUIT_FAILURE_THRESHOLD", "5"))
    dify_circuit_window_seconds: float = float(os.getenv("DIFY_CIRCUIT_WINDOW_SECONDS", "60"))
    dify_circuit_recovery_timeout: float = float(os.getenv("DIFY_CIRCUIT_RECOVERY_TIMEOUT", "30"))


settings = Settings()
