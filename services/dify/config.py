import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)


class DifyConfig:
    base_url: str
    api_key: str

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "DifyConfig":
        return cls(
            base_url=os.getenv("DIFY_BASE_URL", "http://127.0.0.1:80"),
            api_key=os.getenv("DIFY_API_KEY", ""),
        )
