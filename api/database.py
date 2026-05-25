"""兼容层：从 core.database 重新导出"""
from .core.database import Database, DB_PATH

__all__ = ["Database", "DB_PATH"]
