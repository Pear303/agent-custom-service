"""FAQ/帮助台辅助工具"""
from __future__ import annotations


class HelpdeskTool:
    """帮助台。FAQ 搜索、常见问题、话术模板。"""

    def search_faq(self, keywords: str, limit: int = 3) -> str:
        return f"FAQ 搜索结果（待实现）"

    def get_common_questions(self, category: str = None) -> str:
        return "常见问题列表（待实现）"

    def get_templates(self, type: str = "greeting") -> str:
        return "话术模板（待实现）"
