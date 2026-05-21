"""产品目录查询工具"""
from __future__ import annotations

_SAMPLE_PRODUCTS = []


class ProductCatalogTool:
    """产品目录查询。支持 search / detail / categories。"""

    def search(self, keywords: str, category: str = None, limit: int = 5) -> str:
        return f"搜索 '{keywords}' 的结果（待实现）"

    def get_detail(self, product_id: str) -> str:
        return f"产品 {product_id} 详情（待实现）"

    def get_categories(self) -> str:
        return "产品分类（待实现）"
