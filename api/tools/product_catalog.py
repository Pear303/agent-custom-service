"""产品目录查询工具"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CATALOG_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "products.json"

_SAMPLE_PRODUCTS: list[dict[str, Any]] = [
    {
        "id": "P001",
        "name": "企业官网定制开发",
        "category": "网站开发",
        "price": "5000-20000元",
        "price_range": (5000, 20000),
        "delivery": "7-15个工作日",
        "description": "响应式企业官网，含首页、关于我们、产品展示、新闻动态、联系我们等标准页面，支持SEO优化。",
        "features": ["响应式设计", "SEO优化", "后台CMS", "多语言支持", "SSL证书"],
        "tags": ["官网", "企业", "品牌"],
    },
    {
        "id": "P002",
        "name": "微信小程序开发",
        "category": "小程序",
        "price": "8000-50000元",
        "price_range": (8000, 50000),
        "delivery": "10-20个工作日",
        "description": "微信小程序定制开发，涵盖电商、预约、社区、工具等多种场景。",
        "features": ["微信支付", "用户授权", "消息推送", "模板消息", "数据分析"],
        "tags": ["微信", "小程序", "电商"],
    },
    {
        "id": "P003",
        "name": "移动App开发 (iOS/Android)",
        "category": "移动App",
        "price": "20000-100000元",
        "price_range": (20000, 100000),
        "delivery": "20-45个工作日",
        "description": "跨平台移动应用开发，支持iOS和Android双端，使用Flutter/React Native技术栈。",
        "features": ["双端适配", "推送通知", "第三方登录", "数据同步", "离线缓存"],
        "tags": ["App", "移动端", "跨平台"],
    },
    {
        "id": "P004",
        "name": "自动化脚本开发",
        "category": "自动化脚本",
        "price": "1000-8000元",
        "price_range": (1000, 8000),
        "delivery": "1-5个工作日",
        "description": "定制自动化脚本，涵盖数据处理、网页爬虫、自动化测试、定时任务等场景。",
        "features": ["任务调度", "异常处理", "日志记录", "报告生成", "可配置"],
        "tags": ["自动化", "脚本", "数据处理"],
    },
    {
        "id": "P005",
        "name": "数据分析报表系统",
        "category": "数据分析",
        "price": "8000-30000元",
        "price_range": (8000, 30000),
        "delivery": "10-20个工作日",
        "description": "数据可视化报表系统，支持多数据源接入、图表展示、数据导出等功能。",
        "features": ["数据可视化", "多数据源", "交互式图表", "数据导出", "定时刷新"],
        "tags": ["数据分析", "报表", "可视化"],
    },
    {
        "id": "P006",
        "name": "AI智能客服系统",
        "category": "AI应用",
        "price": "15000-80000元",
        "price_range": (15000, 80000),
        "delivery": "15-30个工作日",
        "description": "AI驱动的智能客服系统，支持自然语言理解、多轮对话、知识库管理等。",
        "features": ["自然语言处理", "多轮对话", "知识库管理", "人工转接", "数据统计"],
        "tags": ["AI", "客服", "智能"],
    },
    {
        "id": "P007",
        "name": "电商平台开发",
        "category": "网站开发",
        "price": "30000-150000元",
        "price_range": (30000, 150000),
        "delivery": "20-40个工作日",
        "description": "完整的电商平台解决方案，包含商品管理、订单系统、支付集成、物流追踪等。",
        "features": ["商品管理", "购物车", "在线支付", "订单管理", "会员系统"],
        "tags": ["电商", "商城", "支付"],
    },
    {
        "id": "P008",
        "name": "技术咨询与系统维护",
        "category": "其他",
        "price": "2000-5000元/月",
        "price_range": (2000, 5000),
        "delivery": "持续服务",
        "description": "提供技术咨询服务，包括系统架构评估、性能优化、安全审计、系统运维等。",
        "features": ["性能优化", "安全审计", "系统监控", "故障排查", "技术培训"],
        "tags": ["咨询", "维护", "运维"],
    },
]

_CATEGORIES = sorted({p["category"] for p in _SAMPLE_PRODUCTS})


class ProductCatalogTool:
    """产品目录查询。支持 search / detail / categories。"""

    def __init__(self):
        self._products = self._load_products()

    def _load_products(self) -> list[dict[str, Any]]:
        try:
            if _CATALOG_DATA_PATH.exists():
                data = json.loads(_CATALOG_DATA_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning("Failed to load product catalog: %s", e)
        return _SAMPLE_PRODUCTS

    def search(self, keywords: str, category: str = None, limit: int = 5) -> str:
        if not keywords.strip():
            return "请提供搜索关键词。"

        keywords_lower = keywords.lower()
        scored: list[tuple[int, dict[str, Any]]] = []

        for product in self._products:
            if category and product.get("category", "") != category:
                continue
            score = 0
            name_lower = product["name"].lower()
            desc_lower = product.get("description", "").lower()
            if keywords_lower in name_lower:
                score += 5
            if keywords_lower in desc_lower:
                score += 3
            for tag in product.get("tags", []):
                if keywords_lower in tag.lower():
                    score += 2
            for feature in product.get("features", []):
                if keywords_lower in feature.lower():
                    score += 1
            if score > 0:
                scored.append((score, product))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored]

        if not results:
            available_cats = "、".join(self.get_category_list())
            return (
                f"未找到与「{keywords}」匹配的产品。\n"
                f"可用产品类别：{available_cats}\n"
                "建议尝试使用更通用的关键词或按类别浏览。"
            )

        lines = []
        for p in results[:max(limit, 1)]:
            lines.append(
                f"【{p['name']}】（{p['id']}）\n"
                f"  分类：{p['category']}\n"
                f"  价格：{p['price']}\n"
                f"  交付周期：{p['delivery']}\n"
                f"  简介：{p.get('description', '暂无')}\n"
                f"  特性：{'、'.join(p.get('features', [])[:5])}"
            )

        result_text = "\n\n".join(lines)
        if len(results) > limit:
            result_text += f"\n\n（共找到 {len(results)} 个产品，以上显示前 {limit} 个）"
        return result_text

    def get_detail(self, product_id: str) -> str:
        for product in self._products:
            if product["id"] == product_id:
                return (
                    f"产品名称：{product['name']}\n"
                    f"产品编号：{product['id']}\n"
                    f"分类：{product['category']}\n"
                    f"参考价格：{product['price']}\n"
                    f"交付周期：{product['delivery']}\n"
                    f"产品简介：{product.get('description', '暂无')}\n"
                    f"核心功能：{'、'.join(product.get('features', []))}\n"
                    f"标签：{'、'.join(product.get('tags', []))}"
                )
        return f"未找到编号为 {product_id} 的产品。可用编号：{', '.join(p['id'] for p in self._products)}"

    def get_categories(self) -> str:
        cat_counts: dict[str, int] = {}
        for p in self._products:
            cat = p["category"]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        lines = ["产品分类："]
        for cat, count in cat_counts.items():
            lines.append(f"  {cat}（{count} 个产品）")
        return "\n".join(lines)

    def get_category_list(self) -> list[str]:
        return sorted(set(p["category"] for p in self._products))
