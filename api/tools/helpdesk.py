"""FAQ/帮助台辅助工具"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HELPDESK_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "helpdesk.json"

_DEFAULT_FAQ: list[dict[str, Any]] = [
    {
        "id": 1,
        "question": "如何创建工单？",
        "answer": "您可以通过客服对话窗口直接描述需求，系统会自动创建工单。也可以访问「提交需求」页面填写详细信息。",
        "category": "工单",
        "tags": ["创建", "工单", "提交需求"],
    },
    {
        "id": 2,
        "question": "如何查询工单进度？",
        "answer": "在对话窗口中提供您的工单编号，或直接询问「我的工单进度如何？」，系统会为您实时查询。",
        "category": "工单",
        "tags": ["查询", "进度", "状态"],
    },
    {
        "id": 3,
        "question": "工单处理需要多长时间？",
        "answer": "需求分析通常在提交后 5-10 分钟内完成，PRD 设计和成本估算各需 3-5 分钟。整体流程预计 15-20 分钟。",
        "category": "工单",
        "tags": ["时间", "处理", "预计"],
    },
    {
        "id": 4,
        "question": "如何联系人工客服？",
        "answer": "您可以在对话中直接说明需要人工客服，或使用「转人工」功能。我们的客服人员会在工作时间内尽快响应。",
        "category": "客服",
        "tags": ["人工", "转接", "客服"],
    },
    {
        "id": 5,
        "question": "支持哪些项目类型？",
        "answer": "我们支持网站开发、微信小程序、移动 App、自动化脚本、数据分析和 AI 应用等多种项目类型。",
        "category": "服务",
        "tags": ["类型", "项目", "支持"],
    },
    {
        "id": 6,
        "question": "如何重新提交失败的工单？",
        "answer": "失败或已完成的工单可以重试。在对话中说明需要重试的工单编号，系统会自动将工单重新加入处理队列。",
        "category": "工单",
        "tags": ["重试", "失败", "重新提交"],
    },
    {
        "id": 7,
        "question": "报价是按什么标准计算的？",
        "answer": "报价基于项目复杂度、功能数量、技术难度和预计工时综合计算。最终报价会在成本估算阶段生成报价单供您确认。",
        "category": "费用",
        "tags": ["报价", "费用", "价格"],
    },
    {
        "id": 8,
        "question": "如何查看开发成果？",
        "answer": "开发完成后，您可以在工单详情页面查看生成的文件列表和技术文档。也可通过「恢复到本地」功能将文件下载到本地目录。",
        "category": "开发",
        "tags": ["成果", "查看", "下载"],
    },
]

_COMMON_QUESTIONS: dict[str, list[str]] = {
    "工单": [
        "如何创建工单？",
        "如何查询工单进度？",
        "工单处理需要多长时间？",
        "如何重新提交失败的工单？",
    ],
    "客服": [
        "如何联系人工客服？",
        "客服工作时间是什么？",
        "是否支持电话咨询？",
    ],
    "费用": [
        "报价是按什么标准计算的？",
        "是否支持分期付款？",
        "有免费试用吗？",
    ],
    "服务": [
        "支持哪些项目类型？",
        "是否提供售后服务？",
        "可以修改已提交的需求吗？",
    ],
    "开发": [
        "如何查看开发成果？",
        "开发周期一般多久？",
        "是否支持二次开发？",
    ],
}

_RESPONSE_TEMPLATES: dict[str, str] = {
    "greeting": "您好！我是 AI 智能客服助手，请问有什么可以帮助您的？您可以提交开发需求、查询工单进度或咨询项目相关问题。",
    "farewell": "感谢您的咨询！如果还有其他问题，随时可以联系我。祝您生活愉快！",
    "transfer": "正在为您转接人工客服，请稍候...如需取消，请回复「取消转接」。",
    "waiting": "您的请求正在处理中，请耐心等待。您可以随时询问进度。",
    "error": "很抱歉，系统暂时遇到了一些问题。我们的技术团队正在努力修复中，请您稍后再试。",
    "acknowledgment": "已收到您的信息，我来帮您处理。",
    "clarification": "为了更好地帮助您，请问您能否提供更多详细信息？",
}


class HelpdeskTool:
    """帮助台。FAQ 搜索、常见问题、话术模板。"""

    def __init__(self):
        self._faq = self._load_faq()

    def _load_faq(self) -> list[dict[str, Any]]:
        try:
            if _HELPDESK_DATA_PATH.exists():
                data = json.loads(_HELPDESK_DATA_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning("Failed to load helpdesk data: %s", e)
        return _DEFAULT_FAQ

    def _format_results(self, results: list[dict[str, Any]], limit: int) -> str:
        if not results:
            return "未找到匹配的 FAQ。建议您描述具体问题，我会尽力帮助您。"
        lines = []
        for item in results[:limit]:
            lines.append(f"Q: {item['question']}\nA: {item['answer']}")
        result_text = "\n\n".join(lines)
        if len(results) > limit:
            result_text += f"\n\n（共找到 {len(results)} 条结果，以上显示前 {limit} 条）"
        return result_text

    def search_faq(self, keywords: str, limit: int = 3) -> str:
        if not keywords.strip():
            return "请提供搜索关键词。"
        keywords_lower = keywords.lower()
        scored = []
        for item in self._faq:
            score = 0
            q_lower = item["question"].lower()
            a_lower = item["answer"].lower()
            if keywords_lower in q_lower:
                score += 3
            if keywords_lower in a_lower:
                score += 2
            for tag in item.get("tags", []):
                if keywords_lower in tag.lower():
                    score += 2
            if keywords_lower in item.get("category", "").lower():
                score += 1
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored]
        return self._format_results(results, max(limit, 1))

    def get_common_questions(self, category: str = None) -> str:
        if category and category in _COMMON_QUESTIONS:
            questions = _COMMON_QUESTIONS[category]
            lines = [f"【{category}】常见问题："]
            for i, q in enumerate(questions, 1):
                lines.append(f"{i}. {q}")
            return "\n".join(lines)
        result: list[str] = []
        for cat, questions in _COMMON_QUESTIONS.items():
            result.append(f"【{cat}】")
            for i, q in enumerate(questions[:3], 1):
                result.append(f"  {i}. {q}")
        return "\n".join(result) if result else "暂无常见问题数据。"

    def get_templates(self, type: str = "greeting") -> str:
        if type in _RESPONSE_TEMPLATES:
            return _RESPONSE_TEMPLATES[type]
        available = ", ".join(_RESPONSE_TEMPLATES.keys())
        return f"未知模板类型 '{type}'，可用类型：{available}"
