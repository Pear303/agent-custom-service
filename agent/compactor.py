"""Compactor: 历史对话压缩器，将旧的历史对话总结为今日片段并更新 MEMORY.md / USER.md。
支持两种使用模式：
- 外部历史列表模式（compact/compact_startup）供 AgentRunner 使用
- 内部 MemoryStore 模式（compact_store）供 LCAgent 使用
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .memory import MemoryStore


# 定义 UTC+8 时区
_UTC8 = timezone(timedelta(hours=8))

# 加载压缩提示词模板文件
_PROMPT_FILE = Path(__file__).parent.parent / "templates" / "agent" / "compact_prompt.md"
_PROMPT_TEMPLATE = _PROMPT_FILE.read_text(encoding="utf-8")


def _extract(tag: str, text: str) -> str | None:
    """从文本中提取指定 XML 标签内的内容。
    
    Args:
        tag: XML 标签名称（如 'episode', 'updated_memory'）
        text: 包含 XML 标签的文本
        
    Returns:
        提取的内容字符串，如果未找到标签则返回 None
    """
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _messages_to_text(messages: list) -> str:
    """将历史消息列表扁平化为可读的文本格式，用于 LLM 提示词。
    
    处理不同类型的消息内容：
    - 纯文本消息
    - 多模态消息块（文本、工具调用、工具结果）
    
    Args:
        messages: 消息列表，每个消息是包含 role 和 content 的字典
        
    Returns:
        格式化后的文本字符串，每行以 [角色] 开头
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        
        # 处理纯文本内容
        if isinstance(content, str):
            parts.append(f"[{role}] {content}")
        # 处理多模态内容块列表
        elif isinstance(content, list):
            for block in content:
                # 获取内容块类型，兼容对象属性和字典键
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                
                # 文本块
                if btype == "text":
                    text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "")
                    parts.append(f"[{role}] {text}")
                # 工具调用块
                elif btype == "tool_use":
                    name = getattr(block, "name", None) or block.get("name", "")
                    parts.append(f"[{role}:tool_call] {name}")
                # 工具执行结果块
                elif btype == "tool_result":
                    c = getattr(block, "content", None) or block.get("content", "")
                    # 如果结果是列表，提取所有文本内容
                    if isinstance(c, list):
                        c = " ".join(
                            (getattr(x, "text", None) or (x.get("text") if isinstance(x, dict) else str(x)) or "")
                            for x in c
                        )
                    # 截取前300字符作为摘要
                    snippet = str(c)[:300]
                    parts.append(f"[{role}:tool_result] {snippet}")
                    
    return "\n".join(parts)


class Compactor:
    """历史对话压缩器类。
    
    负责将过长的对话历史压缩为简洁的摘要，并更新记忆文件，
    以避免上下文窗口溢出并保持长期记忆的连贯性。
    """
    
    K = 10  # 保留最近的消息轮数阈值

    def __init__(self, client, model: str, memory_store: MemoryStore, max_tokens: int = 4000):
        """初始化压缩器。
        
        Args:
            client: LLM 客户端实例（如 OpenAI 客户端）
            model: 使用的模型名称
            memory_store: 记忆存储管理器实例
            max_tokens: LLM 响应的最大 token 数限制
        """
        self.client = client
        self.model = model
        self.memory = memory_store
        self.max_tokens = max_tokens

    # ── 公共入口方法 ──────────────────────────────────────────────

    def compact(self, history: list) -> list:
        """压缩历史对话中较早的部分，保留最近的 K 条消息。
        
        【调用方】lc_agent.py (内部使用)
        
        工作流程：
        1. 如果历史长度不超过 K，直接返回
        2. 将 history[:-K] 部分进行压缩归档
        3. 写入今日片段和更新记忆文件
        4. 返回保留的最近 K 条消息
        
        Args:
            history: 完整的对话历史列表
            
        Returns:
            压缩后保留的最近 K 条消息
        """
        if len(history) <= self.K:
            return history
        old = history[: -self.K]
        self._run_compaction(old)
        print(f"[Compacted: {len(old)} turns → today episode + MEMORY updated]")
        return history[-self.K:]

    def compact_startup(self, history: list) -> None:
        """启动时压缩：将所有未归档的历史全量归档，不保留最近条目。
        
        【调用方】lc_agent.py (启动阶段)
        
        用于应用启动场景，确保之前的对话都被妥善归档到记忆文件中。
        
        Args:
            history: 待归档的完整历史列表
        """
        if len(history) < 2:
            return
        self._run_compaction(history)
        print(f"[Startup compacted: {len(history)} unarchived turns → MEMORY updated]")

    def compact_store(self, keep: int | None = None) -> None:
        """直接从 MemoryStore 读取未归档历史并进行压缩。
        
        【调用方】lc_agent.py
        
        专为 LCAgent 设计，无需调用者手动维护历史列表。
        
        Args:
            keep: 保留的最近消息数量，默认为 self.K (10)
        """
        history = self.memory.load_unarchived_history()
        effective_keep = keep if keep is not None else self.K
        if len(history) <= effective_keep:
            return
        old = history[:-effective_keep]
        self._run_compaction(old)
        print(f"[Compacted: {len(old)} turns → today episode + MEMORY updated]")

    # ── 内部实现方法 ──────────────────────────────────────────────

    def _run_compaction(self, old_messages: list) -> None:
        """核心压缩逻辑：调用 LLM 生成摘要 → 解析 XML 响应 → 写入三层记忆 + 压缩标记。
        
        压缩流程：
        1. 构建提示词，包含旧对话、当前记忆、用户信息和今日片段
        2. 调用 LLM 生成压缩摘要
        3. 从响应中提取三个 XML 标签：
           - <episode>: 今日对话片段摘要
           - <updated_memory>: 更新的长期记忆
           - <updated_user>: 更新的用户画像
        4. 分别写入对应的记忆文件
        5. 添加压缩标记，标记这些消息已归档
        
        Args:
            old_messages: 需要压缩的旧消息列表
        """
        # 构建压缩提示词，注入当前状态信息
        prompt = _PROMPT_TEMPLATE.format(
            old_conversation=_messages_to_text(old_messages),
            current_memory=self.memory.read_memory() or "(空)",
            current_user=self.memory.read_user() or "(空)",
            today_episode=self.memory.read_today_episode() or "(空)",
            now_hhmm=datetime.now(_UTC8).strftime("%H:%M"),
        )
        
        # 调用 LLM 进行压缩
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""

        # 解析并保存今日对话片段
        if episode := _extract("episode", text):
            self.memory.append_episode(episode)
            
        # 解析并更新长期记忆
        if new_memory := _extract("updated_memory", text):
            self.memory.write_memory(new_memory)
            
        # 解析并更新用户画像
        if new_user := _extract("updated_user", text):
            self.memory.write_user(new_user)
            
        # 添加压缩标记，标记这些消息已归档
        self.memory.append_compact_marker()