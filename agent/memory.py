"""三层记忆存储系统：原始历史 / 每日情景记忆 / 长期记忆。

实现 LangChain 的 BaseChatMessageHistory 接口以提供标准化的聊天历史持久化，
同时保留自定义的三层记忆架构（JSONL → 每日情景记忆 → 长期 MEMORY.md）。

三层记忆结构：
1. 工作记忆（Working Memory）：内存中的 history 列表，每轮对话追加
2. 情景记忆（Episodic Memory）：按日历日分割的 YYYY-MM-DD.md 文件
3. 长期记忆（Long-term Memory）：MEMORY.md 文件，每轮注入 system prompt
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage


# 定义 UTC+8 时区
_UTC8 = timezone(timedelta(hours=8))

# LangChain 消息类型到 JSONL 角色的映射
_TYPE_TO_JSONL_ROLE: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}

# JSONL 角色到 LangChain 消息类的反向映射
_JSONL_ROLE_TO_MESSAGE_CLS: dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}

"""
# 额外扩展的功能
    def append_history()           # 追加单条记录
    def today_episode_path()       # 今日情景记忆路径
    def read_today_episode()       # 读取情景记忆
    def append_episode()           # 追加情景记忆
    def read_memory()              # 读取长期记忆
    def write_memory()             # 写入长期记忆
    def append_compact_marker()    # 添加压缩标记
"""



class MemoryStore(BaseChatMessageHistory):
    """三层记忆存储管理器。
    
    继承自 LangChain 的 BaseChatMessageHistory，提供标准化的消息持久化接口，
    同时管理三层记忆系统的文件读写和归档逻辑。
    
    文件结构：
    - memory_dir/
      ├── MEMORY.md          # 长期记忆文件
      ├── history.jsonl      # 原始对话日志（JSON Lines 格式）
      ├── tokens.jsonl       # Token 使用记录
      └── YYYY-MM-DD.md      # 每日情景记忆文件
    """
    
    def __init__(self, memory_dir: Path | None = None, user_file: Path | None = None, user_id: str | None = None):
        """初始化记忆存储管理器。
        
        Args:
            memory_dir: 记忆文件存储目录（可选，与 user_id 二选一）
            user_file: 用户偏好档案文件路径（可选，与 user_id 二选一）
            user_id: 用户唯一标识（提供时自动构建路径为 data/{user_id}/memory/）
        """
        if user_id:
            from pathlib import Path as _Path
            _base = _Path(__file__).parent.parent / "data" / "users" / user_id
            self.memory_dir = _base / "memory"
            self.user_file = _base / "USER.md"
        else:
            self.memory_dir = memory_dir
            self.user_file = user_file
        
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self._ensure()

    def _ensure(self) -> None:
        """确保记忆目录和必要文件存在，不存在则创建默认内容。
        
        创建的文件包括：
        - 记忆目录（递归创建）
        - MEMORY.md（带默认标题）
        - history.jsonl（空文件）
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text("# 长期记忆\n\n此文件常驻上下文，记录核心目标、当前任务与关键事实。\n", encoding="utf-8")
        if not self.history_file.exists():
            self.history_file.write_text("")

    # ── 原始层：JSONL 历史日志 ──────────────────────────────
    
    def append_history(self, role: str, content: Any, additional_kwargs: dict | None = None) -> None:
        """向 history.jsonl 追加一条对话记录。
        
        【调用方】lc_agent.py (通过 add_messages 间接调用)
        
        Args:
            role: 消息角色（user/assistant/system/tool/function）
            content: 消息内容（字符串或可序列化的复杂对象）
            additional_kwargs: 额外的元数据（如工具调用信息）
        """
        row = {
            "ts": datetime.now(_UTC8).isoformat(timespec="seconds"),  # ISO 格式时间戳
            "role": role,
            "content": content if isinstance(content, str) else _json_safe(content),
        }
        if additional_kwargs:
            row["additional_kwargs"] = additional_kwargs
            
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量添加 LangChain 消息对象到历史记录。
        
        【调用方】lc_agent.py
        
        将 LangChain 的 BaseMessage 转换为 JSONL 格式并持久化。
        
        Args:
            messages: LangChain 消息序列
        """
        for msg in messages:
            role = _TYPE_TO_JSONL_ROLE.get(msg.type, "unknown")
            extra = getattr(msg, "additional_kwargs", None) or None
            self.append_history(role, msg.content, additional_kwargs=extra)

    # ── 中期层：按日历日（UTC+8）的情景记忆 ─────────────────────
    
    def today_episode_path(self) -> Path:
        """获取今日情景记忆文件的路径。
        
        Returns:
            今日日期对应的 .md 文件路径（格式：YYYY-MM-DD.md）
        """
        date = datetime.now(_UTC8).strftime("%Y-%m-%d")
        return self.memory_dir / f"{date}.md"

    def read_today_episode(self) -> str:
        """读取今日情景记忆内容。
        
        Returns:
            今日情景记忆的文本内容，如果文件不存在则返回空字符串
        """
        p = self.today_episode_path()
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def append_episode(self, content: str) -> None:
        """向今日情景记忆文件追加内容。
        
        如果文件已存在，在现有内容后追加；否则创建新文件并添加标题。
        
        Args:
            content: 要追加的情景记忆内容
        """
        p = self.today_episode_path()
        existing = p.read_text(encoding="utf-8") if p.exists() else f"# {p.stem} 情景记忆\n"
        new_text = existing.rstrip() + "\n\n" + content.strip() + "\n"
        p.write_text(new_text, encoding="utf-8")

    # ── 长期层：MEMORY.md ──────────────────────────────────────
    
    def read_memory(self) -> str:
        """读取长期记忆文件内容。
        
        【调用方】context.py, compactor.py
        
        Returns:
            MEMORY.md 的完整内容，如果文件不存在或解码失败则返回空字符串
        """
        if not self.memory_file.exists():
            return ""
        try:
            return self.memory_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 如果 UTF-8 解码失败，尝试 GBK 编码作为后备方案
            return self.memory_file.read_text(encoding="gbk", errors="ignore")

    def write_memory(self, content: str) -> None:
        """写入长期记忆文件（覆盖式）。
        
        【调用方】compactor.py
        
        Args:
            content: 新的长期记忆内容
        """
        self.memory_file.write_text(content.strip() + "\n", encoding="utf-8")

    # ── 归档标记：compact_event ───────────────────────────────
    
    def append_compact_marker(self) -> None:
        """在 history.jsonl 中添加压缩事件标记。
        
        用于标识某段历史已被压缩归档，启动时可根据此标记跳过已归档部分。
        """
        row = {"ts": datetime.now(_UTC8).isoformat(timespec="seconds"), "type": "compact_event"}
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        """清空当前会话状态（通过添加压缩标记实现）。
        
        注意：不会删除历史文件，只是标记当前状态为已归档。
        """
        self.append_compact_marker()

    def load_unarchived_history(self) -> list:
        """加载最后一个 compact_event 标记之后的未归档对话条目。
        
        扫描 history.jsonl，找到最后一个压缩事件标记，返回其后的所有有效对话记录。
        
        Returns:
            未归档的对话记录列表，每个元素是 {role, content} 字典
        """
        if not self.history_file.exists():
            return []
            
        rows = []
        with self.history_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
        # 找到最后一个 compact_event 的位置
        last_marker = -1
        for i, row in enumerate(rows):
            if row.get("type") == "compact_event":
                last_marker = i
                
        # 返回标记之后的所有有效对话记录
        return [
            {"role": r["role"], "content": r["content"]}
            for r in rows[last_marker + 1:]
            if "role" in r and "content" in r
        ]

    @property
    def messages(self) -> list[BaseMessage]:
        """获取未归档历史对应的 LangChain 消息对象列表。
        
        从 JSONL 格式反序列化为 LangChain 的 BaseMessage 对象，
        保留复杂内容块（如 tool_use / tool_result）的结构。
        
        Returns:
            LangChain 消息对象列表
        """
        raw = self.load_unarchived_history()
        result: list[BaseMessage] = []
        for entry in raw:
            role = entry["role"]
            content = entry["content"]
            extra_kwargs = entry.get("additional_kwargs", None)
            
            # 根据角色选择对应的消息类
            message_cls = _JSONL_ROLE_TO_MESSAGE_CLS.get(role)
            if message_cls is not None:
                # content 从 JSON 反序列化而来，保持原样（str / list / dict）
                # 以保留复杂内容块（如 tool_use / tool_result）的结构
                if extra_kwargs:
                    result.append(message_cls(content=content, additional_kwargs=extra_kwargs))
                else:
                    result.append(message_cls(content=content))
                    
        return result

    # ── 用户偏好档案 ───────────────────────────────────────────
    
    def read_user(self) -> str:
        """读取用户偏好档案内容。
        
        Returns:
            USER.md 文件的完整内容，如果文件不存在则返回空字符串
        """
        return self.user_file.read_text(encoding="utf-8") if self.user_file.exists() else ""

    def write_user(self, content: str) -> None:
        """写入用户偏好档案（覆盖式）。
        
        Args:
            content: 新的用户偏好内容
        """
        self.user_file.write_text(content.strip() + "\n", encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    """将任意对象转换为 JSON 可序列化的形式。
    
    处理 Anthropic 内容块或其他复杂对象，确保可以安全地写入 JSONL 文件。
    
    转换优先级：
    1. 如果对象本身可序列化，直接返回
    2. 列表：递归转换每个元素
    3. 字典：递归转换每个值
    4. Pydantic 模型：调用 model_dump()
    5. 普通对象：提取 __dict__ 属性（排除私有属性）
    6. 其他：转为字符串
    
    Args:
        obj: 任意对象
        
    Returns:
        JSON 可序列化的对象
    """
    # 首先尝试直接序列化
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        pass
        
    # 递归处理列表
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
        
    # 递归处理字典
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
        
    # 处理 Pydantic 模型
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
        
    # 处理普通 Python 对象
    if hasattr(obj, "__dict__"):
        return {k: _json_safe(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        
    # 兜底：转为字符串
    return str(obj)


"""
          每轮对话
             │
             ▼
    ┌────────────────┐
    │  工作记忆        │  memory/history.jsonl
    │  (JSONL 持久化)  │  ← 每轮追加
    └───────┬────────┘
            │ 触发压缩阈值（>140K tokens）
            ▼
    ┌────────────────┐
    │  Compactor 压缩  │  调用 LLM 总结旧对话
    │  (LLM 提炼)      │  解析 <episode> <updated_memory> <updated_user>
    └───────┬────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌──────────┐  ┌──────────┐
│ 情景记忆   │  │ 长期记忆   │
│ YYYY-MM-DD│  │ MEMORY.md│
│ .md       │  │ (常驻)    │
└──────────┘  └──────────┘
"""