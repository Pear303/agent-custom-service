"""Token 使用追踪器：按调用记录 JSONL 日志并提供聚合统计。

功能：
- 记录每次 LLM 调用的 token 消耗（输入/输出/总计）
- 支持 OpenAI 格式和原始计数两种记录方式
- 提供按日期和按模型的统计分析
- 判断是否需要触发历史压缩
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class TokenTracker:
    """Token 使用追踪器。
    
    将每次 LLM 调用的 token 使用情况记录到 JSONL 文件，
    并提供统计分析和压缩触发判断功能。
    
    日志文件格式（tokens.jsonl）：
    {
        "ts": "2024-01-01T12:00:00",
        "model": "deepseek-v4-flash",
        "input": 1000,
        "output": 500,
        "total": 1500
    }
    """
    
    def __init__(self, log_file: Path | None = None, user_id: str | None = None):
        """初始化 Token 追踪器。
        
        Args:
            log_file: JSONL 日志文件路径（可选，与 user_id 二选一）
            user_id: 用户唯一标识（提供时自动构建路径为 data/{user_id}/tokens.jsonl）
        """
        if user_id:
            from pathlib import Path as _Path
            self.log_file = _Path(__file__).parent.parent / "data" / "users" / user_id / "tokens.jsonl"
        else:
            self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_input_tokens = 0

    def record(self, model: str, usage) -> None:
        """从 OpenAI ChatCompletion.usage 对象追加一行到 tokens.jsonl。
        
        【调用方】lc_agent.py
        
        Args:
            model: 使用的模型名称
            usage: OpenAI API 返回的 usage 对象，包含 prompt_tokens、completion_tokens、total_tokens
        """
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "input": getattr(usage, "prompt_tokens", 0) or 0,
            "output": getattr(usage, "completion_tokens", 0) or 0,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }
        self._last_input_tokens = row["input"]
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_raw(self, model: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        """从显式 token 计数追加一行（用于 LangChain callback 等非 OpenAI 格式）。
        
        【调用方】lc_agent.py
        
        Args:
            model: 使用的模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            total_tokens: 总 token 数
        """
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        }
        self._last_input_tokens = input_tokens
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def last_input_tokens(self) -> int:
        """获取最近一次调用的输入 token 数。
        
        【调用方】lc_agent.py
        
        Returns:
            输入 token 数量
        """
        return self._last_input_tokens

    def should_compact(self, max_context: int, threshold: float = 0.6) -> bool:
        """判断是否应该触发历史压缩。
        
        【调用方】lc_agent.py
        
        当最近一次调用的输入 token 数超过阈值时触发压缩。
        
        Args:
            max_context: 上下文窗口的最大 token 数
            threshold: 触发压缩的阈值比例（默认 0.6 即 60%）
            
        Returns:
            如果应该压缩则返回 True
        """
        return self._last_input_tokens > max_context * threshold

    def _iter_rows(self):
        """迭代读取 JSONL 日志文件的所有行。
        
        Yields:
            解析后的字典对象，跳过空行和无效 JSON
        """
        if not self.log_file.exists():
            return
        with self.log_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def stats_by_date(self) -> dict[str, dict[str, int]]:
        """按日期统计 token 使用情况。
        
        Returns:
            字典，键为日期字符串（YYYY-MM-DD），值为包含 input/output/total 的字典
        """
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        for r in self._iter_rows():
            date = r.get("ts", "")[:10]  # 提取日期部分
            for k in ("input", "output", "total"):
                out[date][k] += r.get(k, 0)
        return dict(out)

    def stats_by_model(self) -> dict[str, dict[str, int]]:
        """按模型统计 token 使用情况。
        
        Returns:
            字典，键为模型名称，值为包含 input/output/total 的字典
        """
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        for r in self._iter_rows():
            m = r.get("model", "unknown")
            for k in ("input", "output", "total"):
                out[m][k] += r.get(k, 0)
        return dict(out)