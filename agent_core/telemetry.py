"""Token 使用追踪器：按调用记录 JSONL 日志并提供聚合统计。

从 agent.telemetry 迁移，无 import 变更。
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class TokenTracker:
    """Token 使用追踪器。"""

    def __init__(self, log_file: Path | None = None, user_id: str | None = None):
        if user_id:
            self.log_file = Path(__file__).parent.parent / "data" / "users" / user_id / "tokens.jsonl"
        else:
            self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_input_tokens = 0

    def record(self, model: str, usage) -> None:
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
        return self._last_input_tokens

    def should_compact(self, max_context: int, threshold: float = 0.6) -> bool:
        return self._last_input_tokens > max_context * threshold

    def _iter_rows(self):
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
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        for r in self._iter_rows():
            date = r.get("ts", "")[:10]
            for k in ("input", "output", "total"):
                out[date][k] += r.get(k, 0)
        return dict(out)

    def stats_by_model(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})
        for r in self._iter_rows():
            m = r.get("model", "unknown")
            for k in ("input", "output", "total"):
                out[m][k] += r.get(k, 0)
        return dict(out)
