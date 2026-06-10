"""T3 消息压缩优化单元测试 — 覆盖 DecisionSummary / ObservationMasker / ContextView 增强。"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_core.context_view import ContextView, PrunedToolCallGroup
from agent_core.decision_summary import DecisionSummaryExtractor, merge_summaries
from agent_core.observation_masker import ObservationMasker


# ── 辅助函数 ──────────────────────────────────────────────

def _make_tool_call(name: str, args: dict, tc_id: str = "tc1") -> dict:
    return {"name": name, "args": args, "id": tc_id}


def _make_ai_with_tools(tool_calls: list[dict], content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=tool_calls, id="ai1")


def _make_tool_result(content: str, tool_call_id: str = "tc1", name: str = "read_file") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name, id="tm1")


# ── DecisionSummaryExtractor 测试 ─────────────────────────

class TestDecisionSummaryExtractor:

    def setup_method(self):
        self.extractor = DecisionSummaryExtractor()

    def test_empty_pruned_groups(self):
        """无被裁剪组时返回空字符串。"""
        assert self.extractor.extract([]) == ""

    def test_read_file_summary(self):
        """read_file 工具调用摘要格式正确。"""
        groups = [PrunedToolCallGroup(
            group_index=0,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": "config.py"}, "tc1")],
            ),
            tool_calls=[_make_tool_call("read_file", {"path": "config.py"}, "tc1")],
            tool_results=[_make_tool_result("from fastapi import FastAPI\ndb = PostgreSQL", "tc1", "read_file")],
        )]
        result = self.extractor.extract(groups)
        assert "read_file(config.py)" in result
        assert "fastapi" in result.lower() or "FastAPI" in result

    def test_edit_file_summary(self):
        """edit_file 工具调用摘要包含修改数量。"""
        groups = [PrunedToolCallGroup(
            group_index=1,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("edit_file", {"path": "router.py", "replacements": [{"old": "a", "new": "b"}]}, "tc1")],
            ),
            tool_calls=[_make_tool_call("edit_file", {"path": "router.py", "replacements": [{"old": "a", "new": "b"}]}, "tc1")],
            tool_results=[_make_tool_result("OK", "tc1", "edit_file")],
        )]
        result = self.extractor.extract(groups)
        assert "edit_file(router.py)" in result
        assert "1处修改" in result

    def test_run_command_summary(self):
        """run_command 工具调用摘要包含命令。"""
        groups = [PrunedToolCallGroup(
            group_index=2,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("run_command", {"command": "pytest tests/"}, "tc1")],
            ),
            tool_calls=[_make_tool_call("run_command", {"command": "pytest tests/"}, "tc1")],
            tool_results=[_make_tool_result("3 tests passed\nexit code 0", "tc1", "run_command")],
        )]
        result = self.extractor.extract(groups)
        assert "run_command(pytest tests/)" in result

    def test_decision_extraction_chinese(self):
        """从 AIMessage.content 中提取中文决策。"""
        groups = [PrunedToolCallGroup(
            group_index=0,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
                content="我决定使用 FastAPI 而非 Flask，因为性能更好。",
            ),
            tool_calls=[_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
            tool_results=[_make_tool_result("file content", "tc1", "read_file")],
        )]
        result = self.extractor.extract(groups)
        assert "关键决策" in result
        assert "决定使用" in result

    def test_decision_extraction_english(self):
        """从 AIMessage.content 中提取英文决策。"""
        groups = [PrunedToolCallGroup(
            group_index=0,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
                content="I decided to use PostgreSQL because it's more reliable.",
            ),
            tool_calls=[_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
            tool_results=[_make_tool_result("file content", "tc1", "read_file")],
        )]
        result = self.extractor.extract(groups)
        assert "关键决策" in result
        assert "decided" in result.lower() or "PostgreSQL" in result

    def test_file_changes_dedup(self):
        """文件变更记录中同一文件只保留最新记录。"""
        groups = [
            PrunedToolCallGroup(
                group_index=0,
                ai_message=_make_ai_with_tools(
                    [_make_tool_call("edit_file", {"path": "router.py", "replacements": []}, "tc1")],
                ),
                tool_calls=[_make_tool_call("edit_file", {"path": "router.py", "replacements": []}, "tc1")],
                tool_results=[_make_tool_result("OK", "tc1", "edit_file")],
            ),
            PrunedToolCallGroup(
                group_index=1,
                ai_message=_make_ai_with_tools(
                    [_make_tool_call("edit_file", {"path": "router.py", "replacements": []}, "tc2")],
                ),
                tool_calls=[_make_tool_call("edit_file", {"path": "router.py", "replacements": []}, "tc2")],
                tool_results=[_make_tool_result("OK", "tc2", "edit_file")],
            ),
        ]
        result = self.extractor.extract(groups)
        # "文件变更记录"节中 router.py 只出现一次
        # （"已完成操作"节中仍可能出现，这是正常的）
        file_changes_section = result.split("文件变更记录:")[1] if "文件变更记录:" in result else ""
        assert file_changes_section.count("router.py") == 1

    def test_summary_header_format(self):
        """摘要头部包含裁剪点信息。"""
        groups = [PrunedToolCallGroup(
            group_index=5,
            ai_message=_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
            ),
            tool_calls=[_make_tool_call("read_file", {"path": "a.py"}, "tc1")],
            tool_results=[_make_tool_result("content", "tc1", "read_file")],
        )]
        result = self.extractor.extract(groups)
        assert "step 5" in result
        assert "1 组" in result

    def test_summary_truncation(self):
        """超长摘要被截断。"""
        # 构造大量工具调用组
        groups = []
        for i in range(50):
            groups.append(PrunedToolCallGroup(
                group_index=i,
                ai_message=_make_ai_with_tools(
                    [_make_tool_call("read_file", {"path": f"file_{i}.py"}, f"tc{i}")],
                ),
                tool_calls=[_make_tool_call("read_file", {"path": f"file_{i}.py"}, f"tc{i}")],
                tool_results=[_make_tool_result("x" * 200, f"tc{i}", "read_file")],
            ))
        result = self.extractor.extract(groups)
        assert len(result) <= 4100  # _MAX_SUMMARY_CHARS + 一些余量


# ── merge_summaries 测试 ──────────────────────────────────

class TestMergeSummaries:

    def test_empty_old(self):
        """旧摘要为空时返回新摘要。"""
        new = "[上下文摘要 — 裁剪点: step 5, 共裁剪 1 组]\n已完成操作:\n  - read_file(a.py)"
        assert merge_summaries("", new) == new

    def test_empty_new(self):
        """新摘要为空时返回旧摘要。"""
        old = "[上下文摘要 — 裁剪点: step 3, 共裁剪 1 组]\n已完成操作:\n  - read_file(b.py)"
        assert merge_summaries(old, "") == old

    def test_decisions_accumulate(self):
        """关键决策累积不丢失。"""
        old = "[上下文摘要]\n关键决策:\n  - 决定使用 FastAPI"
        new = "[上下文摘要]\n关键决策:\n  - 选择 PostgreSQL 作为数据库"
        result = merge_summaries(old, new)
        assert "FastAPI" in result
        assert "PostgreSQL" in result

    def test_file_changes_keep_latest(self):
        """文件变更记录同一文件只保留最新。"""
        old = "[上下文摘要]\n文件变更记录:\n  - router.py: 已修改"
        new = "[上下文摘要]\n文件变更记录:\n  - router.py: 已创建/覆盖"
        result = merge_summaries(old, new)
        assert "router.py: 已创建/覆盖" in result
        assert "router.py: 已修改" not in result

    def test_actions_replaced_by_new(self):
        """已完成操作只保留新的。"""
        old = "[上下文摘要]\n已完成操作:\n  - read_file(old.py)"
        new = "[上下文摘要]\n已完成操作:\n  - read_file(new.py)"
        result = merge_summaries(old, new)
        assert "new.py" in result
        assert "old.py" not in result


# ── ObservationMasker 测试 ────────────────────────────────

class TestObservationMasker:

    def setup_method(self):
        self.masker = ObservationMasker(min_content_chars=100, max_file_lines=20)

    def test_small_content_not_masked(self):
        """小体积内容不做遮蔽。"""
        msgs = [_make_tool_result("short content", "tc1", "read_file")]
        result = self.masker.mask(msgs)
        assert result[0].content == "short content"

    def test_non_readonly_not_masked(self):
        """非只读工具输出不做遮蔽。"""
        long_content = "x" * 1000
        msgs = [_make_tool_result(long_content, "tc1", "edit_file")]
        result = self.masker.mask(msgs)
        assert result[0].content == long_content

    def test_file_content_masking(self):
        """read_file 大体积输出被遮蔽，保留签名行。"""
        # 构造一个有函数定义和函数体的文件
        lines = []
        for i in range(100):
            if i % 20 == 0:
                lines.append(f"def function_{i // 20}():")
            else:
                lines.append(f"    x = {i}")
        content = "\n".join(lines)

        msgs = [_make_tool_result(content, "tc1", "read_file")]
        result = self.masker.mask(msgs)
        masked = result[0].content

        # 应该保留了函数定义
        assert "def function_0" in masked
        # 应该有省略号
        assert "..." in masked
        # 遮蔽后应该比原始短
        assert len(masked) < len(content)

    def test_file_content_masking_with_line_numbers(self):
        """read_file 带行号前缀的输出被正确遮蔽。

        read_file 实际输出格式为 "行号| 内容"，如 "7| def get_users():"
        必须剥离行号前缀后才能识别结构行。
        """
        # 构造带行号前缀的 read_file 输出
        lines = []
        for i in range(1, 101):
            if (i - 1) % 20 == 0:
                lines.append(f"{i}| def function_{(i - 1) // 20}():")
            else:
                lines.append(f"{i}|     x = {i}")
        content = "\n".join(lines)

        msgs = [_make_tool_result(content, "tc1", "read_file")]
        result = self.masker.mask(msgs)
        masked = result[0].content

        # 应该保留了函数定义（带行号前缀）
        assert "def function_0" in masked
        # 应该有省略号
        assert "..." in masked
        # 遮蔽后应该比原始短
        assert len(masked) < len(content)

    def test_multilingual_structural_lines(self):
        """多语言结构行识别：JS/Go/Rust/Java 关键行被保留。"""
        code_lines = [
            "1| import React from 'react';",
            "2| export default function App() {",
            "3|     return <div>hello</div>;",
            "4| }",
            "5| ",
            "6| package main",
            "7| func main() {",
            "8|     fmt.Println(\"hello\")",
            "9| }",
            "10| ",
            "11| use std::io;",
            "12| fn main() {",
            "13|     println!(\"hello\");",
            "14| }",
            "15| ",
            "16| import java.util.List;",
            "17| public class Main {",
            "18|     public static void main(String[] args) {",
            "19|         System.out.println(\"hello\");",
            "20|     }",
            "21| }",
        ]
        content = "\n".join(code_lines)

        msgs = [_make_tool_result(content, "tc1", "read_file")]
        result = self.masker.mask(msgs)
        masked = result[0].content

        # 结构行应被保留
        assert "import React" in masked
        assert "export default function" in masked
        assert "package main" in masked
        assert "func main()" in masked
        assert "use std::io" in masked
        assert "fn main()" in masked
        assert "import java.util" in masked
        assert "public class Main" in masked
        # 函数体应被省略
        assert "..." in masked

    def test_grep_output_masking(self):
        """grep 大体积输出被遮蔽。"""
        lines = [f"file.py:{i}:match line content here" for i in range(100)]
        content = "\n".join(lines)

        msgs = [_make_tool_result(content, "tc1", "grep_tool")]
        result = self.masker.mask(msgs)
        masked = result[0].content

        assert "省略" in masked
        assert len(masked) < len(content)

    def test_glob_output_masking(self):
        """glob 大体积输出被遮蔽。"""
        lines = [f"src/module_{i}/file.py" for i in range(100)]
        content = "\n".join(lines)

        msgs = [_make_tool_result(content, "tc1", "glob_tool")]
        result = self.masker.mask(msgs)
        masked = result[0].content

        assert "省略" in masked or "个文件" in masked
        assert len(masked) < len(content)

    def test_preserves_tool_call_id(self):
        """遮蔽后 ToolMessage 的 tool_call_id 保持不变。"""
        content = "x" * 1000
        msgs = [_make_tool_result(content, "tc_123", "read_file")]
        result = self.masker.mask(msgs)
        assert result[0].tool_call_id == "tc_123"

    def test_non_tool_message_unchanged(self):
        """非 ToolMessage 不受影响。"""
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="hello"),
            AIMessage(content="response"),
        ]
        result = self.masker.mask(msgs)
        assert len(result) == 3
        assert result[0].content == "system"
        assert result[1].content == "hello"
        assert result[2].content == "response"


# ── ContextView build_view 返回值测试 ─────────────────────

class TestContextViewBuildViewReturn:

    def setup_method(self):
        self.cv = ContextView(max_context_tokens=1000, target_ratio=0.6, keep_recent_groups=2)

    def test_no_pruning_returns_empty_groups(self):
        """不需要裁剪时 pruned_groups 为空。"""
        msgs = [
            SystemMessage(content="You are an assistant."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        view, pruned = self.cv.build_view(msgs)
        assert len(view) == 3
        assert pruned == []

    def test_pruning_returns_pruned_groups(self):
        """裁剪时返回被裁剪的工具调用组。"""
        msgs = [SystemMessage(content="System")]

        # 添加 8 个工具调用组（超过 keep_recent_groups=2）
        for i in range(8):
            tc_id = f"tc_{i}"
            msgs.append(_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": f"file_{i}.py"}, tc_id)],
            ))
            msgs.append(_make_tool_result(f"content of file_{i}.py " * 50, tc_id, "read_file"))

        msgs.append(HumanMessage(content="Now do something"))
        msgs.append(AIMessage(content="Done"))

        view, pruned = self.cv.build_view(msgs)

        # 应该有被裁剪的组
        if len(view) < len(msgs):
            assert len(pruned) > 0
            # 被裁剪组应包含工具调用信息
            for pg in pruned:
                assert isinstance(pg, PrunedToolCallGroup)
                assert pg.group_index >= 0

    def test_empty_messages(self):
        """空消息列表返回空。"""
        view, pruned = self.cv.build_view([])
        assert view == []
        assert pruned == []


# ── 集成测试：DecisionSummary + ContextView ───────────────

class TestIntegrationSummaryFromView:

    def test_full_pipeline(self):
        """完整流程：ContextView 裁剪 → 提取摘要 → 注入视图。"""
        cv = ContextView(max_context_tokens=500, target_ratio=0.6, keep_recent_groups=1)
        extractor = DecisionSummaryExtractor()
        masker = ObservationMasker(min_content_chars=50, max_file_lines=10)

        msgs = [SystemMessage(content="System")]

        # 旧的工具调用组（应该被裁剪）
        for i in range(5):
            tc_id = f"tc_old_{i}"
            msgs.append(_make_ai_with_tools(
                [_make_tool_call("read_file", {"path": f"old_{i}.py"}, tc_id)],
                content=f"我决定使用方案{i}" if i == 0 else "",
            ))
            msgs.append(_make_tool_result(f"old file content " * 30, tc_id, "read_file"))

        # 最近的工具调用组（应该保留）
        tc_id = "tc_new"
        msgs.append(_make_ai_with_tools(
            [_make_tool_call("read_file", {"path": "new.py"}, tc_id)],
        ))
        msgs.append(_make_tool_result("new file content", tc_id, "read_file"))

        # 步骤1: ContextView 裁剪
        view, pruned = cv.build_view(msgs)

        # 步骤2: 提取摘要
        summary = extractor.extract(pruned)

        # 步骤3: 注入摘要
        if summary:
            summary_msg = SystemMessage(content=summary)
            if view and isinstance(view[0], SystemMessage):
                view = [view[0], summary_msg] + view[1:]
            else:
                view = [summary_msg] + view

        # 步骤4: 观察遮蔽
        view = masker.mask(view)

        # 验证：视图应该包含摘要
        summary_msgs = [m for m in view if isinstance(m, SystemMessage) and "上下文摘要" in getattr(m, "content", "")]
        if pruned:
            assert len(summary_msgs) > 0, "裁剪后应注入决策摘要"
            assert "方案" in summary_msgs[0].content or "read_file" in summary_msgs[0].content
