"""混合调用路由单元测试 — 验证子代理 + 普通/危险工具混合调用场景。

测试 _advance_phase 暂存逻辑、_aggregate_results 恢复逻辑、
_route_after_aggregate 路由逻辑的正确性。

运行方式:
    python -m pytest test_mixed_calls.py -v
    或
    python test_mixed_calls.py
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

# Windows 编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from langchain_core.messages import AIMessage, HumanMessage

# 导入被测函数
from agent_by_langgraph.lg_graph import (
    _DANGEROUS_TOOLS,
    _advance_phase,
    _aggregate_results,
    _route_after_aggregate,
    _route_after_agent,
)


def _make_state(
    messages=None,
    phase="gather",
    stall_count=0,
    pending_tool_calls=None,
    subagent_results=None,
):
    """构造测试用 AgentState。"""
    return {
        "messages": messages or [],
        "_phase": phase,
        "_stall_count": stall_count,
        "_pending_tool_calls": pending_tool_calls or [],
        "_pending_route": "agent",
        "subagent_results": subagent_results or [],
        "plan": "test plan",
        "_approval_next": "",
    }


def _make_tool_call(name: str, args: dict | None = None, tc_id: str = "tc1") -> dict:
    """构造 tool_call 字典。"""
    return {
        "name": name,
        "args": args or {},
        "id": tc_id,
        "type": "tool_call",
    }


class TestAdvancePhasePendingCalls(unittest.TestCase):
    """测试 _advance_phase 的混合调用暂存逻辑。"""

    def test_pure_subagent_calls_no_pending(self):
        """纯子代理调用：不应产生 pending_tool_calls。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "coder", "task": "read"}, "tc1"),
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "reader", "task": "search"}, "tc2"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _advance_phase(state)
        self.assertEqual(result.get("_pending_tool_calls", []), [])

    def test_mixed_calls_pending_normal(self):
        """子代理 + 普通工具：普通工具应被暂存。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "coder", "task": "read"}, "tc1"),
                _make_tool_call("read_file", {"path": "app.py"}, "tc2"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _advance_phase(state)
        pending = result.get("_pending_tool_calls", [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["name"], "read_file")

    def test_mixed_calls_pending_dangerous(self):
        """子代理 + 危险工具：危险工具应被暂存。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "coder", "task": "read"}, "tc1"),
                _make_tool_call("write_file", {"path": "app.py", "content": "x"}, "tc2"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _advance_phase(state)
        pending = result.get("_pending_tool_calls", [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["name"], "write_file")

    def test_mixed_calls_pending_multiple(self):
        """子代理 + 多个普通/危险工具：全部暂存。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "coder", "task": "read"}, "tc1"),
                _make_tool_call("read_file", {"path": "app.py"}, "tc2"),
                _make_tool_call("write_file", {"path": "app.py", "content": "x"}, "tc3"),
                _make_tool_call("run_command", {"command": "test"}, "tc4"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _advance_phase(state)
        pending = result.get("_pending_tool_calls", [])
        self.assertEqual(len(pending), 3)
        names = [tc["name"] for tc in pending]
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("run_command", names)

    def test_no_subagent_no_pending(self):
        """无子代理调用：不应产生 pending_tool_calls。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("read_file", {"path": "app.py"}, "tc1"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _advance_phase(state)
        self.assertEqual(result.get("_pending_tool_calls", []), [])

    def test_stale_pending_cleared(self):
        """无新 pending_calls 时，残留 _pending_tool_calls 应被清空。"""
        ai_msg = AIMessage(content="done", tool_calls=[])
        state = _make_state(messages=[ai_msg], pending_tool_calls=[_make_tool_call("read_file")])
        result = _advance_phase(state)
        self.assertEqual(result.get("_pending_tool_calls", []), [])


class TestAggregateResultsRouting(unittest.TestCase):
    """测试 _aggregate_results 的路由逻辑。"""

    def test_no_results_route_to_agent(self):
        """无子代理结果：路由到 agent。"""
        state = _make_state(subagent_results=[])
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_route"], "agent")

    def test_results_no_pending_route_to_agent(self):
        """有子代理结果但无 pending：路由到 agent。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[],
        )
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_route"], "agent")

    def test_results_with_normal_pending_route_to_tools(self):
        """有子代理结果 + 普通工具 pending：路由到 tools。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[_make_tool_call("read_file", {"path": "app.py"})],
        )
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_route"], "tools")

    def test_results_with_dangerous_pending_route_to_interrupt(self):
        """有子代理结果 + 危险工具 pending：路由到 interrupt_approval。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[_make_tool_call("write_file", {"path": "app.py", "content": "x"})],
        )
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_route"], "interrupt_approval")

    def test_results_with_mixed_pending_route_to_interrupt(self):
        """有子代理结果 + 普通+危险工具 pending：路由到 interrupt_approval。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[
                _make_tool_call("read_file", {"path": "app.py"}),
                _make_tool_call("edit_file", {"path": "app.py", "replacements": []}),
            ],
        )
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_route"], "interrupt_approval")

    def test_pending_msg_has_id(self):
        """pending AIMessage 必须有 id 字段。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[_make_tool_call("read_file")],
        )
        result = _aggregate_results(state)
        # 找到 pending_msg
        pending_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
        self.assertTrue(len(pending_msgs) > 0, "应有 pending AIMessage")
        self.assertIsNotNone(pending_msgs[0].id, "pending AIMessage 必须有 id")
        self.assertTrue(pending_msgs[0].id.startswith("pending-"), "id 应以 'pending-' 开头")

    def test_subagent_results_cleared_after_consume(self):
        """子代理结果消费后应被清空。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
        )
        result = _aggregate_results(state)
        # 清空信号
        from agent_by_langgraph.lg_graph import _SUBAGENT_CLEAR_SENTINEL
        self.assertEqual(result["subagent_results"], [_SUBAGENT_CLEAR_SENTINEL])

    def test_pending_tool_calls_cleared_after_consume(self):
        """pending_tool_calls 消费后应被清空。"""
        state = _make_state(
            subagent_results=["[coder] 完成"],
            pending_tool_calls=[_make_tool_call("read_file")],
        )
        result = _aggregate_results(state)
        self.assertEqual(result["_pending_tool_calls"], [])


class TestRouteAfterAggregate(unittest.TestCase):
    """测试 _route_after_aggregate 路由函数。"""

    def test_route_to_agent(self):
        state = _make_state(pending_tool_calls=[])
        state["_pending_route"] = "agent"
        self.assertEqual(_route_after_aggregate(state), "agent")

    def test_route_to_tools(self):
        state = _make_state()
        state["_pending_route"] = "tools"
        self.assertEqual(_route_after_aggregate(state), "tools")

    def test_route_to_interrupt(self):
        state = _make_state()
        state["_pending_route"] = "interrupt_approval"
        self.assertEqual(_route_after_aggregate(state), "interrupt_approval")

    def test_default_route(self):
        """无 _pending_route 字段时默认路由到 agent。"""
        state = _make_state()
        del state["_pending_route"]
        self.assertEqual(_route_after_aggregate(state), "agent")


class TestRouteAfterAgentMixed(unittest.TestCase):
    """测试 _route_after_agent 的混合调用路由。"""

    def test_pure_subagent_returns_sends(self):
        """纯子代理调用返回 list[Send]。"""
        from langgraph.types import Send
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("dispatch_subagent_lg", {"agent_name": "coder", "task": "read"}, "tc1"),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(s, Send) for s in result))

    def test_no_tool_calls_returns_end(self):
        """无 tool_calls 返回 END。"""
        from langgraph.graph import END
        ai_msg = AIMessage(content="done")
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, END)

    def test_dangerous_tools_route_to_interrupt(self):
        """危险工具路由到 interrupt_approval。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[_make_tool_call("write_file", {"path": "a.py", "content": "x"})],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, "interrupt_approval")

    def test_normal_tools_route_to_tools(self):
        """普通工具路由到 tools。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[_make_tool_call("read_file", {"path": "a.py"})],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, "tools")

    def test_pure_update_todos_routes_to_todos_inline(self):
        """只有 update_todos 调用时路由到 todos_inline（T2 优化）。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[_make_tool_call("update_todos", {"todos": "[]"})],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, "todos_inline")

    def test_update_todos_with_normal_tools_routes_to_tools(self):
        """update_todos + 普通工具混合时路由到 tools（不走内联）。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("update_todos", {"todos": "[]"}),
                _make_tool_call("read_file", {"path": "a.py"}),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, "tools")

    def test_update_todos_with_dangerous_tools_routes_to_interrupt(self):
        """update_todos + 危险工具混合时路由到 interrupt_approval。"""
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                _make_tool_call("update_todos", {"todos": "[]"}),
                _make_tool_call("edit_file", {"path": "a.py", "replacements": []}),
            ],
        )
        state = _make_state(messages=[ai_msg])
        result = _route_after_agent(state)
        self.assertEqual(result, "interrupt_approval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
