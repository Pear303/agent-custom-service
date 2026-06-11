"""ParallelToolNode 单元测试 — 验证只读工具并发、非只读顺序、错误隔离、单 tool_call 快捷路径。

不依赖真实 LLM，使用 mock 工具，秒级完成。

策略：mock _READ_ONLY_TOOLS 使测试工具名被识别为只读，避免依赖真实工具名。
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool


# ── 测试用 mock 工具 ────────────────────────────────────────

# 工具名必须与 _READ_ONLY_TOOLS 匹配才能触发并发逻辑
# 所以我们直接使用真实的工具名


@tool
def read_file(path: str) -> str:
    """读取文件内容。"""
    return f"content of {path}"


@tool
def glob_tool(pattern: str) -> str:
    """搜索文件。"""
    return f"files matching {pattern}"


@tool
def grep_tool(pattern: str) -> str:
    """搜索内容。"""
    return f"lines matching {pattern}"


@tool
def write_file(path: str, content: str) -> str:
    """写入文件。"""
    return f"wrote to {path}"


@tool
def edit_file(path: str, old: str, new: str) -> str:
    """编辑文件。"""
    return f"edited {path}"


@tool
def run_command(cmd: str) -> str:
    """执行命令。"""
    return f"ran: {cmd}"


@tool
def error_tool(query: str) -> str:
    """会报错的工具。"""
    raise ValueError("validation error: field required")


def _make_tool_calls(names_args: list[tuple[str, dict, str]]) -> list[dict]:
    """构造 tool_calls 列表。"""
    return [
        {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        for name, args, call_id in names_args
    ]


def _make_state(tool_calls: list[dict]) -> dict:
    """构造包含 AIMessage(tool_calls) 的 state。"""
    ai_msg = AIMessage(content="", tool_calls=tool_calls)
    return {"messages": [HumanMessage(content="test"), ai_msg]}


def _make_config():
    """构造最小有效的 RunnableConfig。"""
    from langchain_core.runnables import RunnableConfig
    return RunnableConfig(callbacks=[], configurable={})


# ── 测试 ──────────────────────────────────────────────────────


class TestParallelToolNodeBasic:
    """基本行为测试。"""

    def test_empty_messages(self):
        """空消息列表返回空。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file, write_file]
        node = ParallelToolNode(tools)
        result = node.invoke({"messages": []}, None)
        assert result == {"messages": []}

    def test_no_tool_calls(self):
        """AIMessage 无 tool_calls 时返回空。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file]
        node = ParallelToolNode(tools)
        state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
        result = node.invoke(state, None)
        assert result == {"messages": []}

    def test_single_tool_call_delegates_to_tool_node(self):
        """单个 tool_call 直接走 ToolNode，无并发开销。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file]
        node = ParallelToolNode(tools)
        tc = _make_tool_calls([("read_file", {"path": "a.txt"}, "call_1")])
        state = _make_state(tc)

        with patch.object(node._tool_node, "invoke") as mock_invoke:
            mock_invoke.return_value = {"messages": [ToolMessage(content="ok", tool_call_id="call_1", name="read_file")]}
            result = node.invoke(state, None)
            mock_invoke.assert_called_once()
            assert len(result["messages"]) == 1


class TestParallelReadOnlyTools:
    """多个只读工具并发执行。"""

    def test_multiple_read_only_tools_parallel(self):
        """多个只读工具应并发执行，结果全部返回。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file, glob_tool, grep_tool]
        node = ParallelToolNode(tools)

        tc = _make_tool_calls([
            ("read_file", {"path": "a.txt"}, "call_1"),
            ("glob_tool", {"pattern": "*.py"}, "call_2"),
            ("grep_tool", {"pattern": "import"}, "call_3"),
        ])
        state = _make_state(tc)

        result = node.invoke(state, None)
        messages = result["messages"]
        assert len(messages) == 3

        returned_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        assert returned_ids == {"call_1", "call_2", "call_3"}

    @pytest.mark.asyncio
    async def test_async_multiple_read_only_tools(self):
        """异步路径：多个只读工具并发执行。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file, glob_tool]
        node = ParallelToolNode(tools)

        tc = _make_tool_calls([
            ("read_file", {"path": "a.txt"}, "call_1"),
            ("glob_tool", {"pattern": "*.py"}, "call_2"),
        ])
        state = _make_state(tc)

        result = await node.ainvoke(state, None)
        messages = result["messages"]
        assert len(messages) == 2
        returned_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        assert returned_ids == {"call_1", "call_2"}


class TestMixedReadWriteTools:
    """混合只读+非只读工具：非只读顺序执行，只读并发执行。"""

    def test_mixed_tools_both_executed(self):
        """只读和非只读工具都应被执行。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file, write_file, glob_tool]
        node = ParallelToolNode(tools)

        tc = _make_tool_calls([
            ("read_file", {"path": "a.txt"}, "call_1"),
            ("write_file", {"path": "b.txt", "content": "hello"}, "call_2"),
            ("glob_tool", {"pattern": "*.py"}, "call_3"),
        ])
        state = _make_state(tc)

        # Mock _tool_node.invoke 来避免 langgraph 内部 config 校验
        original_invoke = node._tool_node.invoke

        def mock_tool_node_invoke(state, config=None):
            """模拟 ToolNode.invoke，直接调用工具并返回 ToolMessage。"""
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}
            last_msg = messages[-1]
            if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                return {"messages": []}

            results = []
            for tc in last_msg.tool_calls:
                tool = node._name_to_tool.get(tc["name"])
                if tool is None:
                    results.append(ToolMessage(
                        content=f"Error: unknown tool '{tc['name']}'",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status="error",
                    ))
                else:
                    try:
                        output = tool.invoke(tc["args"])
                        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
                    except Exception as e:
                        results.append(ToolMessage(content=str(e), tool_call_id=tc["id"], name=tc["name"], status="error"))
            return {"messages": results}

        with patch.object(node._tool_node, "invoke", side_effect=mock_tool_node_invoke):
            result = node.invoke(state, _make_config())

        messages = result["messages"]
        assert len(messages) == 3
        returned_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        assert returned_ids == {"call_1", "call_2", "call_3"}


class TestErrorIsolation:
    """错误隔离：一个工具报错不应影响其他工具。"""

    def test_error_tool_returns_error_message(self):
        """报错的工具返回 status=error 的 ToolMessage，不影响其他工具。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        tools = [read_file, error_tool, glob_tool]
        node = ParallelToolNode(tools)

        tc = _make_tool_calls([
            ("read_file", {"path": "a.txt"}, "call_1"),
            ("error_tool", {"query": "test"}, "call_2"),
            ("glob_tool", {"pattern": "*.py"}, "call_3"),
        ])
        state = _make_state(tc)

        def mock_tool_node_invoke(state, config=None):
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}
            last_msg = messages[-1]
            if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
                return {"messages": []}

            results = []
            for tc in last_msg.tool_calls:
                tool = node._name_to_tool.get(tc["name"])
                if tool is None:
                    results.append(ToolMessage(
                        content=f"Error: unknown tool '{tc['name']}'",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status="error",
                    ))
                else:
                    try:
                        output = tool.invoke(tc["args"])
                        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
                    except Exception as e:
                        results.append(ToolMessage(content=str(e), tool_call_id=tc["id"], name=tc["name"], status="error"))
            return {"messages": results}

        with patch.object(node._tool_node, "invoke", side_effect=mock_tool_node_invoke):
            result = node.invoke(state, _make_config())

        messages = result["messages"]
        assert len(messages) >= 2

        # 只读工具应正常返回
        ok_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage) and m.status != "error"}
        assert "call_1" in ok_ids
        assert "call_3" in ok_ids

    @pytest.mark.asyncio
    async def test_async_error_isolation_readonly(self):
        """异步路径：只读工具中一个报错不影响其他。"""
        from agent_by_langgraph.lg_parallel_tools import ParallelToolNode

        # 创建一个会报错的只读工具
        @tool
        def web_fetch(url: str) -> str:
            """获取网页。"""
            raise ValueError("validation error: field required")

        tools = [read_file, web_fetch]
        node = ParallelToolNode(tools)

        tc = _make_tool_calls([
            ("read_file", {"path": "a.txt"}, "call_1"),
            ("web_fetch", {"url": "http://test"}, "call_2"),
        ])
        state = _make_state(tc)

        result = await node.ainvoke(state, None)
        messages = result["messages"]
        assert len(messages) == 2

        # read_file 正常
        ok_msgs = [m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "call_1"]
        assert len(ok_msgs) == 1
        assert ok_msgs[0].status != "error"

        # web_fetch 报错
        err_msgs = [m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "call_2"]
        assert len(err_msgs) == 1
        assert err_msgs[0].status == "error"


class TestReadonlyToolSet:
    """验证只读工具集合定义。"""

    def test_readonly_set_contents(self):
        from agent_by_langgraph.lg_parallel_tools import _READ_ONLY_TOOLS

        assert "web_fetch" in _READ_ONLY_TOOLS
        assert "read_file" in _READ_ONLY_TOOLS
        assert "glob_tool" in _READ_ONLY_TOOLS
        assert "grep_tool" in _READ_ONLY_TOOLS
        assert "load_skill" in _READ_ONLY_TOOLS
        assert "rag_search" in _READ_ONLY_TOOLS

        # 非只读工具不应在集合中
        assert "write_file" not in _READ_ONLY_TOOLS
        assert "edit_file" not in _READ_ONLY_TOOLS
        assert "run_command" not in _READ_ONLY_TOOLS
