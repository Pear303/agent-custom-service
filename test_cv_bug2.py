"""测试并行 tool_calls 场景下的 ContextView 裁剪 bug。"""
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from agent_core.context_view import ContextView, _build_tool_call_groups

# 场景：一个 AIMessage 有多个并行 tool_calls
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="帮我创建计算器"),
]

# 前面有很多轮调用
for i in range(8):
    tc_id = f"call_{i:04d}"
    messages.append(AIMessage(
        content="",
        tool_calls=[{"name": "run_command", "args": {"command": f"echo {i}"}, "id": tc_id}],
    ))
    messages.append(ToolMessage(content=f"output {i}: " + "x" * 40, tool_call_id=tc_id, name="run_command"))

# 一个 AIMessage 有 2 个并行 tool_calls
messages.append(AIMessage(
    content="",
    tool_calls=[
        {"name": "run_command", "args": {"command": "ls"}, "id": "call_0008a"},
        {"name": "run_command", "args": {"command": "pwd"}, "id": "call_0008b"},
    ],
    additional_kwargs={"reasoning_content": "I need to check the directory structure... " * 10},
))
messages.append(ToolMessage(content="file1.txt\nfile2.txt\n" + "x" * 40, tool_call_id="call_0008a", name="run_command"))
messages.append(ToolMessage(content="/home/user/project\n" + "x" * 40, tool_call_id="call_0008b", name="run_command"))

# 后面还有几轮
for i in range(9, 12):
    tc_id = f"call_{i:04d}"
    messages.append(AIMessage(
        content="",
        tool_calls=[{"name": "run_command", "args": {"command": f"echo {i}"}, "id": tc_id}],
    ))
    messages.append(ToolMessage(content=f"output {i}: " + "x" * 40, tool_call_id=tc_id, name="run_command"))

messages.append(AIMessage(content="计算器已创建完成"))

print(f"Total messages: {len(messages)}")

# 检查 groups
groups = _build_tool_call_groups(messages)
print(f"Tool call groups: {len(groups)}")
for gi, g in enumerate(groups):
    print(f"  Group {gi}: indices={g}")
    for idx in sorted(g):
        msg = messages[idx]
        if isinstance(msg, AIMessage):
            tc_ids = [tc['id'] for tc in msg.tool_calls]
            print(f"    [{idx}] AIMessage: tool_call_ids={tc_ids}, has_reasoning={bool(msg.additional_kwargs.get('reasoning_content'))}")
        elif isinstance(msg, ToolMessage):
            print(f"    [{idx}] ToolMessage: tool_call_id={msg.tool_call_id}")

# 用小 budget 触发裁剪
cv = ContextView(max_context_tokens=300, target_ratio=0.8, min_window=3, keep_recent_groups=2)
view, pruned = cv.build_view(messages)

print(f"\nAfter ContextView: {len(view)} msgs (pruned {len(messages) - len(view)})")
print(f"Pruned groups: {len(pruned)}")

# 检查完整性
tc_ids = set()
tm_ids = set()
for m in view:
    if isinstance(m, AIMessage) and m.tool_calls:
        for tc in m.tool_calls:
            if tc.get("id"):
                tc_ids.add(tc["id"])
    if isinstance(m, ToolMessage):
        tm_ids.add(m.tool_call_id)

missing = tc_ids - tm_ids
orphan = tm_ids - tc_ids
print(f"\ntool_call_ids in view: {tc_ids}")
print(f"tool_message_ids in view: {tm_ids}")
if missing:
    print(f"\n*** BUG: Missing ToolMessages for: {missing} ***")
    for m in view:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("id") in missing:
                    print(f"  Problem AIMessage: tool_call={tc['name']}, id={tc['id']}")
                    print(f"  has_reasoning={bool(m.additional_kwargs.get('reasoning_content'))}")
                    print(f"  all tool_calls in this msg: {[(t['name'], t['id']) for t in m.tool_calls]}")
else:
    print("No missing ToolMessages - integrity OK")
