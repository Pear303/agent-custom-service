"""精确复现 ContextView 裁剪导致 tool_call 完整性被破坏的 bug。"""
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from agent_core.context_view import ContextView, _build_tool_call_groups, _ensure_tool_call_integrity

# 构造一个场景：
# - 有很多 run_command 调用（模拟真实场景）
# - 最后一个 AIMessage 有 reasoning_content + tool_calls
# - 对应的 ToolMessage 在 token budget 不够时被裁掉
# - _compute_must_keep 因为 _has_reasoning 保留了 AIMessage
# - 但 group 扩展时 ToolMessage 可能不在 group 中

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="帮我创建计算器"),
]

# 添加 15 轮 run_command 调用，每轮有较大 content
for i in range(15):
    tc_id = f"call_{i:04d}"
    has_reasoning = (i == 14)  # 最后一个有 reasoning
    messages.append(AIMessage(
        content="" if i < 14 else "",
        tool_calls=[{"name": "run_command", "args": {"command": f"echo {i}"}, "id": tc_id}],
        additional_kwargs={"reasoning_content": f"thinking step {i}... this is a long reasoning content that takes up tokens " * 5} if has_reasoning else {},
    ))
    messages.append(ToolMessage(content=f"output from command {i}: " + "x" * 50, tool_call_id=tc_id, name="run_command"))

# 最后一条 AI 回复
messages.append(AIMessage(content="计算器已创建完成"))

print(f"Total messages: {len(messages)}")
print(f"Total tokens estimate: {sum(len(str(m.content)) for m in messages)}")

# 检查 groups
groups = _build_tool_call_groups(messages)
print(f"\nTool call groups: {len(groups)}")
for gi, g in enumerate(groups):
    print(f"  Group {gi}: indices={g}")
    for idx in g:
        msg = messages[idx]
        if isinstance(msg, AIMessage):
            print(f"    [{idx}] AIMessage: tool_calls={[(tc['name'], tc['id']) for tc in msg.tool_calls]}")
        elif isinstance(msg, ToolMessage):
            print(f"    [{idx}] ToolMessage: tool_call_id={msg.tool_call_id}")

# 用很小的 budget 来触发裁剪
cv = ContextView(max_context_tokens=200, target_ratio=0.8, min_window=3, keep_recent_groups=2)
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
print(f"\ntool_call_ids: {len(tc_ids)}")
print(f"tool_message_ids: {len(tm_ids)}")
print(f"Missing ToolMessages: {len(missing)} - {list(missing)[:5]}")
print(f"Orphan ToolMessages: {len(orphan)} - {list(orphan)[:5]}")

if missing:
    print("\n*** BUG REPRODUCED: AIMessage(tool_calls) without ToolMessage ***")
    for m in view:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("id") in missing:
                    print(f"  Problem: AIMessage tool_call={tc['name']}, id={tc['id']}")
                    print(f"  has_reasoning={bool(m.additional_kwargs.get('reasoning_content'))}")
