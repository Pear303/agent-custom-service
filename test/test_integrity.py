"""测试 _ensure_tool_call_integrity 对 reasoning_content 的处理。"""
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from agent_core.context_view import _ensure_tool_call_integrity

# 场景1: AIMessage 有 tool_calls 但没有对应的 ToolMessage
msgs = [
    SystemMessage(content='system'),
    HumanMessage(content='test'),
    AIMessage(content='', tool_calls=[{'name': 'run_command', 'args': {'command': 'ls'}, 'id': 'call_00_9BfNVP5o'}], additional_kwargs={'reasoning_content': 'thinking...'}),
    AIMessage(content='done'),
]

result = _ensure_tool_call_integrity(msgs)
print(f'Scenario 1 - Input: {len(msgs)} msgs, Output: {len(result)} msgs')
for m in result:
    if isinstance(m, AIMessage) and m.tool_calls:
        print(f'  AIMessage: tool_calls={m.tool_calls}, has_reasoning={bool(m.additional_kwargs.get("reasoning_content"))}')
    else:
        print(f'  {type(m).__name__}: {str(m.content)[:30]}')

# 场景2: 模拟 ContextView 的 _build_with_budget 可能产生的结果
# AIMessage(tool_calls) 在 must_keep 中（因为有 reasoning_content），
# 但对应的 ToolMessage 不在 must_keep 中
msgs2 = [
    SystemMessage(content='system'),
    HumanMessage(content='test'),
    AIMessage(content='', tool_calls=[{'name': 'run_command', 'args': {'command': 'ls'}, 'id': 'call_00_abc'}], additional_kwargs={'reasoning_content': 'thinking...'}),
    # ToolMessage 被裁掉了
    AIMessage(content='final answer'),
]

result2 = _ensure_tool_call_integrity(msgs2)
print(f'\nScenario 2 - Input: {len(msgs2)} msgs, Output: {len(result2)} msgs')
for m in result2:
    if isinstance(m, AIMessage) and m.tool_calls:
        print(f'  AIMessage: tool_calls={m.tool_calls}')
    else:
        print(f'  {type(m).__name__}: {str(m.content)[:30]}')

# 场景3: 检查 _build_tool_call_groups 是否正确识别了 tool_call 组
from agent_core.context_view import _build_tool_call_groups
msgs3 = [
    SystemMessage(content='system'),
    HumanMessage(content='test'),
    AIMessage(content='', tool_calls=[{'name': 'run_command', 'args': {'command': 'ls'}, 'id': 'call_00_abc'}]),
    ToolMessage(content='output', tool_call_id='call_00_abc', name='run_command'),
    AIMessage(content='done'),
]
groups = _build_tool_call_groups(msgs3)
print(f'\nScenario 3 - Groups: {groups}')
print(f'  Group 0 indices: {groups[0] if groups else "none"}')
