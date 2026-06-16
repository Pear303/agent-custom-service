# LangGraph Demo 脚本缺陷修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `.sisyphus/langgraph/test.py` 中已识别的 10 个缺陷和优化点，使 demo 输出准确、注释清晰、代码规范。

**Architecture:** 单文件修改，按 Part 1-6 顺序逐个修复，每次修复后运行验证。

**Tech Stack:** Python 3.10+, LangGraph, LangChain

---

## 修改文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `.sisyphus/langgraph/test.py` | 修改 | 唯一需要修改的文件 |

---

### Task 1: 修复 Part 2 — step_count 注释与输出不一致

**Files:**
- Modify: `.sisyphus/langgraph/test.py:316-323` (step_c 节点函数)
- Modify: `.sisyphus/langgraph/test.py:363-368` (step_count 注释块)

**问题：** `step_c` 读取 `state["step_count"]` 时值为 2（因为自身 +1 还未写入），输出"共经过 2 步"，但最终 `step_count` 为 3。注释说"三个节点各加一次：0→1→2→3"是对的，但没解释为什么节点 C 输出的是 2 而非 3。

- [ ] **Step 1: 修改 step_c 节点，让输出值与最终 step_count 一致**

将 step_c 的 content 从读取当前 state 改为读取 +1 后的值：

```python
    def step_c(state) -> dict:
        new_count = state["step_count"] + 1
        return {
            "messages": [AIMessage(content=f"[C] 处理完成！共经过 {new_count} 步。")],
            "step_count": new_count,
        }
```

- [ ] **Step 2: 更新注释，解释 step_count 的"先读后写"时序**

将原注释块替换为：

```python
    # [语法] result["step_count"]: 从最终 state 中读取 step_count
    # 【思考】为什么 step_count 是 3？
    #   三个节点各加一次：0 → 1 (step_a) → 2 (step_b) → 3 (step_c)
    #   但 messages 有 4 条：HumanMsg + A + B + C
    #   所以 step_count 不是消息数，而是经过的节点数。
    #
    # 【注意】节点函数中读取 state["step_count"] 时，自身返回的 +1 还未写入。
    #   所以 step_c 内部读到的值是 2（step_b 的输出），需要手动 +1 才是 3。
    #   这是 LangGraph "先读后写"的时序：节点返回的 dict 在节点执行完后才合并到 state。
```

- [ ] **Step 3: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 2`

Expected: 输出中 `[C] 处理完成！共经过 3 步。`，`【最终 step_count】: 3`，两者一致。

---

### Task 2: 修复 Part 3 — 数学分支输出与输入无关

**Files:**
- Modify: `.sisyphus/langgraph/test.py:417` (handle_math 函数)

**问题：** 输入 `128 + 256 等于多少`，路由正确匹配到 math 分支，但 `handle_math` 返回硬编码 `[数学] 3.14 x 2 = 6.28`，与输入无关，误导学习者。

- [ ] **Step 1: 修改 handle_math，做简单的关键词匹配回复**

```python
    def handle_math(state) -> dict:
        last = state["messages"][-1] if state["messages"] else None
        content = last.content if isinstance(last, HumanMessage) else ""
        # 简单模拟：提取输入中的数字和运算符做基础计算
        import re
        match = re.search(r'([\d.]+)\s*([+\-*/])\s*([\d.]+)', content)
        if match:
            a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
            ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                   "*": lambda x, y: x * y, "/": lambda x, y: x / y}
            try:
                result = ops[op](a, b)
                return {"messages": [AIMessage(content=f"[数学] {a} {op} {b} = {result}")]}
            except ZeroDivisionError:
                return {"messages": [AIMessage(content="[数学] 除数不能为零")]}
        return {"messages": [AIMessage(content="[数学] 收到数学问题，但暂不支持该计算格式。")]}
```

- [ ] **Step 2: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 3`

Expected: 输入 `128 + 256 等于多少` 时，回复为 `[数学] 128.0 + 256.0 = 384.0`（而非硬编码的 3.14 x 2）。

---

### Task 3: 修复 Part 4 — MockLLM tool_calls id 硬编码

**Files:**
- Modify: `.sisyphus/langgraph/test.py:596-600` (MockLLM.__init__)
- Modify: `.sisyphus/langgraph/test.py:658-664` (tool_calls id 赋值)

**问题：** 所有工具调用的 `id` 都是硬编码 `"call_mock_1"`。如果扩展为多轮工具调用，同 id 的 ToolMessage 会触发 `add_messages` 的"同 id 覆盖"逻辑，导致前面的 ToolMessage 被覆盖丢失。

- [ ] **Step 1: 在 MockLLM.__init__ 中添加自增计数器**

```python
    class MockLLM:
        def __init__(self):
            self._tools = []
            self._call_counter = 0  # 工具调用 ID 计数器
```

- [ ] **Step 2: 将 tool_calls 中的 id 改为动态生成**

将原来的：
```python
                    "id": "call_mock_1",
```
替换为：
```python
                    "id": f"call_mock_{self._call_counter}",
```

并在 `return AIMessage(...)` 之前加一行自增：
```python
                self._call_counter += 1
                return AIMessage(content="", tool_calls=[{
                    "name": selected.name,
                    "args": {"query": user_text} if selected.name == "search_tool" else {"expr": user_text},
                    "id": f"call_mock_{self._call_counter}",
                    "type": "tool_call",
                }])
```

- [ ] **Step 3: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 4`

Expected: 输出与之前一致（因为当前只有单轮调用），但 id 不再硬编码。

---

### Task 4: 修复 Part 4 — calculator_tool 内部 import 移到函数外

**Files:**
- Modify: `.sisyphus/langgraph/test.py:648-649` (calculator_tool 内部 import)

**问题：** `import operator` 和 `import re` 在 `calculator_tool` 函数内部，每次调用都执行 import 语句（虽然 Python 有模块缓存不会重复加载，但放在函数外更规范）。

- [ ] **Step 1: 删除 calculator_tool 内部的 import 语句**

将：
```python
    @tool
    def calculator_tool(expr: str) -> str:
        """计算数学表达式（仅支持加减乘除）。"""
        import operator
        import re
```
改为：
```python
    @tool
    def calculator_tool(expr: str) -> str:
        """计算数学表达式（仅支持加减乘除）。"""
```

- [ ] **Step 2: 在文件顶部 import 区域添加 operator**

在文件顶部的 import 块中（约第 18-20 行附近），添加：
```python
import operator
```

注意：`re` 已经在文件顶部没有导入，但 `calculator_tool` 和 Task 2 的 `handle_math` 都需要用到 `re`，所以也需要在顶部添加：
```python
import re
```

- [ ] **Step 3: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 4`

Expected: 输出与之前一致，calculator_tool 正常工作。

---

### Task 5: 修复 Part 5 — greet_node 的 chat_count 语义不清

**Files:**
- Modify: `.sisyphus/langgraph/test.py:848` (greet_node 返回值)

**问题：** `greet_node` 返回 `chat_count: 0`，但这是"欢迎轮"不算对话轮次，语义不清。第 1 轮输出 `轮次=0`，第 2 轮输出 `轮次=1`，第 3 轮输出 `轮次=2`，看起来"欢迎轮"不算一轮对话，但注释没有说明。

- [ ] **Step 1: 修改 greet_node，将 chat_count 设为 1 并添加注释**

```python
    def greet_node(state, config=None) -> dict:
        """首轮欢迎节点。chat_count 从 1 开始计数（欢迎轮也算第 1 轮）。"""
        return {"messages": [AIMessage(content="欢迎来到有状态对话！请告诉我你的名字。")], "chat_count": 1}
```

- [ ] **Step 2: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 5`

Expected: 第 1 轮 `轮次=1`，第 2 轮 `轮次=2`，第 3 轮 `轮次=3`。语义更清晰：每轮对话都计数。

---

### Task 6: 修复 Part 6 — 移除多余的 dispatcher 节点

**Files:**
- Modify: `.sisyphus/langgraph/test.py:1027-1029` (dispatcher 函数)
- Modify: `.sisyphus/langgraph/test.py:1082-1089` (图的构建和边)

**问题：** `dispatcher` 节点只是透传 `tasks`，没有任何处理逻辑。条件边 `route_to_workers` 可以直接从 `START` 出发，省掉一个节点，让图更简洁。

- [ ] **Step 1: 删除 dispatcher 函数**

删除以下代码：
```python
    def dispatcher(state) -> dict:
        """入口节点：透传 tasks。"""
        return {"tasks": state.get("tasks", [])}
```

- [ ] **Step 2: 修改图的构建，将条件边直接从 START 出发**

将原来的：
```python
    builder = StateGraph(MainState)
    builder.add_node("dispatcher", dispatcher)
    builder.add_node("sub_worker", sub_worker)
    builder.add_node("aggregator", aggregator)
    builder.add_edge(START, "dispatcher")
    builder.add_conditional_edges("dispatcher", route_to_workers)
    builder.add_edge("sub_worker", "aggregator")
    builder.add_edge("aggregator", END)
```

改为：
```python
    builder = StateGraph(MainState)
    builder.add_node("sub_worker", sub_worker)
    builder.add_node("aggregator", aggregator)
    # 条件边直接从 START 出发，省去透传节点
    builder.add_conditional_edges(START, route_to_workers)
    builder.add_edge("sub_worker", "aggregator")
    builder.add_edge("aggregator", END)
```

- [ ] **Step 3: 更新流程注释**

将原来的流程注释：
```python
    # [流程] START → dispatcher → (Send x N) → sub_worker → aggregator → END
```
改为：
```python
    # [流程] START → (Send x N) → sub_worker → aggregator → END
    # [注意] 条件边直接从 START 出发，route_to_workers 返回 list[Send]，
    #   每个 Send 创建独立的 sub_worker 执行上下文。
```

- [ ] **Step 4: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 6`

Expected: 并行执行结果与之前一致，4 个任务全部完成。

---

### Task 7: 修复 Part 6 — import asyncio 移到函数顶部

**Files:**
- Modify: `.sisyphus/langgraph/test.py:1013` (worker_node 内部 import asyncio)
- Modify: `.sisyphus/langgraph/test.py:1096` (测试代码内部 import asyncio)

**问题：** `import asyncio` 在 `worker_node` 和测试代码中各出现一次，应提到 `demo_part6` 函数顶部。

- [ ] **Step 1: 在 demo_part6 函数顶部添加 import asyncio**

在 `demo_part6` 函数体开头（`_print(...)` 之前），添加：
```python
    import asyncio
```

- [ ] **Step 2: 删除 worker_node 内部的 import asyncio**

将：
```python
        import asyncio
        await asyncio.sleep(delay)
```
改为：
```python
        await asyncio.sleep(delay)
```

- [ ] **Step 3: 删除测试代码内部的 import asyncio**

将：
```python
    import asyncio
    asyncio.run(graph.ainvoke({
```
改为：
```python
    asyncio.run(graph.ainvoke({
```

- [ ] **Step 4: 运行验证**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py" 6`

Expected: 输出与之前一致。

---

### Task 8: 全量回归验证

**Files:**
- 无文件修改，仅运行验证

- [ ] **Step 1: 运行全部 6 个 Part**

Run: `python "f:\XiangMu\AGENT\langchain-test\agent-custom-service\.sisyphus\langgraph\test.py"`

Expected: 所有 6 个 Part 正常执行，无报错，输出语义正确：
- Part 2: `[C] 处理完成！共经过 3 步。`，`step_count: 3`
- Part 3: 数学输入 `128 + 256` 输出 `128.0 + 256.0 = 384.0`
- Part 4: Agent 循环正常
- Part 5: 第 1 轮 `轮次=1`，第 3 轮 `轮次=3`
- Part 6: 4 个并行任务全部完成

- [ ] **Step 2: 检查无 import 错误或运行时异常**

确认输出中无 `ImportError`、`NameError`、`KeyError` 等异常。

---

## 修复优先级总览

| Task | 优先级 | 问题 | 影响范围 |
|------|--------|------|----------|
| Task 1 | 高 | step_count 注释与输出不一致 | Part 2 |
| Task 2 | 高 | 数学分支输出与输入无关 | Part 3 |
| Task 3 | 高 | tool_calls id 硬编码（潜在 Bug） | Part 4 |
| Task 4 | 中 | calculator_tool 内部 import | Part 4 |
| Task 5 | 中 | greet_node chat_count 语义不清 | Part 5 |
| Task 6 | 中 | dispatcher 节点多余 | Part 6 |
| Task 7 | 低 | import asyncio 位置 | Part 6 |
| Task 8 | - | 全量回归验证 | 全部 |
