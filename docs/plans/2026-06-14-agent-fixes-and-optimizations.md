# Agent 缺陷修复与优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Agent 运行中发现的 3 个缺陷和 7 个优化点，提升运行稳定性和效率

**Architecture:** 按优先级分 4 批实施：P0 运行时缺陷 → P1 效率优化 → P2 代码质量 → P3 边缘场景。每批内的任务互相独立，可并行执行。

**Tech Stack:** Python 3.11+, LangGraph, LangChain, DeepSeek API, SQLite (aiosqlite)

---

## 文件结构

| 操作 | 文件路径 | 职责 |
|------|----------|------|
| 修改 | `agent_by_langgraph/lg_tools.py` | 修复 fallback 同步调用、write_file 返回值增强 |
| 修改 | `agent_by_langgraph/lg_graph.py` | 阶段推断优化、反思跳过、偏差检测拆分、结论提取增强 |
| 修改 | `agent_by_langgraph/lg_agent.py` | Checkpointer 初始化优化、编码修复统一 |
| 修改 | `agent_by_langgraph/level_router.py` | 短输入快速路径 |
| 修改 | `agent_lg.py` | 编码修复统一入口 |
| 修改 | `agent_core/tools/context_vars.py` | write_file 返回值增强 |
| 新建 | `test/test_lg_fixes.py` | 修复的单元测试 |

---

## P0: 运行时缺陷（必须修复）

### Task 1: 修复 dispatch_subagent_lg fallback 同步调用

**问题:** `lg_tools.py:155` 的 fallback 路径使用 `subgraph.invoke(sub_input)` 同步调用，但子图节点是 `async def`，会抛出 `NotImplementedError`。

**Files:**
- 修改: `agent_by_langgraph/lg_tools.py:140-170`

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_dispatch_subagent_fallback_uses_async_invoke():
    """fallback 路径应使用 ainvoke 而非同步 invoke，避免 NotImplementedError"""
    from agent_by_langgraph.lg_tools import dispatch_subagent_lg
    from agent_core.subagents.registry import SubagentSpec

    mock_llm = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get_spec.return_value = SubagentSpec(
        name="quick_helper",
        display_name="Quick Helper",
        description="test",
        max_turns=3,
        is_rag=False,
    )
    mock_registry.list_names.return_value = ["quick_helper"]

    # 模拟异步上下文不可用 → 触发 fallback
    mock_subgraph = MagicMock()
    mock_subgraph.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="子代理结果", tool_calls=[])]
    })
    mock_subgraph.invoke = MagicMock(side_effect=NotImplementedError("sync not supported"))

    with patch("agent_by_langgraph.lg_tools.get_subagent_graph", return_value=mock_subgraph):
        with patch("agent_by_langgraph.lg_tools._ctx_sub_reg", MagicMock(get=MagicMock(return_value=mock_registry))):
            with patch("agent_by_langgraph.lg_tools._ctx_llm_ref", MagicMock(get=MagicMock(return_value=mock_llm))):
                # dispatch_subagent_lg 是 async 函数
                result = await dispatch_subagent_lg.ainvoke({
                    "agent_name": "quick_helper",
                    "task": "测试任务",
                })
                # ainvoke 应被调用，而非 invoke
                mock_subgraph.ainvoke.assert_called_once()
                mock_subgraph.invoke.assert_not_called()
                assert "Error" not in result or "子代理结果" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_dispatch_subagent_fallback_uses_async_invoke -v`
Expected: FAIL — 当前 fallback 使用同步 invoke

- [ ] **Step 3: 修复 fallback 路径，改用 ainvoke**

在 `agent_by_langgraph/lg_tools.py` 中，将 fallback 路径从同步 `invoke` 改为异步 `ainvoke`：

```python
# lg_tools.py 第 155 行附近，将:
    result = subgraph.invoke(sub_input)  # fallback 路径无 checkpointer，不传 config

# 改为:
    result = await subgraph.ainvoke(sub_input)  # fallback 路径无 checkpointer，不传 config
```

同时，`dispatch_subagent_lg` 函数本身已经是 `async def`（通过 `@tool` 装饰器定义），所以 `await` 是合法的。如果函数签名不是 async，需要确认 `@tool` 装饰器是否支持异步。查看当前定义：

```python
# 确认 dispatch_subagent_lg 是 async def
@tool
async def dispatch_subagent_lg(agent_name: str, task: str) -> str:
```

如果已经是 `async def`，直接 `await subgraph.ainvoke(sub_input)` 即可。如果不是，需要将函数改为 `async def`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_dispatch_subagent_fallback_uses_async_invoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_tools.py test/test_lg_fixes.py
git commit -m "fix: dispatch_subagent_lg fallback 路径改用 ainvoke，避免 NotImplementedError"
```

---

### Task 2: 修复 Agent 重复写操作 — write_file 返回值增强

**问题:** Agent 用 `write_file` 创建文件后，又用 `run_command` 执行 `echo ... > file` 重复写入。原因是 `write_file` 的返回值没有明确提示"文件已创建成功，无需再用 shell 命令重写"。

**Files:**
- 修改: `agent_core/tools/context_vars.py` 中 `write_file` 函数的返回值
- 修改: `agent_by_langgraph/lg_agent.py` 中系统提示词的"写后必审"约束

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py 追加
def test_write_file_return_includes_no_rewrite_hint():
    """write_file 返回值应包含'无需再用 shell 命令重写'提示"""
    from agent_core.tools import write_file
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # 设置工作目录
        from agent_core.tools import set_workspace
        set_workspace(tmpdir)

        result = write_file.invoke({"path": "test_hint.py", "content": "print('hello')"})
        assert "无需" in result or "不要再用" in result or "shell" in result.lower() or "已成功" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_write_file_return_includes_no_rewrite_hint -v`
Expected: FAIL — 当前返回值无此提示

- [ ] **Step 3: 修改 write_file 返回值**

在 `agent_core/tools/context_vars.py` 中找到 `write_file` 函数，在成功返回时追加提示：

```python
# 在 write_file 函数的返回值中，将类似:
    return f"Successfully wrote {len(content)} chars to {path}"

# 改为:
    return f"Successfully wrote {len(content)} chars to {path}\n[提示] 文件已成功创建，无需再用 run_command/echo 重写此文件。如需修改请用 edit_file。"
```

具体位置需根据实际代码调整。如果返回值格式不同，在末尾追加换行和提示即可。

- [ ] **Step 4: 增强系统提示词中的约束**

在 `agent_by_langgraph/lg_agent.py` 的 `cwd_and_constraints` 字符串中，将"写后必审"约束增强：

```python
# 将:
    f"- **写后必审**: write_file 后用 read_file 检查内容\n"

# 改为:
    f"- **写后必审**: write_file 后用 read_file 检查内容，不要用 run_command 重写同一文件\n"
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_write_file_return_includes_no_rewrite_hint -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agent_core/tools/context_vars.py agent_by_langgraph/lg_agent.py test/test_lg_fixes.py
git commit -m "fix: write_file 返回值增加防重写提示，系统提示词增强约束"
```

---

### Task 3: 修复简单创建任务的阶段推断 — gather→all 回退循环

**问题:** "创建一个 hello.py" 这类简单创建任务，规划器推断阶段为 `gather`，但 gather 阶段只绑定只读工具，Agent 无法执行 `write_file`，导致停滞 2 次后回退到 `all`，浪费 2 轮迭代。

**Files:**
- 修改: `agent_by_langgraph/lg_graph.py:860-910` (`_extract_phase` 函数)

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py 追加
def test_extract_phase_simple_create_task():
    """简单创建任务（单文件、短描述）应直接推断为 modify，而非 gather"""
    from agent_by_langgraph.lg_graph import _extract_phase

    # 简单创建任务：步骤 ≤ 2 且涉及创建
    simple_create_plan = "1. 创建 hello.py → 执行者: 自己, 工具: write_file\n[阶段: gather]"
    # 即使规划器输出 gather，对简单创建任务应推断为 modify
    # 当前行为：返回 "gather"（从标记提取）
    # 期望行为：简单创建任务返回 "modify"
    result = _extract_phase(simple_create_plan)
    assert result == "modify", f"简单创建任务应推断为 modify，实际为 {result}"

def test_extract_phase_simple_create_no_marker():
    """无阶段标记的简单创建任务应推断为 modify"""
    from agent_by_langgraph.lg_graph import _extract_phase

    plan = "1. 创建 hello.py → 执行者: 自己, 工具: write_file\n2. 运行验证 → 执行者: 自己, 工具: run_command"
    result = _extract_phase(plan)
    assert result == "modify", f"简单创建任务应推断为 modify，实际为 {result}"

def test_extract_phase_complex_create_still_gather():
    """复杂创建任务（步骤 ≥ 3）仍应推断为 gather"""
    from agent_by_langgraph.lg_graph import _extract_phase

    plan = (
        "1. 查看目录结构 → 执行者: 自己, 工具: glob_tool\n"
        "2. 阅读已有文件 → 执行者: 自己, 工具: read_file\n"
        "3. 创建主程序 → 执行者: 自己, 工具: write_file\n"
        "4. 创建测试 → 执行者: 自己, 工具: write_file\n"
        "5. 运行测试 → 执行者: 自己, 工具: run_command\n"
    )
    result = _extract_phase(plan)
    assert result == "gather", f"复杂创建任务应推断为 gather，实际为 {result}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_extract_phase_simple_create_task test/test_lg_fixes.py::test_extract_phase_simple_create_no_marker test/test_lg_fixes.py::test_extract_phase_complex_create_still_gather -v`
Expected: 前两个 FAIL，第三个 PASS

- [ ] **Step 3: 修改 `_extract_phase` 函数**

在 `agent_by_langgraph/lg_graph.py` 的 `_extract_phase` 函数中，增加简单创建任务的快速路径：

```python
def _extract_phase(plan_text: str) -> str:
    """从规划输出中提取执行阶段。

    查找 [阶段: xxx] 标记，未找到时根据内容推断：
    - 简单创建任务（步骤 ≤ 2 且涉及写文件）→ "modify"（直接动手，无需先收集信息）
    - 复杂创建任务（步骤 ≥ 3）→ "gather"（先收集信息再动手）
    - 纯修改/编辑类（已有代码需修改）→ "modify"
    - 搜索/查找/阅读类 → "gather"
    - 默认 → "gather"
    """
    match = re.search(r'\[阶段[:：]\s*(gather|modify|verify)\]', plan_text, re.IGNORECASE)
    if match:
        marker_phase = match.group(1).lower()
        # 修正：即使规划器标记为 gather，如果步骤 ≤ 2 且涉及创建/写文件，
        # 应直接推断为 modify，避免 gather 阶段无写工具导致停滞
        step_count = len(re.findall(r'^\d+\.\s', plan_text, re.MULTILINE))
        has_create_keyword = bool(re.search(
            r'(创建|写|编写|生成|新建).*(文件|脚本|程序|模块)',
            plan_text, re.IGNORECASE
        ))
        if marker_phase == "gather" and step_count <= 2 and has_create_keyword:
            logger.info("[阶段修正] 简单创建任务 gather → modify: 步骤=%d", step_count)
            return "modify"
        return marker_phase

    # 推断：步骤数和内容综合判断
    step_count = len(re.findall(r'^\d+\.\s', plan_text, re.MULTILINE))

    # 简单创建任务（步骤 ≤ 2 且涉及创建/写文件）→ modify
    if step_count <= 2:
        if re.search(r'(创建|写|编写|生成|新建).*(文件|脚本|程序|模块)', plan_text, re.IGNORECASE):
            return "modify"

    # 推断：开发/创建/构建类 → gather（先收集信息再动手）
    if re.search(r'(开发|创建|构建|搭建|新建|实现|设计|编写).*(应用|项目|程序|系统|工具|脚本|网页|网站|服务)', plan_text):
        return "gather"
    # 推断：步骤数 >= 3 → gather（复杂任务先收集信息）
    if step_count >= 3:
        return "gather"
    # 推断：纯修改/编辑（不涉及新建）→ modify
    if re.search(r'(修改|编辑|重写|删除|添加代码|修复)', plan_text):
        return "modify"
    # 推断：搜索关键词 → gather
    if re.search(r'(搜索|查找|阅读|分析|检查)', plan_text):
        return "gather"

    return "gather"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_extract_phase_simple_create_task test/test_lg_fixes.py::test_extract_phase_simple_create_no_marker test/test_lg_fixes.py::test_extract_phase_complex_create_still_gather -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_graph.py test/test_lg_fixes.py
git commit -m "fix: 简单创建任务阶段推断从 gather 改为 modify，避免停滞回退循环"
```

---

## P1: 效率优化

### Task 4: LevelRouter 短输入快速路径 — 跳过 LLM 分类

**问题:** 对短输入（如"帮我查看 Python 文件"），LevelRouter 额外消耗一次 LLM 调用做意图分类，但这类简单查询总是级别 3。

**Files:**
- 修改: `agent_by_langgraph/level_router.py:168-200` (`route` 方法)

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py 追加
def test_level_router_short_input_skips_llm():
    """短输入（≤30 字）应直接返回级别 3，不调用 LLM"""
    from agent_by_langgraph.level_router import LevelRouter
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(side_effect=AssertionError("不应调用 LLM"))
    router = LevelRouter(llm=mock_llm)

    config = router.route("帮我查看Python文件")
    assert config.level == 3  # TaskLevel.SCRIPT
    mock_llm.invoke.assert_not_called()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_level_router_short_input_skips_llm -v`
Expected: FAIL — 短输入仍会调用 LLM

- [ ] **Step 3: 修改 `route` 方法，增加短输入快速路径**

在 `agent_by_langgraph/level_router.py` 的 `route` 方法中，在关键词匹配之后、LLM 分类之前，增加短输入快速路径：

```python
def route(self, user_input: str) -> LevelConfig:
    # 步骤1: 关键词快速匹配
    level = self._keyword_match(user_input)
    if level is not None:
        logger.info("[LevelRouter] 关键词匹配 → 级别 %d (%s)", level, _LEVEL_CONFIGS[level].label)
        return _LEVEL_CONFIGS[level]

    # 步骤1.5: 短输入快速路径（≤30 字且无关键词匹配，默认级别 3，跳过 LLM）
    _SHORT_INPUT_THRESHOLD = 30
    if len(user_input) <= _SHORT_INPUT_THRESHOLD:
        default = TaskLevel.SCRIPT
        logger.info("[LevelRouter] 短输入快速路径 → 级别 %d (%s)", default, _LEVEL_CONFIGS[default].label)
        return _LEVEL_CONFIGS[default]

    # 步骤2: LLM 分类（如果可用）
    if self.llm is not None:
        level = self._llm_classify(user_input)
        if level is not None:
            logger.info("[LevelRouter] LLM 分类 → 级别 %d (%s)", level, _LEVEL_CONFIGS[level].label)
            return _LEVEL_CONFIGS[level]

    # 步骤3: 默认级别 3
    default = TaskLevel.SCRIPT
    logger.info("[LevelRouter] 默认 → 级别 %d (%s)", default, _LEVEL_CONFIGS[default].label)
    return _LEVEL_CONFIGS[default]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_level_router_short_input_skips_llm -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/level_router.py test/test_lg_fixes.py
git commit -m "opt: LevelRouter 短输入（≤30字）跳过 LLM 分类，直接返回级别 3"
```

---

### Task 5: 简单任务跳过反思节点

**问题:** 对"创建 hello.py"这种简单任务，反思节点仍然被触发，浪费 token。当计划为"无需规划"或步骤 ≤ 2 时，反思是多余的。

**Files:**
- 修改: `agent_by_langgraph/lg_graph.py:190-269` (`_route_after_agent` 函数)

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py 追加
def test_route_after_agent_skips_reflection_for_simple_plan():
    """计划为'无需规划'时应跳过反思"""
    from agent_by_langgraph.lg_graph import _route_after_agent
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "messages": [HumanMessage(content="你好"), AIMessage(content="你好！")],
        "_phase": "verify",
        "_reflection_count": 0,
        "plan": "无需规划",
    }
    result = _route_after_agent(state)
    assert result != "reflect", f"简单计划不应触发反思，实际路由到 {result}"

def test_route_after_agent_skips_reflection_for_short_plan():
    """步骤 ≤ 2 的计划应跳过反思"""
    from agent_by_langgraph.lg_graph import _route_after_agent
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "messages": [HumanMessage(content="创建hello.py"), AIMessage(content="已创建")],
        "_phase": "verify",
        "_reflection_count": 0,
        "plan": "1. 创建 hello.py\n2. 运行验证",
    }
    result = _route_after_agent(state)
    assert result != "reflect", f"步骤 ≤ 2 的计划不应触发反思，实际路由到 {result}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_route_after_agent_skips_reflection_for_simple_plan test/test_lg_fixes.py::test_route_after_agent_skips_reflection_for_short_plan -v`
Expected: FAIL — 当前会路由到 "reflect"

- [ ] **Step 3: 修改 `_route_after_agent` 中的反思路由条件**

在 `agent_by_langgraph/lg_graph.py` 的 `_route_after_agent` 函数中，修改反思路由的触发条件：

```python
# 将反思路由部分（约第 215-222 行）:
        # verify 阶段 + 有计划 + 反思次数未超限 → 触发反思
        # modify 阶段 + 有计划 + 反思次数为 0 → 触发一次轻量反思
        if phase == "verify" and plan and plan != "无需规划" and reflection_count < 2:
            return "reflect"
        if phase == "modify" and plan and plan != "无需规划" and reflection_count < 1:
            return "reflect"

# 改为:
        # 反思触发条件：
        # 1. 有实质计划（非"无需规划"）
        # 2. 计划步骤 > 2（简单任务不需要反思）
        # 3. 反思次数未超限
        _is_simple_plan = (
            not plan
            or plan == "无需规划"
            or len(re.findall(r'^\d+\.\s', plan, re.MULTILINE)) <= 2
        )
        if not _is_simple_plan:
            if phase == "verify" and reflection_count < 2:
                return "reflect"
            if phase == "modify" and reflection_count < 1:
                return "reflect"
```

注意：需要在函数顶部或条件分支前导入 `re`（已导入）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_route_after_agent_skips_reflection_for_simple_plan test/test_lg_fixes.py::test_route_after_agent_skips_reflection_for_short_plan -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_graph.py test/test_lg_fixes.py
git commit -m "opt: 简单任务（无需规划或步骤≤2）跳过反思节点，节省 token"
```

---

### Task 6: 子代理结论提取增强 — 支持更多中文变体

**问题:** `_extract_conclusion` 中的正则只匹配"结论|总结|结果"，但中文输出可能有"## 总结："（带冒号）或"##执行结果"等变体。

**Files:**
- 修改: `agent_by_langgraph/lg_graph.py:605-630` (`_extract_conclusion` 函数)

- [ ] **Step 1: 编写失败测试**

```python
# test/test_lg_fixes.py 追加
def test_extract_conclusion_with_colon():
    """应匹配带冒号的结论标题"""
    from agent_by_langgraph.lg_graph import _extract_conclusion

    text = "一些内容\n## 总结：\n这是总结内容\n## 其他"
    result = _extract_conclusion(text)
    assert "总结内容" in result

def test_extract_conclusion_with_execution_result():
    """应匹配'执行结果'变体"""
    from agent_by_langgraph.lg_graph import _extract_conclusion

    text = "一些内容\n## 执行结果\n执行成功\n## 其他"
    result = _extract_conclusion(text)
    assert "执行成功" in result

def test_extract_conclusion_with_final_answer():
    """应匹配'最终答案'变体"""
    from agent_by_langgraph.lg_graph import _extract_conclusion

    text = "一些内容\n## 最终答案\n42\n## 其他"
    result = _extract_conclusion(text)
    assert "42" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_lg_fixes.py::test_extract_conclusion_with_colon test/test_lg_fixes.py::test_extract_conclusion_with_execution_result test/test_lg_fixes.py::test_extract_conclusion_with_final_answer -v`
Expected: 部分失败

- [ ] **Step 3: 修改 `_extract_conclusion` 正则**

在 `agent_by_langgraph/lg_graph.py` 的 `_extract_conclusion` 函数中，扩展正则匹配：

```python
def _extract_conclusion(text: str) -> str:
    """从子代理输出中提取结论。

    优先级：
    1. ## 结论/总结/结果/执行结果/最终答案 标题下的内容
    2. 最后 3 行非空文本
    """
    # 策略1: 提取结构化结论（支持冒号、空格等变体）
    conclusion_match = re.search(
        r'(?:^|\n)##\s*(?:结论|总结|结果|执行结果|最终答案|Conclusion|Summary|Result|Final Answer)'
        r'[:：]?\s*\n(.*?)(?:\n##|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if conclusion_match:
        return conclusion_match.group(1).strip()

    # 策略2: 取最后 3 行非空文本
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[-3:])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_lg_fixes.py::test_extract_conclusion_with_colon test/test_lg_fixes.py::test_extract_conclusion_with_execution_result test/test_lg_fixes.py::test_extract_conclusion_with_final_answer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_graph.py test/test_lg_fixes.py
git commit -m "opt: 子代理结论提取支持更多中文变体（执行结果、最终答案、带冒号标题）"
```

---

## P2: 代码质量

### Task 7: 偏差检测拆分为独立函数

**问题:** `_advance_phase` 函数混合了阶段推进和偏差检测两个职责，函数体超过 120 行，维护困难。

**Files:**
- 修改: `agent_by_langgraph/lg_graph.py:959-1200` (`_advance_phase` 函数)

- [ ] **Step 1: 确认现有测试不受影响**

Run: `python -m pytest test/ -v --tb=short 2>&1 | head -50`
Expected: 现有测试通过（或无测试因纯重构而失败）

- [ ] **Step 2: 提取 `_detect_deviation` 函数**

在 `agent_by_langgraph/lg_graph.py` 中，将偏差检测逻辑从 `_advance_phase` 拆分为独立函数：

```python
def _detect_deviation(
    state: AgentState,
    new_stall: int,
) -> tuple[int, str]:
    """偏差检测：分析执行状态，返回 (deviation_count, deviation_reason)。

    检测信号：
    1. 不可恢复的工具错误 → deviation_count +1
    2. 计划步骤停滞（stall_count >= 2）→ deviation_count +1
    3. 连续 3 轮相同工具调用 → deviation_count +1
    4. 连续 3 轮完全相同的工具调用（死循环）→ deviation_count +2
    5. 无偏差信号 → deviation_count 衰减 -1

    Args:
        state: 当前 Agent 状态
        new_stall: 本轮更新后的停滞计数

    Returns:
        (new_deviation_count, new_deviation_reason)
    """
    deviation_count = state.get("_deviation_count", 0) or 0
    deviation_reason = state.get("_deviation_reason", "") or ""
    messages = state.get("messages", [])

    # 信号1：最近一条 ToolMessage 为错误
    _RECOVERABLE_ERROR_PATTERNS = (
        "参数为空", "path 参数", "必须提供", "未找到", "not found",
        "no such file", "does not exist", "missing", "required",
    )
    has_tool_error = False
    is_recoverable_error = False
    tool_error_detail = ""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.status == "error":
                has_tool_error = True
                tool_error_detail = str(msg.content)[:200] if msg.content else ""
                error_lower = tool_error_detail.lower()
                is_recoverable_error = any(
                    pattern.lower() in error_lower for pattern in _RECOVERABLE_ERROR_PATTERNS
                )
            break

    if has_tool_error and not is_recoverable_error:
        deviation_count += 1
        deviation_reason = "工具执行出错"
        logger.info("[偏差检测] 工具错误(不可恢复), deviation_count=%d, detail=%s",
                    deviation_count, tool_error_detail[:80])
    elif has_tool_error and is_recoverable_error:
        logger.info("[偏差检测] 工具错误(可恢复，不计入偏差), detail=%s", tool_error_detail[:80])
    elif new_stall >= 2:
        deviation_count += 1
        deviation_reason = f"计划步骤停滞 (stall_count={new_stall})"
        logger.info("[偏差检测] 计划停滞, deviation_count=%d", deviation_count)
    else:
        # 信号3：连续调用同一工具且参数高度相似
        recent_ai_msgs = [
            m for m in messages[-8:]
            if isinstance(m, AIMessage) and m.tool_calls
        ]
        if len(recent_ai_msgs) >= 3:
            recent_tool_sets = [
                frozenset(tc["name"] for tc in m.tool_calls)
                for m in recent_ai_msgs[-3:]
            ]
            if len(recent_tool_sets) == 3 and recent_tool_sets[0] == recent_tool_sets[1] == recent_tool_sets[2]:
                def _tool_call_sig(msg: AIMessage) -> str:
                    parts = []
                    for tc in msg.tool_calls:
                        args_str = str(tc.get("args", {}))[:100]
                        parts.append(f"{tc['name']}:{args_str}")
                    return "|".join(sorted(parts))

                sigs = [_tool_call_sig(m) for m in recent_ai_msgs[-3:]]
                if sigs[0] == sigs[1] == sigs[2]:
                    deviation_count += 2
                    deviation_reason = f"连续 3 轮完全相同的工具调用（死循环）: {recent_tool_sets[0]}"
                    logger.warning(
                        "[偏差检测] 死循环, deviation_count=%d, sig=%s",
                        deviation_count, sigs[0][:80],
                    )
                else:
                    deviation_count += 1
                    deviation_reason = f"连续 3 轮调用相同工具: {recent_tool_sets[0]}"
                    logger.info(
                        "[偏差检测] 连续同工具, deviation_count=%d, tools=%s",
                        deviation_count, recent_tool_sets[0],
                    )
            else:
                deviation_count = max(0, deviation_count - 1)
                if deviation_count == 0:
                    deviation_reason = ""
        else:
            deviation_count = max(0, deviation_count - 1)
            if deviation_count == 0:
                deviation_reason = ""

    return deviation_count, deviation_reason
```

- [ ] **Step 3: 简化 `_advance_phase`，调用 `_detect_deviation`**

将 `_advance_phase` 中的偏差检测部分替换为调用新函数：

```python
    # 在 _advance_phase 的阶段推进逻辑之后，替换偏差检测部分为:
    new_deviation_count, new_deviation_reason = _detect_deviation(state, new_stall)
```

- [ ] **Step 4: 运行现有测试确认无回归**

Run: `python -m pytest test/ -v --tb=short 2>&1 | head -50`
Expected: 无新增失败

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_graph.py
git commit -m "refactor: 偏差检测从 _advance_phase 拆分为独立 _detect_deviation 函数"
```

---

### Task 8: Windows 编码修复统一到入口文件

**问题:** UTF-8 编码修复代码出现在 `agent_lg.py` 入口和 `lg_agent.py` 类初始化中，重复且分散。

**Files:**
- 修改: `agent_lg.py` — 保留并增强编码修复
- 修改: `agent_by_langgraph/lg_agent.py:15-22` — 移除重复的编码修复

- [ ] **Step 1: 确认 agent_lg.py 入口已有编码修复**

查看 `agent_lg.py` 顶部是否已有 Windows UTF-8 修复代码。如果有，确认覆盖了 `lg_agent.py` 中的所有修复项。

- [ ] **Step 2: 从 lg_agent.py 移除重复的编码修复**

在 `agent_by_langgraph/lg_agent.py` 中，移除顶部的 Windows 编码修复代码：

```python
# 删除以下代码（约第 15-22 行）:
# D15: Windows 控制台 UTF-8 编码修复，防止中文乱码
if sys.platform == "win32":
    try:
        # 设置控制台代码页为 UTF-8，确保 PowerShell/CMD 正确显示中文
        os.system('chcp 65001 >nul 2>&1')
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
```

- [ ] **Step 3: 确保 agent_lg.py 入口的编码修复完整**

在 `agent_lg.py` 顶部确认有完整的编码修复（含 stdin）：

```python
if sys.platform == "win32":
    try:
        os.system('chcp 65001 >nul 2>&1')
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
```

- [ ] **Step 4: 运行 Agent 验证编码正常**

Run: `echo "hello" | python agent_lg.py 2>&1 | head -20`
Expected: 无编码错误

- [ ] **Step 5: Commit**

```bash
git add agent_lg.py agent_by_langgraph/lg_agent.py
git commit -m "refactor: Windows 编码修复统一到 agent_lg.py 入口，移除 lg_agent.py 中的重复代码"
```

---

## P3: 边缘场景

### Task 9: 子代理 checkpointer 使用内存模式，避免 SQLite 锁冲突

**问题:** 子代理 checkpointer 按 user_id 共享 SQLite 数据库，高并发时可能出现锁冲突。子代理无需持久化状态（任务完成后即丢弃），使用内存 checkpointer 更合适。

**Files:**
- 修改: `agent_by_langgraph/lg_graph.py` 中 `_sub_checkpointer_cache` 相关逻辑

- [ ] **Step 1: 查看子代理 checkpointer 的使用方式**

搜索 `_sub_checkpointer_cache` 和 `_get_sub_checkpointer` 函数的位置和调用方式。

- [ ] **Step 2: 将子代理 checkpointer 改为 MemorySaver**

在 `agent_by_langgraph/lg_graph.py` 中，将子代理的 SQLite checkpointer 替换为内存版：

```python
# 将 _get_sub_checkpointer 函数改为:
from langgraph.checkpoint.memory import MemorySaver

def _get_sub_checkpointer(user_id: str):
    """获取子代理的内存 checkpointer（无需持久化，避免 SQLite 锁冲突）。"""
    # 子代理任务完成后即丢弃，无需持久化到磁盘
    return MemorySaver()
```

同时移除 `_sub_checkpointer_cache` 和 `_sub_checkpointer_lock`（不再需要缓存和锁）。

- [ ] **Step 3: 更新 LGAgent.close() 中的清理逻辑**

在 `agent_by_langgraph/lg_agent.py` 的 `close()` 方法中，移除子代理 checkpointer 缓存清理：

```python
# 删除以下代码:
    # 清理子代理 checkpointer 缓存
    from agent_by_langgraph.lg_graph import _sub_checkpointer_cache, _sub_checkpointer_lock
    with _sub_checkpointer_lock:
        if self.user_id and self.user_id in _sub_checkpointer_cache:
            sub_ctx, _ = _sub_checkpointer_cache.pop(self.user_id)
            self._close_checkpointer_ctx_obj(sub_ctx, f"子代理 Checkpointer (user_id={self.user_id})")
```

- [ ] **Step 4: 运行 Agent 验证子代理派遣正常**

Run: `python -c "from agent_by_langgraph.lg_graph import _get_sub_checkpointer; cp = _get_sub_checkpointer('test'); print(type(cp))"`
Expected: 输出 `MemorySaver` 类型

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_graph.py agent_by_langgraph/lg_agent.py
git commit -m "opt: 子代理 checkpointer 改用 MemorySaver，避免 SQLite 锁冲突"
```

---

### Task 10: Checkpointer 初始化优化 — 避免首次调用时重新编译图

**问题:** `_ensure_checkpointer()` 在首次 `ainvoke` 时初始化 SQLite checkpointer 并重新编译整个 StateGraph，导致首次调用延迟增加。

**Files:**
- 修改: `agent_by_langgraph/lg_agent.py:400-470` (`_init_checkpointer` 和 `_ensure_checkpointer`)

- [ ] **Step 1: 分析当前延迟初始化的必要性**

当前延迟初始化的原因是 `AsyncSqliteSaver.from_conn_string()` 返回 async context manager，必须在正确的事件循环中 `__aenter__()`。在 `__init__` 中没有事件循环，所以延迟到首次 `ainvoke`。

优化方案：在 `__init__` 中保存 db_path，在 `run()` 方法启动前（同步 REPL 的 `asyncio.run()` 内部）完成初始化，避免在 `ainvoke` 调用链中重新编译图。

- [ ] **Step 2: 修改 `_init_checkpointer`，保存 db_path 但不创建 checkpointer**

当前代码已经是这样做的（返回 None），无需修改。关键是优化 `_ensure_checkpointer` 的调用时机。

- [ ] **Step 3: 在 `run()` 方法中提前调用 `_ensure_checkpointer`**

在 `agent_by_langgraph/lg_agent.py` 的 `run()` 方法中，在进入 REPL 循环前，提前初始化 checkpointer：

```python
def run(self) -> None:
    """REPL 交互循环。"""
    # 提前初始化 checkpointer（在 asyncio.run 内部完成，避免首次 ainvoke 时重新编译图）
    if self.will_have_checkpointer and not self.checkpointer_ready:
        import asyncio
        try:
            asyncio.run(self._ensure_checkpointer())
            # 重新编译后更新系统提示词引用
            logger.info("[Checkpointer] 预初始化完成")
        except Exception as exc:
            logger.warning("[Checkpointer] 预初始化失败（将在首次调用时重试）: %s", exc)

    while True:
        # ... 原有 REPL 循环
```

- [ ] **Step 4: 简化 `_invoke_with_checkpointer` 中的 checkpointer 检查**

由于 checkpointer 已在 REPL 启动前初始化，`_invoke_with_checkpointer` 中的 `await self._ensure_checkpointer()` 调用可以直接跳过（如果已初始化）：

```python
async def _invoke_with_checkpointer():
    # checkpointer 已在 run() 启动前初始化，此处只需确认
    if not self.checkpointer_ready:
        await self._ensure_checkpointer()
    # ... 其余逻辑不变
```

- [ ] **Step 5: 运行 Agent 验证首次调用无额外延迟**

Run: `echo "hello" | python agent_lg.py 2>&1 | head -30`
Expected: 启动时显示 `[Checkpointer] 预初始化完成`，首次交互无重新编译延迟

- [ ] **Step 6: Commit**

```bash
git add agent_by_langgraph/lg_agent.py
git commit -m "opt: Checkpointer 在 REPL 启动前预初始化，避免首次 ainvoke 时重新编译图"
```

---

## 自检清单

### 1. 规格覆盖

| 问题 | 对应 Task |
|------|-----------|
| fallback 同步调用 | Task 1 |
| 重复写操作 | Task 2 |
| 阶段推断不一致 | Task 3 |
| LevelRouter LLM 开销 | Task 4 |
| 反思浪费 token | Task 5 |
| 结论提取中文变体 | Task 6 |
| 偏差检测复杂度 | Task 7 |
| 编码修复分散 | Task 8 |
| 子代理 SQLite 锁冲突 | Task 9 |
| Checkpointer 延迟初始化 | Task 10 |

### 2. 占位符扫描

无 TBD、TODO、"implement later" 等占位符。所有步骤包含完整代码。

### 3. 类型一致性

- `_detect_deviation` 返回 `tuple[int, str]`，与 `_advance_phase` 中的 `new_deviation_count`（int）和 `new_deviation_reason`（str）类型一致
- `_extract_phase` 返回 `str`，与 `AgentState` 的 `_phase` 字段类型一致
- `_get_sub_checkpointer` 返回 `MemorySaver`，与 `create_agent_graph` 的 `checkpointer` 参数类型兼容
