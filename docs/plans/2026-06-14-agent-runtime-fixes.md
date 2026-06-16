# Agent 运行时缺陷修复 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 Agent 实际运行中发现的 4 个缺陷，确保 REPL 交互稳定运行

**Architecture:** 4 个独立任务，按优先级从高到低执行。每个任务独立可验证，互不依赖。

**Tech Stack:** Python 3.11+, LangGraph, aiosqlite, Windows Terminal

---

## 文件结构

| 操作 | 文件路径 | 职责 |
|------|----------|------|
| 修改 | `agent_by_langgraph/lg_agent.py` | 持久事件循环、close 清理、arun_stream 兼容 |
| 修改 | `agent_lg.py` | Windows 编码修复增强 |

---

## 背景：已修复的问题

以下问题在本次运行诊断中已直接修复（代码已变更），本计划不再重复：

1. **AsyncSqliteSaver 事件循环崩溃** — `RuntimeError: threads can only be started once`
   - 根因：`run()` 中 `asyncio.run()` 每次创建新事件循环，但 aiosqlite 连接线程绑定旧循环
   - 已修复：改为持久事件循环（daemon 线程 + `run_coroutine_threadsafe`）

2. **`__del__` 关闭事件循环报错** — `RuntimeError: Cannot close a running event loop`
   - 已修复：先 `call_soon_threadsafe(loop.stop)` + `join` 线程，再 `close`，并捕获 RuntimeError

---

## Task 1: Windows 控制台中文乱码修复

**问题:** Agent 输出的中文在 Windows 终端中显示为乱码。当前 `agent_lg.py` 使用 `chcp 65001` + `sys.stdout.reconfigure("utf-8")`，但 Windows 终端仍用 GBK 解码 UTF-8 字节，导致乱码。Agent 接收到乱码用户输入后无法正确理解意图。

**根因分析:**
- `chcp 65001` 改变控制台代码页，但 Python 进程的 stdout 编码已由 `reconfigure` 设置为 UTF-8
- 问题在于 Windows 终端模拟器（如 Windows Terminal、PowerShell 5）在 `chcp 65001` 后仍可能用 GBK 渲染
- 更深层原因：Python 3.11+ 在 Windows 上默认使用 `locale.getpreferredencoding()` 返回 `gbk`，`sys.stdout.reconfigure` 只改了 Python 层编码，但终端渲染层仍用系统代码页

**Files:**
- 修改: `agent_lg.py:1-20`

- [ ] **Step 1: 确认当前编码修复代码**

查看 `agent_lg.py` 当前内容：

```python
"""LangGraph Agent 入口 —— python agent_lg.py 启动"""
from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    try:
        os.system('chcp 65001 >nul 2>&1')
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agent_by_langgraph.lg_agent import LGAgent

if __name__ == "__main__":
    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    agent.run()
```

- [ ] **Step 2: 替换编码修复方案**

将 `agent_lg.py` 的编码修复替换为更可靠的方案。核心思路：设置 `PYTHONIOENCODING` 环境变量 + 使用 `sys.stdout.buffer` 直接写入 UTF-8 字节 + 设置控制台输出代码页。

```python
"""LangGraph Agent 入口 —— python agent_lg.py 启动"""
from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    try:
        # 1. 设置控制台代码页为 UTF-8（影响 WriteConsoleW 的行为）
        os.system('chcp 65001 >nul 2>&1')
        # 2. 设置进程级环境变量，确保子进程也使用 UTF-8
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        # 3. 重新配置标准流编码
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        # 4. 设置 Python 默认文件编码
        os.environ.setdefault("PYTHONUTF8", "1")
    except (AttributeError, OSError):
        pass

from agent_by_langgraph.lg_agent import LGAgent

if __name__ == "__main__":
    agent = LGAgent(model="deepseek-v4-flash", max_iterations=50)
    agent.run()
```

- [ ] **Step 3: 在 lg_agent.py 的 run() 方法中增强 print 输出编码**

在 `agent_by_langgraph/lg_agent.py` 的 `run()` 方法中，REPL 提示符 `input("You🫅 : ")` 中的 emoji 在 GBK 终端下也会乱码。增加防御性处理：

在 `run()` 方法开头（REPL 循环前）添加：

```python
        # Windows 终端 emoji 兼容：如果 stdout 编码不支持 emoji，替换提示符
        _repl_prompt = "You🫅 : "
        try:
            _repl_prompt.encode(sys.stdout.encoding or "utf-8")
        except UnicodeEncodeError:
            _repl_prompt = "You> "
```

然后在 `input()` 调用中使用 `_repl_prompt`：

```python
            try:
                user_input = input(_repl_prompt)
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
```

- [ ] **Step 4: 验证修复效果**

Run: `echo "hello" | python agent_lg.py 2>&1 | head -20`
Expected: 无乱码，无编码错误

如果仍然乱码，说明 Windows 终端模拟器本身不支持 UTF-8 渲染，需要用户切换到 Windows Terminal（推荐）或使用 `PYTHONIOENCODING=utf-8 python agent_lg.py` 启动。

- [ ] **Step 5: Commit**

```bash
git add agent_lg.py agent_by_langgraph/lg_agent.py
git commit -m "fix: Windows 控制台中文乱码修复（PYTHONIOENCODING + emoji 兼容）"
```

---

## Task 2: 持久事件循环的 arun_stream 兼容性

**问题:** `run()` 方法已改用持久事件循环（daemon 线程 + `run_coroutine_threadsafe`），但 `arun_stream()` 方法仍在调用方的事件循环中直接 `await`，如果调用方的事件循环与 checkpointer 初始化时的事件循环不同，会复现 `RuntimeError: threads can only be started once`。

**根因分析:**
- `arun_stream()` 是异步方法，调用方已有事件循环（如 FastAPI 的 uvicorn）
- `await self._ensure_checkpointer()` 在调用方的事件循环中初始化 SQLite 连接
- 如果调用方的事件循环重启（如测试场景），checkpointer 的 aiosqlite 连接线程会失效
- 需要确保 `arun_stream` 也能正确处理事件循环生命周期

**Files:**
- 修改: `agent_by_langgraph/lg_agent.py:717-818` (`arun_stream` 方法)

- [ ] **Step 1: 分析 arun_stream 的事件循环使用**

当前 `arun_stream()` 直接在调用方的事件循环中 `await`：
- `await self._ensure_checkpointer()` — 初始化 checkpointer
- `async for event in self.graph.astream_events(...)` — 流式执行

如果调用方的事件循环与 checkpointer 创建时的事件循环相同（通常如此），则没问题。
但如果 `run()` 先被调用（在持久事件循环中初始化了 checkpointer），然后 `arun_stream()` 在另一个事件循环中调用，checkpointer 的 aiosqlite 连接线程绑定到持久事件循环，无法在新循环中使用。

- [ ] **Step 2: 在 arun_stream 中检测事件循环不匹配**

在 `arun_stream()` 方法中，在 `await self._ensure_checkpointer()` 之前，添加事件循环一致性检查：

```python
        # 检查 checkpointer 是否在另一个事件循环中初始化
        # 如果是，需要在新事件循环中重新初始化（关闭旧的，创建新的）
        if self.checkpointer_ready:
            import asyncio
            current_loop = asyncio.get_running_loop()
            repl_loop = getattr(self, '_repl_loop', None)
            if repl_loop is not None and repl_loop is not current_loop:
                # checkpointer 在持久事件循环中初始化，但当前在不同事件循环
                # 需要关闭旧 checkpointer 并重新初始化
                logger.warning(
                    "[arun_stream] 事件循环不匹配，重新初始化 checkpointer"
                )
                self._close_checkpointer_ctx("_checkpointer_ctx", "主 Checkpointer")
                self._checkpointer_initialized = False
                self._checkpointer_failed = False
                if hasattr(self, '_checkpointer_db_path') and self._checkpointer_db_path:
                    # 重新初始化将在下方 await self._ensure_checkpointer() 中完成
                    pass
```

- [ ] **Step 3: 在 arun_stream 中确保 _ensure_checkpointer 在当前事件循环中执行**

当前代码 `await self._ensure_checkpointer()` 已经在当前事件循环中执行，无需修改。Step 2 的重置逻辑确保了如果事件循环不匹配，checkpointer 会被重新初始化。

- [ ] **Step 4: 验证 arun_stream 在不同场景下正常工作**

场景 1：直接调用 arun_stream（无 run() 先行）
```python
import asyncio
async def test():
    agent = LGAgent(model="deepseek-v4-flash")
    async for token in agent.arun_stream("hello"):
        print(token, end="")
asyncio.run(test())
```

场景 2：先 run() 再 arun_stream（事件循环不匹配）
```python
# 此场景需要特殊处理，因为 run() 的持久事件循环与 arun_stream 的事件循环不同
# Step 2 的检查逻辑会检测到不匹配并重新初始化
```

- [ ] **Step 5: Commit**

```bash
git add agent_by_langgraph/lg_agent.py
git commit -m "fix: arun_stream 事件循环不匹配时重新初始化 checkpointer"
```

---

## Task 3: 持久事件循环线程安全 — 防止重复创建

**问题:** `run()` 方法中的持久事件循环创建逻辑没有线程安全保护。如果 `run()` 被意外并发调用（如信号处理、多线程场景），可能创建多个事件循环线程。

**Files:**
- 修改: `agent_by_langgraph/lg_agent.py:510-525` (持久事件循环创建逻辑)

- [ ] **Step 1: 查看当前持久事件循环创建代码**

当前代码（约第 510-525 行）：

```python
            if not hasattr(self, '_repl_loop') or self._repl_loop is None or self._repl_loop.is_closed():
                import asyncio
                self._repl_loop = asyncio.new_event_loop()
                import threading
                def _loop_runner(loop):
                    asyncio.set_event_loop(loop)
                    loop.run_forever()
                self._repl_thread = threading.Thread(target=_loop_runner, args=(self._repl_loop,), daemon=True)
                self._repl_thread.start()
```

问题：
1. `hasattr` + 赋值不是原子操作，多线程可能同时通过检查
2. `_repl_thread.start()` 后没有等待事件循环就绪，`run_coroutine_threadsafe` 可能在循环启动前提交

- [ ] **Step 2: 添加线程安全保护和就绪信号**

将持久事件循环创建逻辑提取为方法，添加锁和就绪事件：

```python
    def _ensure_repl_loop(self):
        """确保 REPL 持久事件循环已创建并就绪（线程安全）。"""
        import asyncio
        import threading

        if hasattr(self, '_repl_loop') and self._repl_loop is not None and not self._repl_loop.is_closed():
            return  # 已创建且运行中

        with self._invoke_lock:  # 复用已有的 invoke_lock
            # 双重检查
            if hasattr(self, '_repl_loop') and self._repl_loop is not None and not self._repl_loop.is_closed():
                return

            self._repl_loop_ready = threading.Event()
            self._repl_loop = asyncio.new_event_loop()

            def _loop_runner(loop, ready_event):
                asyncio.set_event_loop(loop)
                ready_event.set()  # 通知主线程事件循环已就绪
                loop.run_forever()

            self._repl_thread = threading.Thread(
                target=_loop_runner,
                args=(self._repl_loop, self._repl_loop_ready),
                daemon=True,
                name="lg-agent-repl-loop",
            )
            self._repl_thread.start()
            # 等待事件循环就绪（最多 5 秒）
            if not self._repl_loop_ready.wait(timeout=5):
                raise RuntimeError("REPL 持久事件循环启动超时")
```

- [ ] **Step 3: 替换 run() 中的内联创建逻辑**

将 `run()` 方法中的内联创建代码替换为调用 `_ensure_repl_loop()`：

```python
            # 使用持久事件循环执行 ainvoke
            self._ensure_repl_loop()
```

- [ ] **Step 4: 更新 close() 方法**

`close()` 方法中的事件循环关闭逻辑保持不变（已有 try/except RuntimeError 保护）。

- [ ] **Step 5: 验证**

Run: `echo "hello" | python agent_lg.py 2>&1 | head -20`
Expected: 正常运行，无事件循环相关错误

- [ ] **Step 6: Commit**

```bash
git add agent_by_langgraph/lg_agent.py
git commit -m "fix: 持久事件循环添加线程安全保护和就绪信号"
```

---

## Task 4: _ensure_checkpointer 重编译图的回调丢失风险

**问题:** `_ensure_checkpointer()` 重新编译图后，虽然合并了旧回调，但 `run()` 中的 `StreamHandler` 和 `ReasoningCollector` 是每次 REPL 轮次新建的局部变量，旧回调列表中的引用是上一轮的实例。如果 checkpointer 在第二轮才初始化（如首次调用失败后重试），合并的旧回调是过期的。

**Files:**
- 修改: `agent_by_langgraph/lg_agent.py:638-700` (`_ensure_checkpointer` 方法)

- [ ] **Step 1: 分析当前回调合并逻辑**

当前 `_ensure_checkpointer()` 中的回调合并：

```python
            # 保存旧图的回调列表
            old_callbacks = list(getattr(self.graph, '_lg_llm_callbacks', []))
            # 重新编译图
            self.graph = create_agent_graph(...)
            # 合并旧回调到新图
            new_callbacks = list(getattr(self.graph, '_lg_llm_callbacks', []))
            for cb in old_callbacks:
                if not isinstance(cb, TokenTrackerCallback) and cb not in new_callbacks:
                    new_callbacks.append(cb)
            self.graph._lg_llm_callbacks = new_callbacks
```

问题：`old_callbacks` 中的 `StreamHandler` 和 `ReasoningCollector` 是上一轮 REPL 的局部变量，本轮不会使用。本轮在 `_invoke_with_checkpointer()` 中重新设置了 `config["callbacks"]`，所以实际回调是正确的。但 `_lg_llm_callbacks` 中残留的旧回调可能导致：
1. 内存泄漏（旧回调持有对局部变量的引用）
2. TokenTrackerCallback 被重复添加（新图已自带，旧图的也被合并）

- [ ] **Step 2: 简化回调合并逻辑**

修改 `_ensure_checkpointer()` 中的回调合并，只保留 `TokenTrackerCallback`，丢弃过期的 per-invoke 回调：

```python
            # 保存旧图的 TokenTrackerCallback（跨轮次持久）
            old_tracker_callbacks = [
                cb for cb in getattr(self.graph, '_lg_llm_callbacks', [])
                if isinstance(cb, TokenTrackerCallback)
            ]

            # 重新编译图以注入 checkpointer
            self.graph = create_agent_graph(
                self.llm, self.tools, self._system_prompt,
                llm_callbacks=[TokenTrackerCallback(self.token_tracker, self.model)],
                checkpointer=checkpointer,
            )

            # 新图已自带 TokenTrackerCallback，无需再合并旧的
            # per-invoke 回调（StreamHandler, ReasoningCollector）在每轮 ainvoke 时
            # 由 _invoke_with_checkpointer() 重新设置到 config["callbacks"]，
            # 不应存储到 _lg_llm_callbacks 中（避免过期引用和内存泄漏）
```

- [ ] **Step 3: 验证**

Run: `echo "hello" | python agent_lg.py 2>&1 | head -20`
Expected: 正常运行，token 统计正确

- [ ] **Step 4: Commit**

```bash
git add agent_by_langgraph/lg_agent.py
git commit -m "fix: _ensure_checkpointer 回调合并只保留 TokenTrackerCallback，避免过期引用"
```

---

## 自检清单

### 1. 规格覆盖

| 问题 | 对应 Task |
|------|-----------|
| Windows 中文乱码 | Task 1 |
| arun_stream 事件循环不匹配 | Task 2 |
| 持久事件循环线程安全 | Task 3 |
| checkpointer 回调丢失/过期 | Task 4 |

### 2. Placeholder 扫描

- 无 "TBD"、"TODO"、"implement later" 等占位符
- 所有步骤包含具体代码
- 所有命令包含预期输出

### 3. 类型一致性

- `_ensure_repl_loop()` 方法在 Task 3 中定义，在 Task 3 的 Step 3 中调用
- `_repl_loop_ready` 在 Task 3 中定义，与 `_repl_loop` 生命周期一致
- `_close_checkpointer_ctx` 在 Task 2 中调用，已存在于当前代码中
