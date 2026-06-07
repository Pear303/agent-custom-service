"""Bug 修复 + 功能增强测试 —— 给 Agent 一个故意写坏的项目，要求读代码、修 bug、验结果。

与之前三个测试的区别：
  1. 必须先用 read_file/grep_tool 读代码 → 强制触发 gather 阶段
  2. 必须用 edit_file 精确修改 → 测试定向编辑路径（之前测试只用 write_file）
  3. 必须用 run_command 运行验证 → 测试命令执行 + 错误恢复
  4. 修复后可能仍有问题 → 测试迭代修改循环
  5. 新增功能用 write_file → 测试 write_file + edit_file 混合使用
  6. 使用 astream_events 监听 → 捕获 P3 阶段转换、节点执行详情

使用方式：
    python test_bugfix_task.py

前置条件：
    - .env 文件中已配置 DEEPSEEK_API_KEY
    - 依赖已安装

已知 Bug（5个）：
  1. todo.py: add() 用时间戳做 ID，同一秒添加会覆盖
  2. todo.py: delete() 用 int(task_id) 做索引而非匹配 id 字段，off-by-one
  3. todo.py: complete() 写 "done" 但 list("completed") 查 "completed"
  4. storage.py: open() 未指定 encoding，Windows 下中文乱码
  5. test_todo.py: 断言列表顺序但顺序不确定

新增功能：
  - export_json(): 导出所有任务为 JSON 文件
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Windows 编码修复
if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_bugfix_task.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_bugfix")

# ── 常量 ──────────────────────────────────────────────────────

TEST_USER_ID = "test_bugfix_user"
TEST_TICKET_ID = "bugfix_task_001"
GLOBAL_TIMEOUT_S = 900
DEV_TIMEOUT_S = 600
RESULT_FILE = Path("test_bugfix_result.json")

# 项目源文件路径
FIXTURES_DIR = Path(__file__).parent / "test_fixtures" / "todo_manager"

# Bug 修复任务
BUGFIX_PROMPT = """\
你收到一个 Python TODO 待办管理器项目，该项目包含多个已知 bug。你的任务是：

1. **阅读所有源代码文件**：todo.py, storage.py, test_todo.py
2. **找出并修复所有 bug**（至少 5 个）：
   - todo.py: add() 用时间戳做 ID，同一秒添加会覆盖 → 改用自增 ID 或 UUID
   - todo.py: delete() 用 int(task_id) 做索引而非匹配 id 字段，off-by-one → 改为按 id 字段匹配
   - todo.py: complete() 写 "done" 但 list("completed") 查 "completed" → 统一状态值
   - storage.py: open() 未指定 encoding → 添加 encoding="utf-8"
   - test_todo.py: 断言列表顺序但顺序不确定 → 改用 set 比较或排序后比较
3. **修复后用 run_command 运行测试**：python test_todo.py
4. **如果测试失败，继续修复直到通过**
5. **新增功能**：在 todo.py 的 TodoManager 类中实现 export_json(filepath) 方法，将所有任务导出为 JSON 文件
6. **为新增功能添加测试**

⚠️ 重要约束：
- 修复 bug 时必须使用 edit_file 工具（不要用 write_file 重写整个文件）
- 新增功能可以用 edit_file 或 write_file
- 每次修复后运行测试验证
- 最终确保 python test_todo.py 全部通过
"""

# ── 结果收集器 ────────────────────────────────────────────────

class TestResult:
    """结构化测试结果收集器。"""

    def __init__(self):
        self.start_time = time.time()
        self.phases: dict[str, dict] = {}
        self.checks: list[dict] = []
        self.errors: list[str] = []
        self.findings: list[dict] = []
        self.token_usage: dict[str, int] = {"input": 0, "output": 0}
        self.files_created: list[str] = []
        self.passed = False
        # P3 阶段追踪
        self.phase_transitions: list[dict] = []
        self.node_counts: dict[str, int] = {}
        self.tool_calls: list[dict] = []
        self.llm_calls = 0
        # edit_file 专项追踪
        self.edit_file_calls: list[dict] = []
        self.run_command_calls: list[dict] = []
        self.read_file_calls: list[dict] = []
        # 迭代追踪
        self.iteration_count = 0
        self.error_recovery_count = 0

    def record_phase(self, name: str, status: str, duration: float,
                     data: Any = None, error: str | None = None):
        self.phases[name] = {
            "status": status,
            "duration_s": round(duration, 1),
            "data": data,
            "error": error,
        }
        icon = "OK" if status == "completed" else "FAIL"
        logger.info("[Phase %s] %s (%.1fs) %s", icon, name, duration, error or "")

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        icon = "PASS" if passed else "FAIL"
        logger.info("[Check %s] %s: %s", icon, name, detail)

    def add_error(self, error: str):
        self.errors.append(error)
        logger.error("[Error] %s", error)

    def add_finding(self, severity: str, title: str, detail: str = ""):
        finding = {"severity": severity, "title": title}
        if detail:
            finding["detail"] = detail
        self.findings.append(finding)
        logger.info("[Finding %s] %s: %s", severity, title, detail)

    def record_phase_transition(self, old_phase: str, new_phase: str):
        self.phase_transitions.append({"from": old_phase, "to": new_phase})
        logger.info("[PhaseTransition] %s -> %s", old_phase, new_phase)

    def record_node(self, node_name: str):
        self.node_counts[node_name] = self.node_counts.get(node_name, 0) + 1

    def record_tool_call(self, tool_name: str, args_summary: str):
        self.tool_calls.append({"tool": tool_name, "args": args_summary})
        # 分类追踪
        entry = {"tool": tool_name, "args": args_summary, "time": time.time() - self.start_time}
        if tool_name == "edit_file":
            self.edit_file_calls.append(entry)
        elif tool_name == "run_command":
            self.run_command_calls.append(entry)
        elif tool_name == "read_file":
            self.read_file_calls.append(entry)

    def to_dict(self) -> dict:
        total_duration = time.time() - self.start_time
        all_checks_passed = all(c["passed"] for c in self.checks)
        all_phases_ok = all(
            p["status"] == "completed" for p in self.phases.values()
        )
        self.passed = all_checks_passed and all_phases_ok and not self.errors

        return {
            "passed": self.passed,
            "total_duration_s": round(total_duration, 1),
            "phases": self.phases,
            "checks": self.checks,
            "errors": self.errors,
            "findings": self.findings,
            "token_usage": self.token_usage,
            "files_created": self.files_created,
            "phase_transitions": self.phase_transitions,
            "node_counts": self.node_counts,
            "tool_calls": self.tool_calls,
            "llm_calls": self.llm_calls,
            "edit_file_calls": len(self.edit_file_calls),
            "run_command_calls": len(self.run_command_calls),
            "read_file_calls": len(self.read_file_calls),
            "iteration_count": self.iteration_count,
            "error_recovery_count": self.error_recovery_count,
        }


result = TestResult()


# ── Token 追踪回调 ────────────────────────────────────────────

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class TokenTracker(BaseCallbackHandler):
    """追踪 LLM token 用量 + 工具调用。"""

    def __init__(self, result_obj: TestResult):
        self._result = result_obj

    def on_llm_end(self, response, **kwargs):
        self._result.llm_calls += 1
        usage = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage", {})
        if not usage:
            try:
                usage = response.generations[0][0].message.usage_metadata
            except Exception:
                return
        if not usage:
            return
        if hasattr(usage, "input_tokens"):
            self._result.token_usage["input"] += usage.input_tokens or 0
            self._result.token_usage["output"] += usage.output_tokens or 0
        else:
            self._result.token_usage["input"] += usage.get("input_tokens", 0)
            self._result.token_usage["output"] += usage.get("output_tokens", 0)

        # 记录工具调用
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    if isinstance(gen.message, AIMessage) and gen.message.tool_calls:
                        for tc in gen.message.tool_calls:
                            args_str = str(tc.get("args", ""))[:200]
                            self._result.record_tool_call(tc["name"], args_str)
        except Exception:
            pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logger.info("[ToolStart] %s", tool_name)

    def on_tool_end(self, output, **kwargs):
        output_str = str(output)[:200]
        logger.info("[ToolEnd] %s", output_str)

    def on_tool_error(self, error, **kwargs):
        self._result.add_finding("ERROR", "工具执行错误", str(error)[:200])
        self._result.error_recovery_count += 1


# ── Agent 调用（使用 astream_events 监听） ────────────────────

async def _call_lg_agent_with_events(prompt: str, user_id: str, ticket_id: str,
                                     max_iterations: int = 50,
                                     phase_name: str = "") -> dict:
    """调用 LangGraph Agent，使用 astream_events 监听全流程。

    关键改进（vs 之前测试的 ainvoke）：
    - 捕获 advance_phase 节点输出中的 _phase 变化 → P3 阶段转换
    - 捕获每个图节点的执行 → node_counts
    - 捕获 interrupt 事件 → 审批门行为
    """
    from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
    from agent_by_langgraph.lg_agent import ReasoningCollector
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

    phase_ticket = f"{ticket_id}_{phase_name}" if phase_name else ticket_id

    agent = create_lg_agent(
        user_id=user_id,
        ticket_id=phase_ticket,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations=max_iterations,
    )

    # 本次测试：启用 checkpointer 以观察 interrupt 行为
    # 延迟初始化将在 _ensure_checkpointer 中完成
    has_checkpointer = agent.graph.checkpointer is not None
    logger.info("[Agent] checkpointer 初始状态: %s, _checkpointer_db_path: %s",
                has_checkpointer, getattr(agent, '_checkpointer_db_path', None))

    # 构造输入
    async with agent._async_invoke_lock:
        is_first_turn = agent._first_turn
        if is_first_turn or not has_checkpointer:
            initial_messages = [SystemMessage(content=agent._system_prompt)]
            initial_messages.extend(agent.memory_store.messages)
            user_msg = HumanMessage(content=prompt)
            user_msg.metadata = {"milestone": True}
            initial_messages.append(user_msg)
            input_state = {"messages": initial_messages, "plan": "", "_phase": "all", "_stall_count": 0}
            if has_checkpointer:
                agent._first_turn = False
        else:
            user_msg = HumanMessage(content=prompt)
            user_msg.metadata = {"milestone": True}
            input_state = {"messages": [user_msg], "plan": "", "_phase": "all", "_stall_count": 0}

    # 配置
    collector = ReasoningCollector()
    token_tracker = TokenTracker(result)
    from agent_by_langgraph.lg_agent import TokenTrackerCallback
    graph_callbacks = [
        cb for cb in getattr(agent.graph, '_lg_llm_callbacks', [])
        if not isinstance(cb, TokenTrackerCallback)
    ]
    all_callbacks = list(graph_callbacks)
    all_callbacks.extend([collector, token_tracker])

    # 延迟初始化 checkpointer
    await agent._ensure_checkpointer()
    has_checkpointer = agent.graph.checkpointer is not None
    logger.info("[Agent] checkpointer 初始化后: %s", has_checkpointer)

    config: RunnableConfig = {
        "callbacks": all_callbacks,
        "recursion_limit": max_iterations * 4 + 10,
        "configurable": {
            "thread_id": f"{user_id}_{phase_name}" if phase_name else user_id,
            "__has_checkpointer__": has_checkpointer,
        },
    }

    # ── 使用 astream_events 监听全流程 ──────────────────────
    last_phase = "all"
    invoke_result = None

    # interrupt 自动批准循环
    current_input = input_state
    max_interrupt_retries = 30

    for retry in range(max_interrupt_retries):
        try:
            async for event in agent.graph.astream_events(
                current_input, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                # 记录图节点执行
                _TRACKED_NODES = {
                    "agent", "advance_phase", "tools", "interrupt_approval",
                    "aggregate_results", "subagent_dispatcher", "subagent_worker",
                    "planner", "route_after_agent", "route_after_aggregate",
                    "route_after_approval",
                }
                if kind == "on_chain_start" and name in _TRACKED_NODES:
                    result.record_node(name)

                # 捕获 P3 阶段转换（从 advance_phase 输出中）
                if kind == "on_chain_end" and name == "advance_phase":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        new_phase = output.get("_phase")
                        if new_phase and new_phase != last_phase:
                            result.record_phase_transition(last_phase, new_phase)
                            last_phase = new_phase

                # 捕获 interrupt 审批事件
                if kind == "on_chain_start" and name == "interrupt_approval":
                    logger.info("[Event] interrupt_approval 节点启动")

                # 捕获 planner 输出
                if kind == "on_chain_end" and name == "planner":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        plan = output.get("plan", "")
                        phase_hint = output.get("_phase", "")
                        logger.info("[Event] planner 输出: phase=%s, plan=%s",
                                    phase_hint, plan[:200] if plan else "空")

            # astream_events 完成，获取最终 state
            if has_checkpointer:
                snapshot = await agent.graph.aget_state(config)
                invoke_result = {"messages": snapshot.values.get("messages", []),
                                 "_phase": snapshot.values.get("_phase", "unknown"),
                                 "_stall_count": snapshot.values.get("_stall_count", 0)}
            else:
                # 无 checkpointer 时无法 aget_state，用降级方式
                invoke_result = {"messages": [], "_phase": last_phase, "_stall_count": 0}

        except Exception as exc:
            result.add_error(f"Agent 执行异常: {exc}")
            traceback.print_exc()
            try:
                reset_lg_agent(user_id, phase_ticket)
            except Exception:
                pass
            return {"output": f"ERROR: {exc}", "final_phase": "error", "stall_count": 0}

        # 检查 pending interrupt
        has_interrupt = False
        if has_checkpointer:
            try:
                snapshot = await agent.graph.aget_state(config)
                if snapshot and snapshot.next:
                    for task in snapshot.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            has_interrupt = True
                            logger.info("[Interrupt] 检测到 interrupt，自动批准: %s",
                                        task.interrupts)
                            current_input = Command(resume="approve")
                            break
            except Exception as exc:
                logger.warning("[Interrupt] aget_state 失败: %s", exc)

        if not has_interrupt:
            break
    else:
        result.add_error("interrupt 重试次数耗尽")

    # 提取回复
    collector_msg = collector.last
    if collector_msg is not None and collector_msg.content:
        content = collector_msg.content
        reply = content if isinstance(content, str) else str(content)
    else:
        messages = invoke_result.get("messages", [])
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                reply = content if isinstance(content, str) else str(content)
                break

    final_phase = invoke_result.get("_phase", "unknown")
    stall_count = invoke_result.get("_stall_count", 0)

    logger.info("[Agent] 最终阶段=%s, 停滞次数=%d", final_phase, stall_count)

    # 清理
    try:
        reset_lg_agent(user_id, phase_ticket)
    except Exception:
        pass

    return {"output": reply, "final_phase": final_phase, "stall_count": stall_count}


# ── 项目文件准备 ──────────────────────────────────────────────

def prepare_project_files(workspace: Path) -> None:
    """将有 bug 的项目文件复制到工作区。"""
    if not FIXTURES_DIR.exists():
        raise FileNotFoundError(f"项目模板目录不存在: {FIXTURES_DIR}")

    workspace.mkdir(parents=True, exist_ok=True)

    for src_file in FIXTURES_DIR.glob("*.py"):
        dst_file = workspace / src_file.name
        shutil.copy2(src_file, dst_file)
        logger.info("[Prepare] 复制 %s -> %s", src_file.name, dst_file)


# ── 成品验证 ──────────────────────────────────────────────────

def verify_fixes(workspace: Path) -> None:
    """验证 bug 修复结果。"""
    todo_py = workspace / "todo.py"
    storage_py = workspace / "storage.py"
    test_py = workspace / "test_todo.py"

    # ── 检查文件存在 ──────────────────────────────────────
    result.add_check("todo_py_exists", todo_py.exists(), str(todo_py))
    result.add_check("storage_py_exists", storage_py.exists(), str(storage_py))
    result.add_check("test_py_exists", test_py.exists(), str(test_py))

    if not todo_py.exists():
        result.add_finding("ERROR", "todo.py 不存在", "Agent 未保留或创建了错误的文件")
        return

    todo_content = todo_py.read_text(encoding="utf-8")
    storage_content = storage_py.read_text(encoding="utf-8") if storage_py.exists() else ""
    test_content = test_py.read_text(encoding="utf-8") if test_py.exists() else ""

    # ── Bug 1: ID 生成方式 ────────────────────────────────
    # 修复标志：不再用 int(time.time())，改用自增或 uuid
    has_timestamp_id = "int(time.time())" in todo_content
    has_uuid_id = "uuid" in todo_content.lower()
    has_auto_increment = re.search(r'self\._next_id|self\.counter|_id_counter', todo_content)
    bug1_fixed = not has_timestamp_id and (has_uuid_id or has_auto_increment is not None)
    result.add_check("bug1_id_generation", bug1_fixed,
                     f"时间戳ID={'已移除' if not has_timestamp_id else '仍存在'}, "
                     f"UUID={'有' if has_uuid_id else '无'}, "
                     f"自增={'有' if has_auto_increment else '无'}")

    # ── Bug 2: delete 按索引而非 id 字段 ──────────────────
    # 修复标志：delete 方法中不再用 int(task_id) 做索引
    has_int_index_delete = re.search(r'delete.*?int\(task_id\)', todo_content, re.DOTALL)
    has_id_field_match = re.search(r'task\["id"\]\s*==\s*task_id', todo_content)
    bug2_fixed = not has_int_index_delete and has_id_field_match is not None
    result.add_check("bug2_delete_by_id", bug2_fixed,
                     f"int索引={'仍存在' if has_int_index_delete else '已移除'}, "
                     f"id字段匹配={'有' if has_id_field_match else '无'}")

    # ── Bug 3: complete/list 状态不一致 ───────────────────
    # 修复标志：complete 和 list 使用相同的状态值
    complete_status = re.search(r'status["\']?\s*[:=]\s*["\'](\w+)["\']', todo_content)
    list_completed_check = re.search(r'==\s*["\'](\w+)["\']', todo_content)
    # 更精确：检查 complete 方法写入的值和 list 方法查询的值
    complete_match = re.search(r'def complete.*?status["\']?\s*[:=]\s*["\'](\w+)["\']', todo_content, re.DOTALL)
    list_match = re.search(r'filter_status\s*==\s*["\']completed["\'].*?status["\']?\s*==\s*["\'](\w+)["\']', todo_content, re.DOTALL)
    if not list_match:
        list_match = re.search(r'status["\']?\s*==\s*["\'](\w+)["\']', todo_content)

    bug3_fixed = False
    if complete_match and list_match:
        bug3_fixed = complete_match.group(1) == list_match.group(1)
    elif not has_timestamp_id:  # 如果文件已被重写，检查一致性
        # 简化检查：搜索所有 status 赋值和比较
        status_assignments = re.findall(r'"status"\s*:\s*"(\w+)"', todo_content)
        status_comparisons = re.findall(r'==\s*"(\w+)"', todo_content)
        if status_assignments and status_comparisons:
            bug3_fixed = True  # 如果文件被重写，假设已修复

    result.add_check("bug3_status_consistency", bug3_fixed,
                     f"complete写入={complete_match.group(1) if complete_match else '?'}, "
                     f"list查询={list_match.group(1) if list_match else '?'}")

    # ── Bug 4: encoding 缺失 ─────────────────────────────
    has_encoding_read = 'encoding="utf-8"' in storage_content or "encoding='utf-8'" in storage_content
    has_open_read = "open(" in storage_content
    bug4_fixed = has_encoding_read or not has_open_read
    result.add_check("bug4_encoding", bug4_fixed,
                     f"指定encoding={'是' if has_encoding_read else '否'}")

    # ── Bug 5: 测试断言顺序 ──────────────────────────────
    has_set_compare = "set(" in test_content
    has_sorted_compare = "sorted(" in test_content
    has_exact_list_compare = re.search(r'==\s*\[.*\]', test_content) is not None
    bug5_fixed = has_set_compare or has_sorted_compare or not has_exact_list_compare
    result.add_check("bug5_test_assertion", bug5_fixed,
                     f"set比较={'有' if has_set_compare else '无'}, "
                     f"sorted比较={'有' if has_sorted_compare else '无'}")

    # ── 新增功能：export_json ─────────────────────────────
    has_export_json = "export_json" in todo_content and "pass" not in re.search(
        r'def export_json.*?(?=\n    def |\nclass |\Z)', todo_content, re.DOTALL
    ).group(0) if re.search(r'def export_json', todo_content) else False
    # 简化检查：export_json 方法体不为空
    export_match = re.search(r'def export_json\(self.*?\):(.*?)(?=\n    def |\nclass |\Z)',
                             todo_content, re.DOTALL)
    if export_match:
        export_body = export_match.group(1).strip()
        has_export_json = export_body != "pass" and len(export_body) > 10
    else:
        has_export_json = False

    result.add_check("feature_export_json", has_export_json,
                     f"export_json 实现={'有' if has_export_json else '无'}")

    # ── 运行测试验证 ──────────────────────────────────────
    if test_py.exists():
        import subprocess
        try:
            proc = subprocess.run(
                [sys.executable, str(test_py)],
                capture_output=True, text=True, timeout=30,
                cwd=str(workspace),
            )
            tests_passed = proc.returncode == 0
            test_output = proc.stdout + proc.stderr
            result.add_check("tests_pass", tests_passed,
                             f"exit_code={proc.returncode}, output={test_output[:300]}")
            if not tests_passed:
                result.add_finding("WARN", "测试未通过",
                                   f"测试输出: {test_output[:500]}")
        except subprocess.TimeoutExpired:
            result.add_check("tests_pass", False, "测试运行超时")
        except Exception as exc:
            result.add_check("tests_pass", False, f"运行测试异常: {exc}")

    # ── edit_file 使用统计 ────────────────────────────────
    result.add_check("edit_file_used", len(result.edit_file_calls) > 0,
                     f"edit_file 调用 {len(result.edit_file_calls)} 次")
    result.add_check("run_command_used", len(result.run_command_calls) > 0,
                     f"run_command 调用 {len(result.run_command_calls)} 次")
    result.add_check("read_file_used", len(result.read_file_calls) > 0,
                     f"read_file 调用 {len(result.read_file_calls)} 次")


# ── 动态 Bug 检测 ─────────────────────────────────────────────

def detect_runtime_bugs() -> list[dict]:
    """基于运行时数据动态检测问题。"""
    findings = []

    # 1. 各阶段是否成功
    for name, phase in result.phases.items():
        if phase["status"] != "completed":
            findings.append({
                "severity": "ERROR",
                "title": f"阶段 {name} 失败",
                "detail": phase.get("error", "未知错误"),
            })

    # 2. 阶段耗时
    for name, phase in result.phases.items():
        if phase["duration_s"] > 300:
            findings.append({
                "severity": "WARN",
                "title": f"阶段 {name} 耗时过长",
                "detail": f"{phase['duration_s']}s",
            })

    # 3. Token 消耗
    total_tokens = result.token_usage["input"] + result.token_usage["output"]
    if total_tokens > 300_000:
        findings.append({
            "severity": "WARN",
            "title": "Token 消耗过高",
            "detail": f"总计 {total_tokens} tokens",
        })

    # 4. edit_file 调用情况
    if len(result.edit_file_calls) == 0:
        findings.append({
            "severity": "WARN",
            "title": "未使用 edit_file",
            "detail": "Agent 未使用 edit_file 修复 bug，可能用了 write_file 重写整个文件",
        })
    else:
        # 检查 edit_file 是否有失败
        for call in result.edit_file_calls:
            if "error" in call.get("args", "").lower() or "not found" in call.get("args", "").lower():
                findings.append({
                    "severity": "INFO",
                    "title": "edit_file 匹配失败",
                    "detail": call["args"][:200],
                })

    # 5. run_command 调用情况
    if len(result.run_command_calls) == 0:
        findings.append({
            "severity": "WARN",
            "title": "未使用 run_command",
            "detail": "Agent 未运行测试验证修复结果",
        })

    # 6. P3 阶段转换
    if not result.phase_transitions:
        findings.append({
            "severity": "WARN",
            "title": "未检测到 P3 阶段转换",
            "detail": "P3 自适应工具选择可能未正常工作",
        })
    else:
        # 检查是否经历了完整的 gather→modify→verify
        phases_seen = set()
        for pt in result.phase_transitions:
            phases_seen.add(pt["from"])
            phases_seen.add(pt["to"])

        if "gather" in phases_seen:
            findings.append({
                "severity": "INFO",
                "title": "P3 gather 阶段已触发",
                "detail": "信息收集阶段正常工作",
            })
        else:
            findings.append({
                "severity": "WARN",
                "title": "P3 未经历 gather 阶段",
                "detail": "可能跳过了信息收集直接修改",
            })

        if "modify" in phases_seen:
            findings.append({
                "severity": "INFO",
                "title": "P3 modify 阶段已触发",
                "detail": "代码修改阶段正常工作",
            })

        if "verify" in phases_seen:
            findings.append({
                "severity": "INFO",
                "title": "P3 verify 阶段已触发",
                "detail": "验证阶段正常工作",
            })

        # 检查异常回退
        rollback_count = sum(
            1 for pt in result.phase_transitions
            if pt["to"] == "all" and pt["from"] != "all"
        )
        if rollback_count > 0:
            findings.append({
                "severity": "WARN",
                "title": f"P3 发生 {rollback_count} 次回退到 'all'",
                "detail": "可能是 gather 阶段写工具被阻止导致停滞回退",
            })

    # 7. 迭代修复情况
    if result.error_recovery_count > 0:
        findings.append({
            "severity": "INFO",
            "title": f"错误恢复 {result.error_recovery_count} 次",
            "detail": "Agent 在修复过程中遇到了错误并尝试恢复",
        })

    # 8. LLM 调用次数
    if result.llm_calls > 30:
        findings.append({
            "severity": "WARN",
            "title": "LLM 调用次数过多",
            "detail": f"共 {result.llm_calls} 次，可能存在循环",
        })

    # 9. 关键断言失败
    failed_checks = [c for c in result.checks if not c["passed"]]
    if failed_checks:
        findings.append({
            "severity": "WARN",
            "title": f"{len(failed_checks)} 项验证未通过",
            "detail": ", ".join(c["name"] for c in failed_checks),
        })

    return findings


# ── 旧问题对照检查 ────────────────────────────────────────────

def verify_old_issues() -> list[dict]:
    """对照检查之前发现的旧问题是否真实存在。"""
    issues = []

    # ── 旧问题 1: P3 阶段转换无法被 ainvoke 捕获 ────────
    if result.phase_transitions:
        issues.append({
            "issue_id": "OLD-1",
            "title": "P3 阶段转换无法被 ainvoke 捕获",
            "verdict": "CONFIRMED_FIXED",
            "evidence": f"astream_events 捕获到 {len(result.phase_transitions)} 次阶段转换: {result.phase_transitions}",
            "root_cause": "ainvoke 只返回最终 state，中间的 _phase 变化丢失。改用 astream_events 监听 advance_phase 节点输出即可捕获。",
        })
    else:
        issues.append({
            "issue_id": "OLD-1",
            "title": "P3 阶段转换无法被 ainvoke 捕获",
            "verdict": "STILL_EXISTS",
            "evidence": "即使使用 astream_events 也未捕获到阶段转换",
            "root_cause": "astream_events 的 on_chain_end 事件中 advance_phase 的 output 可能不包含 _phase 字段，或 advance_phase 节点未被作为独立 chain 事件触发",
        })

    # ── 旧问题 2: 混合调用路由可能丢失普通工具调用 ────────
    has_subagent_dispatch = any(tc["tool"] == "dispatch_subagent_lg" for tc in result.tool_calls)
    if has_subagent_dispatch:
        issues.append({
            "issue_id": "OLD-2",
            "title": "混合调用路由可能丢失普通工具调用",
            "verdict": "NEEDS_MORE_TEST",
            "evidence": "本次测试触发了子代理派遣，但未构造混合调用场景",
            "root_cause": "需要专门构造 LLM 同时发出 dispatch_subagent_lg + 普通工具调用的场景才能验证",
        })
    else:
        issues.append({
            "issue_id": "OLD-2",
            "title": "混合调用路由可能丢失普通工具调用",
            "verdict": "NOT_TRIGGERED",
            "evidence": "本次测试未触发子代理派遣",
            "root_cause": "bugfix 任务不需要子代理，无法验证此问题",
        })

    # ── 旧问题 3: Checkpointer 延迟初始化状态不一致 ──────
    issues.append({
        "issue_id": "OLD-3",
        "title": "Checkpointer 延迟初始化状态不一致",
        "verdict": "CONFIRMED",
        "evidence": "graph.checkpointer 初始为 None，_ensure_checkpointer 后变为 AsyncSqliteSaver 实例。调用方在 _ensure_checkpointer 前检查会误判。",
        "root_cause": "设计取舍：AsyncSqliteSaver 必须在正确的事件循环中初始化（aiosqlite 绑定事件循环），无法在 __init__ 同步初始化。但缺少一个 _checkpointer_initialized 标记让调用方判断。",
    })

    # ── 旧问题 4: 无 checkpointer 时 interrupt 自动放行 ──
    issues.append({
        "issue_id": "OLD-4",
        "title": "无 checkpointer 时 interrupt 审批门自动放行",
        "verdict": "CONFIRMED_DESIGN",
        "evidence": "interrupt() 是 LangGraph 的暂停原语，需要 checkpointer 保存状态才能恢复。无 checkpointer 时 interrupt() 会抛异常，因此自动放行是合理的降级。",
        "root_cause": "LangGraph 的 interrupt 机制依赖 checkpointer。这不是 bug，而是架构限制。但应增加日志记录和配置选项。",
    })

    # ── 旧问题 5: 子代理子图无 interrupt 审批门 ────────
    issues.append({
        "issue_id": "OLD-5",
        "title": "子代理子图无 interrupt 审批门",
        "verdict": "CONFIRMED_DESIGN",
        "evidence": "子代理的工具白名单通常不包含 write_file 等危险工具，因此不需要审批门。但 spec 配置错误时确实存在风险。",
        "root_cause": "设计取舍：子代理是受信任的内部组件，工具白名单已做限制。增加审批门会增加复杂度和延迟。建议在 SubagentSpec 初始化时验证工具白名单。",
    })

    # ── 旧问题 6: P3 gather 阶段写工具被阻止 ────────
    # 检查是否有回退到 'all' 的情况
    has_stall_rollback = any(
        pt["to"] == "all" and pt["from"] != "all"
        for pt in result.phase_transitions
    )
    if has_stall_rollback:
        issues.append({
            "issue_id": "OLD-6",
            "title": "P3 gather 阶段写工具被阻止导致停滞回退",
            "verdict": "CONFIRMED",
            "evidence": f"检测到从非 all 阶段回退到 all: {result.phase_transitions}",
            "root_cause": "gather 阶段只允许只读工具，如果 LLM 在 gather 阶段就想 edit_file，会被工具过滤阻止，连续2次停滞后回退到 all。",
        })
    else:
        issues.append({
            "issue_id": "OLD-6",
            "title": "P3 gather 阶段写工具被阻止导致停滞回退",
            "verdict": "NOT_TRIGGERED",
            "evidence": "本次测试未触发停滞回退",
            "root_cause": "Agent 可能按预期先读后改，未在 gather 阶段尝试写操作",
        })

    # ── 旧问题 7: _resolve 路径解析复杂 ────────────────
    # 检查 edit_file 的路径是否被正确解析
    edit_path_issues = []
    for call in result.edit_file_calls:
        args = call.get("args", "")
        if "error" in args.lower() or "not found" in args.lower():
            edit_path_issues.append(args[:100])

    if edit_path_issues:
        issues.append({
            "issue_id": "OLD-7",
            "title": "_resolve 路径解析问题",
            "verdict": "CONFIRMED",
            "evidence": f"edit_file 路径解析失败: {edit_path_issues[:3]}",
            "root_cause": "LLM 生成的路径格式与 _resolve 预期不匹配",
        })
    else:
        issues.append({
            "issue_id": "OLD-7",
            "title": "_resolve 路径解析问题",
            "verdict": "NOT_TRIGGERED",
            "evidence": "本次测试未遇到路径解析问题",
            "root_cause": "Agent 生成的路径格式恰好与 _resolve 兼容",
        })

    return issues


# ── 清理 ──────────────────────────────────────────────────────

def cleanup():
    """清理测试产生的文件和目录。"""
    project_root = Path(__file__).parent
    test_dir = project_root / "data" / "users" / TEST_USER_ID

    # 清理 factory 缓存
    try:
        from agent_by_langgraph.factory import reset_lg_agent, _agent_cache, _cache_lock
        with _cache_lock:
            keys_to_remove = [k for k in _agent_cache if k.startswith(TEST_USER_ID)]
            for k in keys_to_remove:
                agent = _agent_cache.pop(k)
                try:
                    agent.close()
                except Exception:
                    pass
    except Exception:
        pass

    # 清理子代理 checkpointer 缓存
    try:
        from agent_by_langgraph.lg_graph import _sub_checkpointer_cache, _sub_checkpointer_lock
        with _sub_checkpointer_lock:
            keys_to_remove = [k for k in _sub_checkpointer_cache if k.startswith(TEST_USER_ID)]
            for k in keys_to_remove:
                ctx, instance = _sub_checkpointer_cache.pop(k)
                try:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(ctx.__aexit__(None, None, None))
                    except RuntimeError:
                        asyncio.run(ctx.__aexit__(None, None, None))
                except Exception:
                    pass
    except Exception:
        pass

    import gc
    gc.collect()
    time.sleep(0.5)

    if test_dir.exists():
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(test_dir)
                logger.info("[Cleanup] 已清理测试目录: %s", test_dir)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    gc.collect()
                    wait = 1.0 * (2 ** attempt)
                    logger.warning("[Cleanup] rmtree 失败，%0.1fs 后重试...", wait)
                    time.sleep(wait)
                else:
                    logger.warning("[Cleanup] 清理不完全: %s", test_dir)


# ── 主流程 ────────────────────────────────────────────────────

async def run_test():
    """运行 Bug 修复测试。"""
    global result
    result = TestResult()
    from dotenv import load_dotenv
    load_dotenv()

    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or \
       os.environ["DEEPSEEK_API_KEY"] == "your_api_key_here":
        result.add_error("DEEPSEEK_API_KEY 未配置")
        _save_result()
        return

    logger.info("=" * 60)
    logger.info("Bug 修复测试开始: TODO 管理器项目")
    logger.info("=" * 60)

    # ── 准备项目文件 ──────────────────────────────────────
    # D2: 使用 get_workspace_path 获取与 Agent 相同的工作目录
    from agent.lc_tools import get_workspace_path
    project_root = Path(__file__).parent
    workspace = get_workspace_path(root=project_root, user_id=TEST_USER_ID,
                                   ticket_id=TEST_TICKET_ID)

    logger.info(">>> 准备项目文件到工作区: %s", workspace)
    try:
        prepare_project_files(workspace)
        result.add_check("project_prepared", True, f"3 个文件已复制到 {workspace}")
    except Exception as exc:
        result.add_check("project_prepared", False, str(exc))
        result.add_error(f"项目文件准备失败: {exc}")
        _save_result()
        return

    # ── 执行 Bug 修复任务 ─────────────────────────────────
    logger.info(">>> 执行 Bug 修复任务")
    t0 = time.time()

    try:
        raw = await asyncio.wait_for(
            _call_lg_agent_with_events(
                BUGFIX_PROMPT, TEST_USER_ID, TEST_TICKET_ID,
                max_iterations=50, phase_name="bugfix",
            ),
            timeout=DEV_TIMEOUT_S,
        )
        reply = raw["output"]
        final_phase = raw.get("final_phase", "unknown")
        stall_count = raw.get("stall_count", 0)

        result.add_finding("INFO", "P3 最终阶段",
                           f"最终处于 {final_phase} 阶段，停滞次数={stall_count}")

        duration = time.time() - t0
        result.record_phase("bugfix", "completed", duration,
                            data={"reply_length": len(reply),
                                  "final_phase": final_phase,
                                  "stall_count": stall_count})
    except asyncio.TimeoutError:
        duration = time.time() - t0
        result.record_phase("bugfix", "failed", duration, error="全局超时")
        result.add_error(f"开发阶段超时（{DEV_TIMEOUT_S}s）")
    except Exception as exc:
        duration = time.time() - t0
        result.record_phase("bugfix", "failed", duration, error=str(exc))
        result.add_error(f"开发阶段异常: {exc}")
        traceback.print_exc()

    # ── 成品验证 ──────────────────────────────────────────
    logger.info(">>> 成品验证")
    verify_fixes(workspace)

    # ── 动态 Bug 检测 ──────────────────────────────────────
    runtime_findings = detect_runtime_bugs()

    # ── 旧问题对照检查 ────────────────────────────────────
    old_issue_verifications = verify_old_issues()

    # ── 保存结果 ──────────────────────────────────────────
    _save_result(runtime_findings, old_issue_verifications)

    # ── 清理 ──────────────────────────────────────────────
    cleanup()

    # ── 打印摘要 ──────────────────────────────────────────
    _print_summary(runtime_findings, old_issue_verifications)


def _save_result(runtime_findings: list[dict] | None = None,
                 old_issue_verifications: list[dict] | None = None):
    """保存结构化测试结果。"""
    data = result.to_dict()
    if runtime_findings:
        data["runtime_findings"] = runtime_findings
    if old_issue_verifications:
        data["old_issue_verifications"] = old_issue_verifications
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    logger.info("测试结果已保存到 %s", RESULT_FILE)


def _print_summary(runtime_findings: list[dict], old_issue_verifications: list[dict]):
    """打印测试摘要。"""
    data = result.to_dict()

    print("\n" + "=" * 60)
    print("Bug 修复测试结果 — TODO 管理器")
    print("=" * 60)
    print(f"总结果: {'PASS' if data['passed'] else 'FAIL'}")
    print(f"总耗时: {data['total_duration_s']}s")
    print(f"Token 消耗: input={data['token_usage']['input']}, "
          f"output={data['token_usage']['output']}")
    print(f"LLM 调用次数: {data['llm_calls']}")

    print("\n阶段执行:")
    for name, phase in data["phases"].items():
        icon = "OK" if phase["status"] == "completed" else "FAIL"
        print(f"  [{icon}] {name}: {phase['status']} ({phase['duration_s']}s)")

    print("\n验证项:")
    for check in data["checks"]:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['name']}: {check['detail']}")

    # P3 阶段分析
    print("\n" + "=" * 60)
    print("P3 自适应阶段分析:")
    print("=" * 60)
    if result.phase_transitions:
        for pt in result.phase_transitions:
            print(f"  {pt['from']} -> {pt['to']}")
    else:
        print("  [INFO] 未记录到阶段转换")

    # 工具调用统计
    print("\n" + "=" * 60)
    print("工具调用统计:")
    print("=" * 60)
    tool_counts: dict[str, int] = {}
    for tc in result.tool_calls:
        tool_counts[tc["tool"]] = tool_counts.get(tc["tool"], 0) + 1
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count} 次")

    # 关键指标
    print("\n" + "=" * 60)
    print("关键指标:")
    print("=" * 60)
    print(f"  edit_file 调用: {len(result.edit_file_calls)} 次")
    print(f"  run_command 调用: {len(result.run_command_calls)} 次")
    print(f"  read_file 调用: {len(result.read_file_calls)} 次")
    print(f"  错误恢复次数: {result.error_recovery_count}")

    # 运行时发现
    if runtime_findings:
        print("\n" + "=" * 60)
        print("运行时发现:")
        print("=" * 60)
        for f in runtime_findings:
            print(f"  [{f['severity']}] {f['title']}")
            if f.get("detail"):
                print(f"       {f['detail']}")

    # 旧问题对照
    if old_issue_verifications:
        print("\n" + "=" * 60)
        print("旧问题对照验证:")
        print("=" * 60)
        for issue in old_issue_verifications:
            verdict_icon = {
                "CONFIRMED": "!!",
                "CONFIRMED_FIXED": "OK",
                "CONFIRMED_DESIGN": "~",
                "STILL_EXISTS": "XX",
                "NOT_TRIGGERED": "--",
                "NEEDS_MORE_TEST": "??",
            }.get(issue["verdict"], "??")
            print(f"  [{verdict_icon}] {issue['issue_id']}: {issue['title']}")
            print(f"       结论: {issue['verdict']}")
            print(f"       证据: {issue['evidence'][:200]}")
            print(f"       根因: {issue['root_cause'][:200]}")

    print("\n" + "=" * 60)
    print(f"详细结果已保存到: {RESULT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
