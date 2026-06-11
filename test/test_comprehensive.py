"""5号综合能力测试 —— 全栈项目开发 + 迭代验证，全面检验 Agent 能力。

与现有4个测试的差异化定位：
  - test_e2e_simple: 简单计算器，三大分析→开发，不触发子代理/迭代修复
  - test_cultivation_task: 创意HTML，只用write_file，无read/edit/run
  - test_bugfix_task: 代码修复，read→edit→run循环，无分析阶段/子代理
  - test_webpage_task: 多文件网页，三大分析+开发，无代码执行验证

本测试覆盖全部8大维度：
  1. P3阶段推进: gather→modify→verify完整链路
  2. 工具使用多样性: 6+种工具被调用（read/write/edit/run/grep/glob/web_fetch/update_todos）
  3. 迭代修复能力: run_command失败后能edit_file修复
  4. 子代理派遣: dispatch_subagent_lg被调用
  5. 多文件产出: 3个文件全部创建（主程序+测试+文档）
  6. 代码可运行性: python test_weather_app.py 通过
  7. interrupt审批: 危险工具触发interrupt并自动批准
  8. Token/性能: 合理的token消耗和迭代次数

任务：开发一个「Python CLI 天气查询应用」
  - 主程序: weather_app.py（使用免费API，含错误处理、缓存、多城市查询）
  - 测试文件: test_weather_app.py（单元测试，mock外部API）
  - 文档: README.md（安装说明、使用示例、API说明）

使用方式：
    python test_comprehensive.py

前置条件：
    - .env 文件中已配置 DEEPSEEK_API_KEY
    - 依赖已安装
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

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Windows 编码修复（必须在 logging.basicConfig 之前）
if sys.platform == "win32":
    os.system('chcp 65001 >nul 2>&1')
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_comprehensive.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_comprehensive")

# ── 常量 ──────────────────────────────────────────────────────

TEST_USER_ID = "test_comprehensive_user"
TEST_TICKET_ID = "comprehensive_task_001"
GLOBAL_TIMEOUT_S = 900       # 15 分钟硬上限
DEV_TIMEOUT_S = 600          # 开发阶段超时
RESULT_FILE = Path("test_comprehensive_result.json")

# 综合任务：开发一个 Python CLI 天气查询应用
COMPREHENSIVE_PROMPT = """\
你需要开发一个 Python CLI 天气查询应用。这是一个完整的项目，包含主程序、测试和文档。

## 项目需求

### 主程序 weather_app.py
实现一个命令行天气查询工具，功能如下：
1. 查询单个城市天气：python weather_app.py Beijing
2. 查询多个城市天气：python weather_app.py Beijing Shanghai Guangzhou
3. 支持 --format json 输出 JSON 格式
4. 支持 --cache-minutes 30 设置缓存时间
5. 使用 wttr.in 免费 API（无需 API Key）：https://wttr.in/{city}?format=j1
6. 内置简单文件缓存（JSON 文件存储，默认缓存30分钟）
7. 友好的错误处理（网络错误、城市不存在等）
8. 只使用 Python 标准库（urllib, json, argparse, os, time, datetime）

### 测试文件 test_weather_app.py
使用 unittest 编写测试：
1. 测试天气查询功能（mock urllib.request，不真正调用API）
2. 测试缓存功能（缓存写入、读取、过期）
3. 测试命令行参数解析
4. 测试错误处理（网络错误、城市不存在）
5. 至少 8 个测试用例
6. 使用 python test_weather_app.py 即可运行（不需要 pytest）

### 文档 README.md
1. 项目简介
2. 功能列表
3. 安装说明（无第三方依赖）
4. 使用示例（至少3个）
5. API 说明（wttr.in 接口说明）
6. 项目结构

## 工作要求

1. **先规划再编码**：先理清项目结构，再逐个创建文件
2. **创建文件后要验证**：
   - 用 read_file 检查文件内容是否正确
   - 用 run_command 运行 python test_weather_app.py 验证测试
   - 如果测试失败，用 edit_file 修复问题，然后再次运行测试
3. **迭代修复**：如果首次测试不通过，必须修复直到通过
4. **最终确认**：确保 python test_weather_app.py 全部通过后再输出完成摘要

⚠️ 重要约束：
- 必须使用 write_file 创建文件
- 必须使用 run_command 运行测试验证
- 如果测试失败，必须使用 edit_file 修复（不要用 write_file 重写整个文件）
- 最终输出 JSON 摘要：{"project_name": "天气查询应用", "files_created": [...], "test_passed": true}
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
        # 工具分类追踪
        self.edit_file_calls: list[dict] = []
        self.run_command_calls: list[dict] = []
        self.read_file_calls: list[dict] = []
        self.write_file_calls: list[dict] = []
        self.grep_glob_calls: list[dict] = []
        self.web_fetch_calls: list[dict] = []
        self.subagent_dispatch_calls: list[dict] = []
        self.update_todos_calls: list[dict] = []
        # 迭代追踪
        self.iteration_count = 0
        self.error_recovery_count = 0
        # interrupt 追踪
        self.interrupt_count = 0
        self.interrupt_auto_approved = 0

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
        entry = {"tool": tool_name, "args": args_summary, "time": time.time() - self.start_time}
        if tool_name == "edit_file":
            self.edit_file_calls.append(entry)
        elif tool_name == "run_command":
            self.run_command_calls.append(entry)
        elif tool_name == "read_file":
            self.read_file_calls.append(entry)
        elif tool_name == "write_file":
            self.write_file_calls.append(entry)
        elif tool_name in ("grep_tool", "glob_tool"):
            self.grep_glob_calls.append(entry)
        elif tool_name == "web_fetch":
            self.web_fetch_calls.append(entry)
        elif tool_name == "dispatch_subagent_lg":
            self.subagent_dispatch_calls.append(entry)
        elif tool_name == "update_todos":
            self.update_todos_calls.append(entry)

    def to_dict(self) -> dict:
        total_duration = time.time() - self.start_time
        all_checks_passed = all(c["passed"] for c in self.checks)
        all_phases_ok = all(
            p["status"] == "completed" for p in self.phases.values()
        )
        self.passed = all_checks_passed and all_phases_ok and not self.errors

        # 8大维度评分
        dimension_scores = self._compute_dimension_scores()

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
            "tool_summary": {
                "write_file": len(self.write_file_calls),
                "read_file": len(self.read_file_calls),
                "edit_file": len(self.edit_file_calls),
                "run_command": len(self.run_command_calls),
                "grep_glob": len(self.grep_glob_calls),
                "web_fetch": len(self.web_fetch_calls),
                "dispatch_subagent_lg": len(self.subagent_dispatch_calls),
                "update_todos": len(self.update_todos_calls),
            },
            "iteration_count": self.iteration_count,
            "error_recovery_count": self.error_recovery_count,
            "interrupt_count": self.interrupt_count,
            "interrupt_auto_approved": self.interrupt_auto_approved,
            "dimension_scores": dimension_scores,
        }

    def _compute_dimension_scores(self) -> dict:
        """计算8大维度评分。"""
        scores = {}

        # 维度1: P3阶段推进
        phases_seen = set()
        for pt in self.phase_transitions:
            phases_seen.add(pt["from"])
            phases_seen.add(pt["to"])
        has_gather = "gather" in phases_seen
        has_modify = "modify" in phases_seen
        has_verify = "verify" in phases_seen
        if has_gather and has_modify and has_verify:
            scores["D1_P3_阶段推进"] = {"score": "PASS", "detail": "gather→modify→verify完整链路"}
        elif has_gather and has_modify:
            scores["D1_P3_阶段推进"] = {"score": "PARTIAL", "detail": f"缺少verify阶段, 已见: {phases_seen}"}
        elif has_gather or has_modify:
            scores["D1_P3_阶段推进"] = {"score": "PARTIAL", "detail": f"阶段不完整, 已见: {phases_seen}"}
        else:
            scores["D1_P3_阶段推进"] = {"score": "FAIL", "detail": "未检测到P3阶段转换"}

        # 维度2: 工具使用多样性
        unique_tools = set(tc["tool"] for tc in self.tool_calls)
        tool_count = len(unique_tools)
        if tool_count >= 6:
            scores["D2_工具多样性"] = {"score": "PASS", "detail": f"{tool_count}种工具: {unique_tools}"}
        elif tool_count >= 4:
            scores["D2_工具多样性"] = {"score": "PARTIAL", "detail": f"{tool_count}种工具: {unique_tools}"}
        else:
            scores["D2_工具多样性"] = {"score": "FAIL", "detail": f"仅{tool_count}种工具: {unique_tools}"}

        # 维度3: 迭代修复能力
        has_run = len(self.run_command_calls) > 0
        has_edit = len(self.edit_file_calls) > 0
        if has_run and has_edit:
            scores["D3_迭代修复"] = {"score": "PASS", "detail": f"run_command {len(self.run_command_calls)}次, edit_file {len(self.edit_file_calls)}次"}
        elif has_run:
            scores["D3_迭代修复"] = {"score": "PARTIAL", "detail": f"有run_command({len(self.run_command_calls)}次)但无edit_file修复"}
        else:
            scores["D3_迭代修复"] = {"score": "FAIL", "detail": "未使用run_command验证"}

        # 维度4: 子代理派遣
        if self.subagent_dispatch_calls:
            scores["D4_子代理派遣"] = {"score": "PASS", "detail": f"dispatch_subagent_lg {len(self.subagent_dispatch_calls)}次"}
        else:
            scores["D4_子代理派遣"] = {"score": "FAIL", "detail": "未触发子代理派遣"}

        # 维度5: 多文件产出
        expected_files = {"weather_app.py", "test_weather_app.py", "README.md"}
        found_files = set()
        for f in self.files_created:
            fname = Path(f).name
            if fname in expected_files:
                found_files.add(fname)
        if found_files == expected_files:
            scores["D5_多文件产出"] = {"score": "PASS", "detail": f"3/3文件: {found_files}"}
        elif len(found_files) >= 2:
            scores["D5_多文件产出"] = {"score": "PARTIAL", "detail": f"{len(found_files)}/3文件: {found_files}, 缺: {expected_files - found_files}"}
        else:
            scores["D5_多文件产出"] = {"score": "FAIL", "detail": f"仅{len(found_files)}/3文件: {found_files}"}

        # 维度6: 代码可运行性（从checks中提取）
        test_passed = any(
            c["name"] == "tests_pass" and c["passed"]
            for c in self.checks
        )
        syntax_ok = any(
            c["name"] == "weather_app_syntax" and c["passed"]
            for c in self.checks
        )
        if test_passed:
            scores["D6_代码可运行性"] = {"score": "PASS", "detail": "测试全部通过"}
        elif syntax_ok:
            scores["D6_代码可运行性"] = {"score": "PARTIAL", "detail": "语法正确但测试未通过"}
        else:
            scores["D6_代码可运行性"] = {"score": "FAIL", "detail": "代码无法运行"}

        # 维度7: interrupt审批
        if self.interrupt_count > 0:
            scores["D7_Interrupt审批"] = {"score": "PASS", "detail": f"触发{self.interrupt_count}次interrupt, 自动批准{self.interrupt_auto_approved}次"}
        else:
            scores["D7_Interrupt审批"] = {"score": "SKIP", "detail": "无checkpointer时interrupt自动放行（设计预期）"}

        # 维度8: Token/性能
        total_tokens = self.token_usage["input"] + self.token_usage["output"]
        if total_tokens < 100_000:
            scores["D8_Token性能"] = {"score": "PASS", "detail": f"总计{total_tokens} tokens, {self.llm_calls}次LLM调用"}
        elif total_tokens < 200_000:
            scores["D8_Token性能"] = {"score": "PARTIAL", "detail": f"总计{total_tokens} tokens（偏高）, {self.llm_calls}次LLM调用"}
        else:
            scores["D8_Token性能"] = {"score": "FAIL", "detail": f"总计{total_tokens} tokens（过高）, {self.llm_calls}次LLM调用"}

        return scores


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
        logger.info("[ToolEnd] %s", str(output)[:150])

    def on_tool_error(self, error, **kwargs):
        self._result.add_finding("ERROR", "工具执行错误", str(error)[:200])
        self._result.error_recovery_count += 1


# ── Agent 调用（astream_events 监听全流程） ────────────────────

async def _call_lg_agent_with_events(prompt: str, user_id: str, ticket_id: str,
                                     max_iterations: int = 60,
                                     phase_name: str = "") -> dict:
    """调用 LangGraph Agent，使用 astream_events 监听全流程。

    捕获：
    - advance_phase 节点输出中的 _phase 变化 → P3 阶段转换
    - 每个图节点的执行 → node_counts
    - interrupt 事件 → 审批门行为
    - 工具调用详情
    """
    from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
    from agent_by_langgraph.lg_agent import ReasoningCollector
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

    agent = create_lg_agent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations=max_iterations,
    )

    # 启用 checkpointer 以观察 interrupt 行为
    has_checkpointer = agent.will_have_checkpointer
    logger.info("[Agent] checkpointer 初始状态: %s", has_checkpointer)

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
    has_checkpointer = agent.checkpointer_ready
    logger.info("[Agent] checkpointer 初始化后: %s", has_checkpointer)

    config: RunnableConfig = {
        "callbacks": all_callbacks,
        "recursion_limit": max_iterations * 4 + 10,
        "configurable": {
            "thread_id": f"{user_id}_{phase_name}" if phase_name else user_id,
            "__has_checkpointer__": has_checkpointer,
        },
    }

    # ── astream_events 监听 ──────────────────────────────
    last_phase = "all"
    invoke_result = None

    _TRACKED_NODES = {
        "agent", "advance_phase", "tools", "interrupt_approval",
        "aggregate_results", "subagent_dispatcher", "subagent_worker",
        "planner", "route_after_agent", "route_after_aggregate",
        "route_after_approval",
    }

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
                if kind == "on_chain_start" and name in _TRACKED_NODES:
                    result.record_node(name)

                # 捕获 P3 阶段转换
                if kind == "on_chain_end" and name == "advance_phase":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        new_phase = output.get("_phase")
                        if new_phase and new_phase != last_phase:
                            result.record_phase_transition(last_phase, new_phase)
                            last_phase = new_phase

                # 捕获 interrupt 审批事件
                if kind == "on_chain_start" and name == "interrupt_approval":
                    result.interrupt_count += 1
                    logger.info("[Event] interrupt_approval 节点启动 (第%d次)", result.interrupt_count)

                # 捕获 planner 输出
                if kind == "on_chain_end" and name == "planner":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        plan = output.get("plan", "")
                        phase_hint = output.get("_phase", "")
                        logger.info("[Event] planner: phase=%s, plan=%s",
                                    phase_hint, plan[:200] if plan else "空")

                # 捕获子代理派遣
                if kind == "on_chain_start" and name == "subagent_dispatcher":
                    logger.info("[Event] subagent_dispatcher 启动")
                if kind == "on_chain_start" and name == "subagent_worker":
                    logger.info("[Event] subagent_worker 启动")

            # astream_events 完成，获取最终 state
            if has_checkpointer:
                snapshot = await agent.graph.aget_state(config)
                invoke_result = {
                    "messages": snapshot.values.get("messages", []),
                    "_phase": snapshot.values.get("_phase", "unknown"),
                    "_stall_count": snapshot.values.get("_stall_count", 0),
                }
            else:
                invoke_result = {"messages": [], "_phase": last_phase, "_stall_count": 0}

        except Exception as exc:
            result.add_error(f"Agent 执行异常: {exc}")
            traceback.print_exc()
            try:
                reset_lg_agent(user_id, ticket_id)
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
                            result.interrupt_auto_approved += 1
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
        reset_lg_agent(user_id, ticket_id)
    except Exception:
        pass

    return {"output": reply, "final_phase": final_phase, "stall_count": stall_count}


# ── 成品验证 ──────────────────────────────────────────────────

def verify_product(workspace: Path) -> None:
    """验证开发结果。"""
    logger.info(">>> 成品验证: %s", workspace)

    # ── 1. 文件存在性检查 ──────────────────────────────────
    weather_py = workspace / "weather_app.py"
    test_py = workspace / "test_weather_app.py"
    readme_md = workspace / "README.md"

    # 也扫描子目录（Agent 可能创建在子目录中）
    if not weather_py.exists():
        for p in workspace.rglob("weather_app.py"):
            weather_py = p
            break
    if not test_py.exists():
        for p in workspace.rglob("test_weather_app.py"):
            test_py = p
            break
    if not readme_md.exists():
        for p in workspace.rglob("README.md"):
            readme_md = p
            break

    result.add_check("weather_app_exists", weather_py.exists(), str(weather_py))
    result.add_check("test_weather_app_exists", test_py.exists(), str(test_py))
    result.add_check("readme_exists", readme_md.exists(), str(readme_md))

    # 收集所有已创建的文件
    found_files = []
    if workspace.exists():
        for p in workspace.rglob("*"):
            if p.is_file() and p.name != "checkpoints.db" and "checkpoints" not in str(p):
                found_files.append(p)
    result.files_created = [str(f) for f in found_files]

    # ── 2. 主程序代码质量检查 ──────────────────────────────
    if weather_py.exists():
        content = weather_py.read_text(encoding="utf-8", errors="replace")

        # 语法检查
        import py_compile
        try:
            py_compile.compile(str(weather_py), doraise=True)
            result.add_check("weather_app_syntax", True, "Python 语法正确")
        except py_compile.PyCompileError as e:
            result.add_check("weather_app_syntax", False, f"语法错误: {e}")

        # 关键功能检查
        has_argparse = "argparse" in content
        has_urllib = "urllib" in content
        has_cache = "cache" in content.lower()
        has_error_handling = "try" in content and "except" in content
        has_main = 'if __name__' in content

        result.add_check("weather_app_argparse", has_argparse, "使用 argparse 解析命令行参数")
        result.add_check("weather_app_urllib", has_urllib, "使用 urllib 请求 API")
        result.add_check("weather_app_cache", has_cache, "包含缓存功能")
        result.add_check("weather_app_error_handling", has_error_handling, "包含错误处理")
        result.add_check("weather_app_main", has_main, "包含 if __name__ 入口")

        # 检查是否使用标准库
        has_requests = "import requests" in content
        if has_requests:
            result.add_finding("WARN", "使用了第三方库 requests",
                               "需求要求只用标准库，但代码 import requests")
    else:
        result.add_finding("ERROR", "weather_app.py 不存在", "Agent 未创建主程序文件")

    # ── 3. 测试文件质量检查 ────────────────────────────────
    if test_py.exists():
        content = test_py.read_text(encoding="utf-8", errors="replace")

        has_unittest = "unittest" in content or "TestCase" in content
        has_mock = "mock" in content.lower() or "patch" in content
        test_count = len(re.findall(r'def test_', content))

        result.add_check("test_uses_unittest", has_unittest, "使用 unittest 框架")
        result.add_check("test_uses_mock", has_mock, "使用 mock 避免真实 API 调用")
        result.add_check("test_count_sufficient", test_count >= 5,
                         f"测试用例数量: {test_count} (要求>=5)")

        # 语法检查
        try:
            py_compile.compile(str(test_py), doraise=True)
            result.add_check("test_syntax", True, "测试文件语法正确")
        except py_compile.PyCompileError as e:
            result.add_check("test_syntax", False, f"测试文件语法错误: {e}")
    else:
        result.add_finding("ERROR", "test_weather_app.py 不存在", "Agent 未创建测试文件")

    # ── 4. README 质量检查 ─────────────────────────────────
    if readme_md.exists():
        content = readme_md.read_text(encoding="utf-8", errors="replace")

        has_install = any(kw in content.lower() for kw in ["安装", "install", "运行", "用法", "usage"])
        has_example = any(kw in content.lower() for kw in ["示例", "example", "用法"])
        has_api = any(kw in content.lower() for kw in ["api", "接口", "wttr"])

        result.add_check("readme_install", has_install, "包含安装/运行说明")
        result.add_check("readme_example", has_example, "包含使用示例")
        result.add_check("readme_api", has_api, "包含 API 说明")
    else:
        result.add_finding("WARN", "README.md 不存在", "Agent 未创建文档")

    # ── 5. 运行测试验证 ────────────────────────────────────
    if test_py.exists():
        logger.info(">>> 运行测试: python %s", test_py)
        try:
            import subprocess
            proc = subprocess.run(
                [sys.executable, str(test_py)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(test_py.parent),
            )
            test_output = proc.stdout + proc.stderr
            test_passed = proc.returncode == 0

            result.add_check("tests_pass", test_passed,
                             f"exit_code={proc.returncode}, output={test_output[:300]}")

            if not test_passed:
                result.add_finding("WARN", "测试未全部通过",
                                   f"退出码={proc.returncode}, 输出={test_output[:500]}")
                # 检查是否至少部分通过
                passed_match = re.search(r'(\d+) test', test_output)
                if passed_match:
                    result.add_finding("INFO", "部分测试结果", test_output[:300])
            else:
                # 统计通过的测试数量
                ok_match = re.search(r'(\d+)\s*(?:test|OK)', test_output)
                if ok_match:
                    result.add_finding("INFO", "测试通过",
                                       f"{ok_match.group(0)} 个测试通过")
        except subprocess.TimeoutExpired:
            result.add_check("tests_pass", False, "测试运行超时(60s)")
            result.add_finding("ERROR", "测试运行超时", "可能存在死循环或阻塞")
        except Exception as exc:
            result.add_check("tests_pass", False, f"运行测试失败: {exc}")
    else:
        result.add_check("tests_pass", False, "测试文件不存在，无法运行")

    # ── 6. 主程序导入检查 ──────────────────────────────────
    if weather_py.exists():
        try:
            import importlib.util
            import unittest.mock
            spec = importlib.util.spec_from_file_location("weather_app_test", str(weather_py))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # mock sys.argv 避免 argparse 报错
                with unittest.mock.patch("sys.argv", ["weather_app.py"]):
                    spec.loader.exec_module(mod)
                result.add_check("weather_app_importable", True, "模块可加载")
            else:
                result.add_check("weather_app_importable", False, "无法创建模块规格")
        except SystemExit:
            # argparse 在无参数时可能 sys.exit，这是正常的
            result.add_check("weather_app_importable", True, "模块可加载（argparse正常退出）")
        except Exception as e:
            result.add_check("weather_app_importable", False, f"导入失败: {e}")


# ── 动态 Bug 检测 ─────────────────────────────────────────────

def detect_runtime_bugs() -> list[dict]:
    """基于运行时数据动态检测问题。"""
    findings = []

    # 1. 阶段失败检查
    for name, phase in result.phases.items():
        if phase["status"] != "completed":
            findings.append({
                "severity": "ERROR",
                "title": f"阶段 {name} 失败",
                "detail": phase.get("error", "未知错误"),
            })

    # 2. 阶段耗时异常
    for name, phase in result.phases.items():
        if phase["duration_s"] > 300:
            findings.append({
                "severity": "WARN",
                "title": f"阶段 {name} 耗时过长",
                "detail": f"{phase['duration_s']}s",
            })

    # 3. Token 消耗异常
    total_tokens = result.token_usage["input"] + result.token_usage["output"]
    if total_tokens > 200_000:
        findings.append({
            "severity": "WARN",
            "title": "Token 消耗过高",
            "detail": f"总计 {total_tokens} tokens",
        })

    # 4. 无文件产出
    if not result.files_created:
        findings.append({
            "severity": "ERROR",
            "title": "无文件产出",
            "detail": "开发阶段未生成任何文件",
        })

    # 5. 未使用 run_command 验证
    if not result.run_command_calls:
        findings.append({
            "severity": "WARN",
            "title": "未使用 run_command",
            "detail": "Agent 未运行测试验证代码",
        })

    # 6. 未使用 edit_file 修复
    if result.run_command_calls and not result.edit_file_calls:
        findings.append({
            "severity": "WARN",
            "title": "未使用 edit_file 修复",
            "detail": "Agent 运行了命令但未做迭代修复（可能首次就通过了，也可能忽略了失败）",
        })

    # 7. P3 阶段转换异常
    if not result.phase_transitions:
        findings.append({
            "severity": "WARN",
            "title": "未检测到 P3 阶段转换",
            "detail": "P3 自适应工具选择可能未正常工作",
        })
    else:
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

    # 8. LLM 调用次数过多
    if result.llm_calls > 40:
        findings.append({
            "severity": "WARN",
            "title": "LLM 调用次数过多",
            "detail": f"共 {result.llm_calls} 次，可能存在循环",
        })

    # 9. 重复工具调用模式检测
    tool_sequence = [tc["tool"] for tc in result.tool_calls]
    if len(tool_sequence) > 6:
        # 检测连续3次以上相同工具调用
        for i in range(len(tool_sequence) - 3):
            if tool_sequence[i] == tool_sequence[i+1] == tool_sequence[i+2] == tool_sequence[i+3]:
                findings.append({
                    "severity": "WARN",
                    "title": f"工具 {tool_sequence[i]} 连续调用4次+",
                    "detail": "可能陷入循环",
                })
                break

    # 10. 错误恢复
    if result.error_recovery_count > 0:
        findings.append({
            "severity": "INFO",
            "title": f"错误恢复 {result.error_recovery_count} 次",
            "detail": "Agent 在执行过程中遇到了错误并尝试恢复",
        })

    # 11. 关键断言失败
    failed_checks = [c for c in result.checks if not c["passed"]]
    if failed_checks:
        findings.append({
            "severity": "WARN",
            "title": f"{len(failed_checks)} 项验证未通过",
            "detail": ", ".join(c["name"] for c in failed_checks),
        })

    return findings


# ── 清理 ──────────────────────────────────────────────────────

def cleanup():
    """清理测试产生的文件和目录。"""
    project_root = Path(__file__).parent.parent
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

    # 轮询检查 .db 文件是否可写
    db_files = list(test_dir.rglob("*.db")) if test_dir.exists() else []
    for _ in range(10):
        all_released = True
        for db in db_files:
            try:
                with open(db, "a"):
                    pass
            except (PermissionError, OSError):
                all_released = False
                break
        if all_released:
            break
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
    """运行5号综合能力测试。"""
    global result
    result = TestResult()
    from dotenv import load_dotenv
    load_dotenv()

    # 测试环境允许无 checkpointer 时自动放行危险工具
    os.environ.setdefault("AUTO_APPROVE_WITHOUT_CHECKPOINTER", "true")

    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or \
       os.environ["DEEPSEEK_API_KEY"] == "your_api_key_here":
        result.add_error("DEEPSEEK_API_KEY 未配置！请在 .env 文件中设置。")
        _save_result()
        return

    logger.info("=" * 60)
    logger.info("5号综合能力测试开始: Python CLI 天气查询应用")
    logger.info("=" * 60)

    # ── 准备：清理旧数据 ──────────────────────────────────
    project_root = Path(__file__).parent.parent
    test_data_dir = project_root / "data" / "users" / TEST_USER_ID
    if test_data_dir.exists():
        logger.info("[Prepare] 清理旧数据: %s", test_data_dir)
        shutil.rmtree(test_data_dir, ignore_errors=True)

    # ── 执行开发任务 ──────────────────────────────────────
    logger.info(">>> 执行开发任务")
    t0 = time.time()

    try:
        raw = await asyncio.wait_for(
            _call_lg_agent_with_events(
                COMPREHENSIVE_PROMPT, TEST_USER_ID, TEST_TICKET_ID,
                max_iterations=60, phase_name="dev",
            ),
            timeout=DEV_TIMEOUT_S,
        )
        reply = raw["output"]
        final_phase = raw.get("final_phase", "unknown")
        stall_count = raw.get("stall_count", 0)

        result.add_finding("INFO", "P3 最终阶段",
                           f"最终处于 {final_phase} 阶段，停滞次数={stall_count}")

        duration = time.time() - t0
        result.record_phase("development", "completed", duration,
                            data={"reply_length": len(reply),
                                  "final_phase": final_phase,
                                  "stall_count": stall_count})
    except asyncio.TimeoutError:
        duration = time.time() - t0
        result.record_phase("development", "failed", duration, error="全局超时")
        result.add_error(f"开发阶段超时（{DEV_TIMEOUT_S}s）")
    except Exception as exc:
        duration = time.time() - t0
        result.record_phase("development", "failed", duration, error=str(exc))
        result.add_error(f"开发阶段异常: {exc}")
        traceback.print_exc()

    # ── 成品验证 ──────────────────────────────────────────
    from agent_core.tools import get_workspace_path
    workspace = get_workspace_path(root=project_root, user_id=TEST_USER_ID,
                                   ticket_id=TEST_TICKET_ID)
    logger.info(">>> 成品验证，工作区: %s", workspace)
    verify_product(workspace)

    # ── 动态 Bug 检测 ──────────────────────────────────────
    runtime_findings = detect_runtime_bugs()

    # ── 保存结果 ──────────────────────────────────────────
    _save_result(runtime_findings)

    # ── 清理 ──────────────────────────────────────────────
    cleanup()

    # ── 打印摘要 ──────────────────────────────────────────
    _print_summary(runtime_findings)


def _save_result(runtime_findings: list[dict] | None = None):
    """保存结构化测试结果。"""
    data = result.to_dict()
    if runtime_findings:
        data["runtime_findings"] = runtime_findings
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    logger.info("测试结果已保存到 %s", RESULT_FILE)


def _print_summary(runtime_findings: list[dict]):
    """打印测试摘要。"""
    data = result.to_dict()

    print("\n" + "=" * 70)
    print("5号综合能力测试结果 — Python CLI 天气查询应用")
    print("=" * 70)
    print(f"总结果: {'PASS' if data['passed'] else 'FAIL'}")
    print(f"总耗时: {data['total_duration_s']}s")
    print(f"Token 消耗: input={data['token_usage']['input']}, "
          f"output={data['token_usage']['output']}")
    print(f"LLM 调用次数: {data['llm_calls']}")

    # ── 8大维度评分 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("8大维度评分:")
    print("=" * 70)
    scores = data.get("dimension_scores", {})
    pass_count = sum(1 for v in scores.values() if v["score"] == "PASS")
    partial_count = sum(1 for v in scores.values() if v["score"] == "PARTIAL")
    fail_count = sum(1 for v in scores.values() if v["score"] == "FAIL")
    skip_count = sum(1 for v in scores.values() if v["score"] == "SKIP")
    for dim, info in scores.items():
        icon = {"PASS": "OK", "PARTIAL": "~ ", "FAIL": "XX", "SKIP": "--"}.get(info["score"], "??")
        print(f"  [{icon}] {dim}: {info['detail']}")

    print(f"\n  总计: {pass_count} PASS, {partial_count} PARTIAL, {fail_count} FAIL, {skip_count} SKIP")

    # ── 阶段执行 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("阶段执行:")
    print("=" * 70)
    for name, phase in data["phases"].items():
        icon = "OK" if phase["status"] == "completed" else "FAIL"
        print(f"  [{icon}] {name}: {phase['status']} ({phase['duration_s']}s)")

    # ── 验证项 ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("验证项:")
    print("=" * 70)
    for check in data["checks"]:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['name']}: {check['detail']}")

    # ── P3 阶段分析 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("P3 自适应阶段分析:")
    print("=" * 70)
    if result.phase_transitions:
        for pt in result.phase_transitions:
            print(f"  {pt['from']} -> {pt['to']}")
    else:
        print("  [INFO] 未记录到阶段转换")

    # ── 工具调用统计 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("工具调用统计:")
    print("=" * 70)
    tool_summary = data.get("tool_summary", {})
    for tool, count in sorted(tool_summary.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {tool}: {count} 次")
    unique_tools = sum(1 for c in tool_summary.values() if c > 0)
    print(f"  --- 共 {unique_tools} 种工具被使用 ---")

    # ── 关键指标 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("关键指标:")
    print("=" * 70)
    print(f"  write_file 调用: {len(result.write_file_calls)} 次")
    print(f"  read_file 调用: {len(result.read_file_calls)} 次")
    print(f"  edit_file 调用: {len(result.edit_file_calls)} 次")
    print(f"  run_command 调用: {len(result.run_command_calls)} 次")
    print(f"  grep/glob 调用: {len(result.grep_glob_calls)} 次")
    print(f"  web_fetch 调用: {len(result.web_fetch_calls)} 次")
    print(f"  子代理派遣: {len(result.subagent_dispatch_calls)} 次")
    print(f"  错误恢复次数: {result.error_recovery_count}")
    print(f"  interrupt 次数: {result.interrupt_count} (自动批准: {result.interrupt_auto_approved})")

    # ── 运行时发现 ────────────────────────────────────────
    if runtime_findings:
        print("\n" + "=" * 70)
        print("运行时发现:")
        print("=" * 70)
        for f in runtime_findings:
            print(f"  [{f['severity']}] {f['title']}")
            if f.get("detail"):
                print(f"       {f['detail']}")

    # ── 生成的文件 ────────────────────────────────────────
    if result.files_created:
        print("\n" + "=" * 70)
        print("生成的文件:")
        print("=" * 70)
        for f in result.files_created:
            print(f"  - {f}")

    print("\n" + "=" * 70)
    print(f"详细结果已保存到: {RESULT_FILE.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_test())
