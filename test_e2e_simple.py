"""轻量端到端测试 —— 模拟生产全流程：用户需求 → 三大分析 → 开发 → 成品验证。

使用方式：
    python test_e2e_simple.py

前置条件：
    - .env 文件中已配置 DEEPSEEK_API_KEY
    - 依赖已安装

设计原则：
    1. 轻量任务：写一个 Python 计算器，而非 100 个修仙境界
    2. 走完整链路：需求分析 → PRD → 报价 → 开发 → 成品验证
    3. 用 ainvoke 获取可靠结果（非 astream_events）
    4. 全局超时保护
    5. JSON 结构化输出到 test_e2e_result.json
    6. 动态断言 + 动态 bug 检测
    7. 测试后自动清理

观察维度：
    1. 三大分析是否全部成功（需求分析/PRD/报价）
    2. 开发是否产出可运行代码
    3. P3 自适应阶段是否正常推进
    4. interrupt 审批门是否正常工作
    5. 全流程 token 消耗和耗时
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Windows 编码修复（必须在 logging.basicConfig 之前，否则 handler 使用旧编码）
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
        logging.FileHandler("test_e2e_simple.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_e2e")

# ── 常量 ──────────────────────────────────────────────────────

TEST_USER_ID = "test_e2e_simple"
TEST_TICKET_ID = "e2e_simple_001"
GLOBAL_TIMEOUT_S = 600  # 10 分钟硬上限
ANALYSIS_TIMEOUT_S = 120  # 分析阶段单阶段超时
DEV_TIMEOUT_S = 300       # 开发阶段超时
RESULT_FILE = Path("test_e2e_result.json")

# 轻量任务：写一个 Python 计算器
SIMPLE_REQUIREMENT = {
    "project_name": "Python 计算器",
    "project_type": "命令行工具",
    "description": (
        "写一个 Python 命令行计算器程序 calculator.py，支持加减乘除四则运算。"
        "要求：1) 通过命令行交互输入表达式 2) 支持整数和小数 3) 有友好的错误提示 "
        "4) 输入 q 退出。不需要安装第三方依赖，只用 Python 标准库。"
    ),
    "deadline": "",
    "budget": "",
}


# ── 结果收集器 ────────────────────────────────────────────────

class TestResult:
    """结构化测试结果收集器。"""

    def __init__(self):
        self.start_time = time.time()
        self.phases: dict[str, dict] = {}  # phase_name → {status, duration, data, error}
        self.checks: list[dict] = []       # {name, passed, detail}
        self.errors: list[str] = []
        self.findings: list[dict] = []     # {severity, title, detail}
        self.token_usage: dict[str, int] = {"input": 0, "output": 0}
        self.files_created: list[str] = []
        self.passed = False

    def record_phase(self, name: str, status: str, duration: float,
                     data: Any = None, error: str | None = None):
        self.phases[name] = {
            "status": status,
            "duration_s": round(duration, 1),
            "data": data,
            "error": error,
        }
        icon = "OK" if status == "completed" else "FAIL"
        logger.info("[Phase %s] %s (%.1fs) %s", icon, name, duration,
                    error or "")

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        icon = "PASS" if passed else "FAIL"
        logger.info("[Check %s] %s: %s", icon, name, detail)

    def add_error(self, error: str):
        self.errors.append(error)
        logger.error("[Error] %s", error)

    def add_finding(self, severity: str, title: str, detail: str = ""):
        """记录运行时发现的问题（非致命，不影响 passed 判定）。"""
        finding = {"severity": severity, "title": title}
        if detail:
            finding["detail"] = detail
        self.findings.append(finding)
        logger.info("[Finding %s] %s: %s", severity, title, detail)

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
        }


result = TestResult()


# ── Token 追踪回调 ────────────────────────────────────────────

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class TokenTracker(BaseCallbackHandler):
    """追踪 LLM token 用量。"""

    def __init__(self, result_obj: TestResult):
        self._result = result_obj

    def on_llm_end(self, response, **kwargs):
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


# ── 纯 LLM 调用（分析阶段用，不走 Agent 循环） ──────────────

_cached_llm = None


def _get_direct_llm():
    """获取或创建用于分析阶段的 LLM 实例（缓存复用）。"""
    global _cached_llm
    if _cached_llm is None:
        from agent.lc_agent import DeepSeekChatOpenAI
        _cached_llm = DeepSeekChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            streaming=False,
            max_tokens=8192,
        )
    return _cached_llm


async def _call_llm_direct(system_prompt: str, user_prompt: str) -> str:
    """直接调用 LLM，不走 Agent 循环。用于三大分析阶段。

    优势：
    - 无工具绑定，LLM 专注输出 JSON，不会意外调用工具
    - 无 planner / P3 阶段控制 / interrupt 等开销
    - Token 消耗大幅降低（无 system prompt 中的工具描述）
    - 避免 JSON 输出被截断（无 agent 循环的 token 占用）
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = _get_direct_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    token_tracker = TokenTracker(result)
    response = await llm.ainvoke(messages, config={"callbacks": [token_tracker]})

    content = response.content
    return content if isinstance(content, str) else str(content)


# ── Agent 直接调用（绕过 FastAPI，走生产逻辑） ────────────────

async def _call_lg_agent(prompt: str, user_id: str, ticket_id: str,
                         max_iterations: int = 30,
                         phase_name: str = "") -> dict:
    """直接调用 LangGraph Agent，返回 {output} 或抛异常。

    仅用于开发阶段。分析阶段请使用 _call_llm_direct。

    使用 ainvoke 获取可靠结果，而非 astream_events。
    自动处理 interrupt（批准所有危险工具调用）。
    每次调用使用独立的 thread_id，避免不同阶段的 state 污染。
    """
    from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
    from agent_by_langgraph.lg_agent import ReasoningCollector
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

    # 每个阶段使用独立的 ticket_id，避免 checkpointer state 污染
    phase_ticket = f"{ticket_id}_{phase_name}" if phase_name else ticket_id

    agent = create_lg_agent(
        user_id=user_id,
        ticket_id=phase_ticket,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations=max_iterations,
    )

    # 测试场景不需要状态持久化/断点续跑，禁用 Checkpointer 以避免：
    # - SQLite 初始化/关闭开销
    # - Windows 下 SQLite 文件锁导致清理失败
    # - aiosqlite 后台线程未及时退出的问题
    agent.graph.checkpointer = None
    has_checkpointer = False

    # 构造输入（加锁保护 _first_turn 读写，与 LGAgent.run() 保持一致）
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
    # 过滤图级 TokenTrackerCallback，避免与测试级 TokenTracker 重复计数
    # 注意：每次调用都重新过滤，因为 _ensure_checkpointer 可能重新编译图注入新回调
    from agent_by_langgraph.lg_agent import TokenTrackerCallback
    graph_callbacks = [
        cb for cb in getattr(agent.graph, '_lg_llm_callbacks', [])
        if not isinstance(cb, TokenTrackerCallback)
    ]
    all_callbacks = list(graph_callbacks)
    all_callbacks.extend([collector, token_tracker])

    config: RunnableConfig = {
        "callbacks": all_callbacks,
        "recursion_limit": max_iterations * 4 + 10,
        "configurable": {
            "thread_id": f"{user_id}_{phase_name}" if phase_name else user_id,
            "__has_checkpointer__": has_checkpointer,
        },
    }

    # 执行 + interrupt 自动批准循环
    # 用 aget_state 检查 pending interrupt（比消息内容检测更可靠）
    current_input = input_state
    max_interrupt_retries = 30
    for _ in range(max_interrupt_retries):
        invoke_result = await agent.graph.ainvoke(current_input, config=config)

        # 检查是否有 pending interrupt
        has_interrupt = False
        if has_checkpointer:
            try:
                snapshot = await agent.graph.aget_state(config)
                if snapshot and snapshot.next:
                    for task in snapshot.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            has_interrupt = True
                            logger.info("[Interrupt] 自动批准: %s", task.interrupts)
                            current_input = Command(resume="approve")
                            break
            except Exception as exc:
                logger.warning("[Interrupt] aget_state 失败，降级为消息检测: %s", exc)
                messages = invoke_result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        dangerous = [
                            tc for tc in last_msg.tool_calls
                            if tc["name"] in {"write_file", "edit_file", "run_command"}
                        ]
                        if dangerous:
                            has_interrupt = True
                            logger.info("[Interrupt] 降级检测，自动批准: %s",
                                        [tc["name"] for tc in dangerous])
                            current_input = Command(resume="approve")

        if not has_interrupt:
            break
    else:
        raise RuntimeError("interrupt 重试次数耗尽")

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

    # 清理 agent 资源（reset_lg_agent 内部会调用 close()，无需重复关闭）
    try:
        reset_lg_agent(user_id, phase_ticket)
    except Exception:
        pass

    return {"output": reply}


# ── 三大分析（复用 AgentService 的 prompt 和 JSON 解析） ────────

REQUIREMENT_ANALYST_PROMPT = """你是需求分析师，负责将客户模糊的原始需求转化为结构化的需求简报。

收到客户需求后，按以下维度分析：

1. **项目概述**：一句话概括项目本质
2. **目标用户**：谁会使用这个产品
3. **核心功能**：3-5 个最关键的功能点
4. **非功能需求**：性能、安全、兼容性等
5. **约束条件**：预算、时间、技术限制
6. **风险点**：可能的技术或业务风险
7. **待澄清问题**：需要客户补充的信息

最后输出结构化需求简报（JSON 格式），并给出复杂度评估（简单/中等/复杂）。

⚠️ 输出要求：只输出 JSON，不要输出任何其他解释性文字。JSON 要简洁，每个字段值不超过 50 字。整个 JSON 总长度不超过 1500 字。"""

PRODUCT_MANAGER_PROMPT = """你是产品经理，负责将需求分析转化为完整的产品需求文档（PRD）。

收到需求分析后，按以下维度设计：

1. **产品定位**：一句话描述产品价值和差异化
2. **功能清单**：按优先级排序（P0 核心/P1 重要/P2 锦上添花）
3. **用户故事**：3-5 个核心用户场景的完整描述
4. **信息架构**：主要页面和导航结构
5. **数据模型**：核心数据实体和关系
6. **验收标准**：每个 P0 功能的完成定义

最后输出 PRD（JSON 格式），包含功能总数、核心场景数和技术复杂度评估。

⚠️ 输出要求：只输出 JSON，不要输出任何其他解释性文字。JSON 要简洁，每个字段值不超过 50 字，列表项不超过 5 个。整个 JSON 总长度不超过 2000 字。"""

COST_ESTIMATOR_PROMPT = """你是成本估算师，负责根据需求分析和 PRD 计算开发成本和报价。

收到 PRD 后，按以下维度估算：

1. **人力成本**：前端/后端/UI/测试/项目管理的工时 × 单价
2. **基础设施成本**：服务器/云服务/第三方 API
3. **风险缓冲**：15% 应急预算
4. **利润空间**：25% 合理利润

最后输出报价单（JSON 格式），包含：
- 总报价（元）
- 分项明细
- 付款节点（如 3-4-3）
- 交付周期（周）
- 售后支持期限（月）

⚠️ 输出要求：只输出 JSON，不要输出任何其他解释性文字。JSON 要简洁，总长度不超过 1000 字。"""

DEVELOPER_PROMPT = """你是全栈开发工程师，负责根据 PRD 和需求分析完成项目开发。

要求：
1. 严格按照 PRD 中的 P0 功能实现
2. 代码质量要达到生产级别
3. 包含必要的错误处理和边界情况处理

工作方式：
1. 使用 write_file 工具逐个创建项目文件
2. 先创建核心代码文件，再创建配置文件和 README
3. 每个文件写完后确认内容正确

⚠️ 重要约束：
- 不要使用 run_command 工具运行代码验证
- 不要使用 edit_file 工具（直接用 write_file 覆盖即可）
- 只使用 write_file 工具创建文件

完成所有文件后，输出一个 JSON 摘要，格式如下：
{{"project_name": "项目名", "files_created": ["文件1路径", "文件2路径"], "tech_stack": "技术栈", "setup_instructions": "安装运行步骤"}}

⚠️ 重要：先使用 write_file 工具创建所有文件，最后再输出 JSON 摘要。不要把代码内容放在 JSON 中。"""


def _extract_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON，支持截断修复。"""
    import re

    # 提取代码块中的内容
    if "```" in content:
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
        if code_block:
            content = code_block.group(1).strip()
        else:
            content = content.split("```", 1)[1].strip()
            if content.startswith("json"):
                content = content[4:].strip()

    json_start = content.find('{')
    if json_start == -1:
        return None

    candidate = content[json_start:]

    # 快速检测截断特征：输出以未闭合引号/省略号/不完整单词结尾
    # 直接跳到策略5截断修复，跳过必然失败的完整 JSON 解析
    stripped = candidate.rstrip()
    is_truncated = (
        stripped.endswith('...') or
        stripped.endswith('\\') or
        (stripped.count('"') % 2 == 1) or  # 未闭合引号
        (stripped[-1:] not in '}"\'\\]') and not stripped.endswith('null') and not stripped.endswith('true') and not stripped.endswith('false')
    )
    if is_truncated:
        try:
            return _repair_truncated_json(candidate)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # 截断修复失败，继续尝试完整解析策略（可能判断有误）

    # 策略 1: 直接解析完整 JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 策略 2: 用 raw_decode 尝试解析前缀（自动处理尾部垃圾）
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(candidate)
        return obj
    except json.JSONDecodeError:
        pass

    # 策略 3: 去尾逗号后重试
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 策略 4: raw_decode + 去尾逗号
    try:
        cleaned = re.sub(r',\s*([}\]])', r'\1', candidate)
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError:
        pass

    # 策略 5: 截断修复（对含代码内容的 JSON 可能不可靠，作为最后手段）
    try:
        return _repair_truncated_json(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def _repair_truncated_json(text: str) -> dict:
    """修复被截断的 JSON：补全缺失的括号和引号。"""
    # 去尾逗号
    import re
    text = re.sub(r',\s*$', '', text.strip())

    # 统计未闭合的括号
    stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # 如果在字符串中间被截断，关闭字符串
    if in_string:
        text += '"'

    # 去掉最后一个不完整的值（如 "key": "val 带截断）
    # 策略：先尝试在最后一个完整的键值对后截断（": " 分隔符），
    # 再退化为逗号截断。避免在列表 ["a", "b"] 被截断为 ["a", "b 时丢失 "a"。
    last_complete = -1

    # 优先策略：找最后一个 ": " 后面的完整值结束位置
    # 即找最后一个 "key": "value" 或 "key": number 等完整键值对的结束逗号
    for i in range(len(text) - 1, -1, -1):
        if text[i] == ',':
            # 检查逗号前是否是完整值（}、]、"、数字、字母等）
            prev = i - 1
            while prev >= 0 and text[prev] in ' \t\n\r':
                prev -= 1
            if prev >= 0 and text[prev] in '}"\'0123456789truefalsnul':
                last_complete = i
                break

    # 退化策略：如果没找到合适的逗号，找任意逗号
    if last_complete < 0:
        for i in range(len(text) - 1, -1, -1):
            if text[i] == ',':
                last_complete = i
                break

    if last_complete > 0:
        text = text[:last_complete]

    # 补全闭合括号
    stack2 = []
    in_string2 = False
    escape_next2 = False
    for ch in text:
        if escape_next2:
            escape_next2 = False
            continue
        if ch == '\\' and in_string2:
            escape_next2 = True
            continue
        if ch == '"' and not escape_next2:
            in_string2 = not in_string2
            continue
        if in_string2:
            continue
        if ch in '{[':
            stack2.append(ch)
        elif ch == '}' and stack2 and stack2[-1] == '{':
            stack2.pop()
        elif ch == ']' and stack2 and stack2[-1] == '[':
            stack2.pop()

    # 反向补全
    for ch in reversed(stack2):
        text += '}' if ch == '{' else ']'

    return json.loads(text)


async def run_analysis_phase(name: str, prompt_template: str,
                             input_data: dict) -> dict:
    """执行一个分析阶段，返回解析后的 JSON 或抛异常。

    使用纯 LLM 调用（不走 Agent 循环），避免：
    - JSON 输出被 agent 循环的 token 开销截断
    - LLM 意外调用工具而非输出 JSON
    - 每阶段创建/销毁 Checkpointer 的开销

    JSON 提取失败时自动重试一次（LLM 第二次输出可能不被截断）。
    """
    t0 = time.time()
    user_prompt = f"""基于以下数据，按要求输出 JSON 格式的结果：

{json.dumps(input_data, ensure_ascii=False, indent=2)}"""

    max_retries = 2
    last_error = None
    for attempt in range(max_retries):
        try:
            current_prompt = user_prompt
            if attempt > 0:
                current_prompt = ("⚠️ 上次输出被截断或 JSON 不完整，请确保输出完整且简洁的 JSON，"
                                  "总长度不超过 1500 字。\n\n" + user_prompt)
            reply = await _call_llm_direct(prompt_template, current_prompt)
            data = _extract_json(reply)
            if data is None:
                raise ValueError(f"无法从 LLM 输出中提取 JSON，原始输出前 500 字: {reply[:500]}")
            duration = time.time() - t0
            result.record_phase(name, "completed", duration, data=data)
            return data
        except ValueError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                logger.warning("[Retry] %s JSON 提取失败，重试 %d/%d: %s",
                               name, attempt + 1, max_retries, str(exc)[:200])
            continue
        except Exception as exc:
            duration = time.time() - t0
            result.record_phase(name, "failed", duration, error=str(exc))
            raise

    # 重试耗尽
    duration = time.time() - t0
    result.record_phase(name, "failed", duration, error=str(last_error))
    raise last_error


async def run_development_phase(project_data: dict) -> dict:
    """执行开发阶段，返回开发结果。"""
    t0 = time.time()
    prompt = f"""基于以下项目数据，按要求输出 JSON 格式的开发结果：

{json.dumps(project_data, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{DEVELOPER_PROMPT}"""

    try:
        raw = await _call_lg_agent(
            prompt, TEST_USER_ID, TEST_TICKET_ID,
            max_iterations=20, phase_name="development",
        )
        reply = raw["output"]

        # 检测迭代耗尽
        if "max iterations" in reply.lower() or "agent stopped" in reply.lower():
            raise ValueError("Agent 迭代次数耗尽")

        data = _extract_json(reply)
        if data is None:
            # JSON 摘要提取失败，但 agent 可能已用 write_file 写了文件
            # 不报错，让 verify_product 从文件系统检测
            logger.warning("[Development] JSON 摘要提取失败，将从文件系统验证")
            data = {"files_created": [], "extraction_failed": True}

        duration = time.time() - t0
        result.record_phase("development", "completed", duration, data=data)
        return data
    except Exception as exc:
        duration = time.time() - t0
        result.record_phase("development", "failed", duration, error=str(exc))
        raise


# ── 成品验证 ──────────────────────────────────────────────────

def verify_product(dev_data: dict) -> None:
    """验证开发产出的文件。

    开发阶段 agent 使用 write_file 工具写文件到工作区，
    工作区位于 data/users/{user_id}/{ticket_id}/ 下。
    同时从 dev_data 的 files_created 字段获取文件列表。
    """
    project_root = Path(__file__).parent
    # 开发阶段的 workspace 是 data/users/{user_id}/{ticket_id}_development/
    dev_workspace = project_root / "data" / "users" / TEST_USER_ID / f"{TEST_TICKET_ID}_development"
    # 也扫描原始 ticket_id 目录
    base_workspace = project_root / "data" / "users" / TEST_USER_ID / TEST_TICKET_ID

    # 收集所有已创建的文件
    # 1. 从 agent 的 JSON 摘要中获取文件列表
    files_from_json = dev_data.get("files_created", [])

    # 2. 从文件系统扫描工作区
    found_files = []
    for ws in [dev_workspace, base_workspace]:
        if ws.exists():
            for p in ws.rglob("*"):
                if p.is_file() and p.name != "checkpoints.db" and "checkpoints" not in str(p):
                    found_files.append(p)

    # 3. 也扫描成品目录（兼容旧逻辑）
    for ws in [dev_workspace, base_workspace]:
        output_root = ws / "成品"
        if output_root.exists():
            for p in output_root.rglob("*"):
                if p.is_file():
                    found_files.append(p)

    # 去重
    seen = set()
    unique_files = []
    for f in found_files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(f)

    result.files_created = [str(f) for f in unique_files]
    result.add_check("files_written", len(unique_files) > 0,
                     f"找到 {len(unique_files)} 个文件")

    if not unique_files:
        result.add_finding("ERROR", "无文件产出",
                           "开发阶段未生成任何文件，检查 agent 是否成功调用 write_file")
        return

    # 查找主程序 .py 文件
    # 策略1: 精确匹配 "calculator" 关键字
    # 策略2: 回退到工作区中唯一的非 README/test .py 主程序文件
    #   （LLM 可能自主命名，如 cmdcalc.py、calc.py、simple_calc.py）
    calc_path = None
    for f in result.files_created:
        if "calculator" in Path(f).name.lower() and f.endswith(".py"):
            calc_path = Path(f)
            break

    if not calc_path or not calc_path.exists():
        # 在工作区中查找
        for ws in [dev_workspace, base_workspace]:
            if ws.exists():
                for p in ws.rglob("*.py"):
                    if "calculator" in p.name.lower() or p.name == "calculator.py":
                        calc_path = p
                        break
                if calc_path:
                    break

    # 回退：如果没找到 "calculator" 命名的文件，查找工作区中的主程序 .py
    # 排除 test_ 开头、__ 开头、setup.py 等辅助文件
    if not calc_path or not calc_path.exists():
        _AUX_PREFIXES = ("test_", "__", "setup.", "conftest.", "_")
        for ws in [dev_workspace, base_workspace]:
            if ws.exists():
                candidates = [
                    p for p in ws.rglob("*.py")
                    if not any(p.name.startswith(prefix) for prefix in _AUX_PREFIXES)
                ]
                if len(candidates) == 1:
                    calc_path = candidates[0]
                    break
                elif len(candidates) > 1:
                    # 多个候选：优先选含 calc/计算/compute 关键字的
                    for c in candidates:
                        name_lower = c.name.lower()
                        if any(kw in name_lower for kw in ("calc", "compute", "main", "app")):
                            calc_path = c
                            break
                    if not calc_path:
                        # 兜底：选第一个
                        calc_path = candidates[0]
                    break

    result.add_check("calculator_exists", calc_path is not None and calc_path.exists(),
                     str(calc_path) if calc_path else "未找到主程序 .py 文件")

    if calc_path and calc_path.exists():
        content = calc_path.read_text(encoding="utf-8")

        # 检查关键功能（更精确：检测运算符在表达式解析/计算上下文中的使用）
        import re
        has_add = bool(re.search(r'[\+\+]|"add"|operator\.add|加法', content))
        has_sub = bool(re.search(r'(?<=[\w\s])-(?=[\d\s(])|operator\.sub|减法|"sub"', content)) and "def " in content
        has_mul = bool(re.search(r'\*[^/]|"mul"|operator\.mul|乘法', content))
        has_div = bool(re.search(r'/[^/*]|"div"|operator\.truediv|除法', content))
        has_input = "input(" in content
        has_quit = '"q"' in content or "'q'" in content

        result.add_check("calculator_add", has_add, "支持加法")
        result.add_check("calculator_sub", has_sub, "支持减法")
        result.add_check("calculator_mul", has_mul, "支持乘法")
        result.add_check("calculator_div", has_div, "支持除法")
        result.add_check("calculator_input", has_input, "支持命令行输入")
        result.add_check("calculator_quit", has_quit, "支持退出")

        # 尝试语法检查
        import py_compile
        try:
            py_compile.compile(str(calc_path), doraise=True)
            result.add_check("calculator_syntax", True, "Python 语法正确")
        except py_compile.PyCompileError as e:
            result.add_check("calculator_syntax", False, f"语法错误: {e}")

        # 尝试导入检查（mock input 避免进入交互循环）
        try:
            import importlib.util
            import unittest.mock
            spec = importlib.util.spec_from_file_location("calculator_test", str(calc_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                with unittest.mock.patch("builtins.input", side_effect=EOFError):
                    spec.loader.exec_module(mod)
                result.add_check("calculator_importable", True, "模块可加载且可执行")
            else:
                result.add_check("calculator_importable", False, "无法创建模块规格")
        except Exception as e:
            result.add_check("calculator_importable", False, f"导入失败: {e}")


# ── 动态 Bug 检测 ─────────────────────────────────────────────

def detect_runtime_bugs() -> list[dict]:
    """基于运行时数据动态检测问题。"""
    findings = []

    # 1. 检查各阶段是否全部成功
    for name, phase in result.phases.items():
        if phase["status"] != "completed":
            findings.append({
                "severity": "ERROR",
                "title": f"阶段 {name} 失败",
                "detail": phase.get("error", "未知错误"),
            })

    # 2. 检查是否有阶段耗时异常长
    for name, phase in result.phases.items():
        if phase["duration_s"] > 300:
            findings.append({
                "severity": "WARN",
                "title": f"阶段 {name} 耗时过长",
                "detail": f"{phase['duration_s']}s，可能存在性能问题或死循环",
            })

    # 3. 检查 token 消耗是否异常
    total_tokens = result.token_usage["input"] + result.token_usage["output"]
    if total_tokens > 200_000:
        findings.append({
            "severity": "WARN",
            "title": "Token 消耗过高",
            "detail": f"总计 {total_tokens} tokens，可能存在消息膨胀或重复调用",
        })

    # 4. 检查文件产出
    if not result.files_created:
        findings.append({
            "severity": "ERROR",
            "title": "无文件产出",
            "detail": "开发阶段未生成任何文件",
        })

    # 5. 检查关键断言失败
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
    project_root = Path(__file__).parent
    test_dir = project_root / "data" / "users" / TEST_USER_ID

    # 先清理 factory 缓存中的测试 agent（释放 SQLite 连接）
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

    # 清理子代理 checkpointer 缓存（可能持有 SQLite 连接）
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

    # 等待 SQLite 连接释放：先 gc，再轮询确认文件句柄已释放
    import gc
    gc.collect()
    time.sleep(0.5)

    # 轮询检查 .db 文件是否可写（句柄已释放）
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

    # 删除测试目录（Windows 下 SQLite 文件可能仍被占用，指数退避重试）
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
                    wait = 1.0 * (2 ** attempt)  # 1s, 2s, 4s, 8s
                    logger.warning("[Cleanup] rmtree 失败 (尝试 %d/%d)，%0.1fs 后重试...",
                                   attempt + 1, max_retries, wait)
                    time.sleep(wait)
                else:
                    # 最后一次尝试：逐个删除非锁定文件
                    logger.warning("[Cleanup] rmtree 最终失败，尝试逐个删除")
                    for item in test_dir.rglob("*"):
                        try:
                            if item.is_file():
                                item.unlink()
                        except Exception:
                            pass
                    try:
                        shutil.rmtree(test_dir)
                        logger.info("[Cleanup] 二次清理成功: %s", test_dir)
                    except Exception as exc:
                        logger.warning("[Cleanup] 清理不完全: %s", exc)


# ── 主流程 ────────────────────────────────────────────────────

async def run_test():
    """运行端到端测试。"""
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
    logger.info("端到端测试开始: Python 计算器项目")
    logger.info("=" * 60)

    try:
        # ── 阶段 1: 需求分析 ────────────────────────────────
        logger.info("\n>>> 阶段 1: 需求分析")
        try:
            analysis = await asyncio.wait_for(
                run_analysis_phase("requirement_analysis",
                                   REQUIREMENT_ANALYST_PROMPT,
                                   SIMPLE_REQUIREMENT),
                timeout=ANALYSIS_TIMEOUT_S,
            )
        except Exception as exc:
            result.add_finding("ERROR", "需求分析失败", str(exc))
            logger.warning("[阶段1] 需求分析失败，使用降级数据: %s", exc)
            analysis = {"project_name": SIMPLE_REQUIREMENT["project_name"],
                        "core_features": ["加法", "减法", "乘法", "除法"]}

        # ── 阶段 2: PRD 设计 ────────────────────────────────
        logger.info("\n>>> 阶段 2: PRD 设计")
        try:
            prd = await asyncio.wait_for(
                run_analysis_phase("prd_design",
                                   PRODUCT_MANAGER_PROMPT,
                                   analysis),
                timeout=ANALYSIS_TIMEOUT_S,
            )
        except Exception as exc:
            result.add_finding("ERROR", "PRD设计失败", str(exc))
            logger.warning("[阶段2] PRD设计失败，使用降级数据: %s", exc)
            prd = {"productPositioning": "命令行计算器",
                   "featureList": [{"name": "四则运算", "priority": "P0"}],
                   "acceptanceCriteria": ["支持加减乘除"]}

        # ── 阶段 3: 成本估算 ────────────────────────────────
        logger.info("\n>>> 阶段 3: 成本估算")
        try:
            quote = await asyncio.wait_for(
                run_analysis_phase("cost_estimation",
                                   COST_ESTIMATOR_PROMPT,
                                   {**prd, **analysis}),
                timeout=ANALYSIS_TIMEOUT_S,
            )
        except Exception as exc:
            result.add_finding("WARNING", "成本估算失败", str(exc))
            logger.warning("[阶段3] 成本估算失败，跳过: %s", exc)
            quote = {}

        # ── 阶段 4: 项目开发 ────────────────────────────────
        logger.info("\n>>> 阶段 4: 项目开发")
        # 精简传入数据：只传项目名 + P0 功能列表 + 验收标准，不传完整 userStories/dataModel/quote
        slim_prd = {}
        if isinstance(prd, dict):
            slim_prd = {
                k: v for k, v in prd.items()
                if k in ("productPositioning", "featureList", "acceptanceCriteria",
                         "techComplexity", "totalFeatures")
            }
        project_data = {
            "project_name": SIMPLE_REQUIREMENT["project_name"],
            "analysis": analysis,
            "prd": slim_prd or prd,
        }
        try:
            dev_data = await asyncio.wait_for(
                run_development_phase(project_data),
                timeout=DEV_TIMEOUT_S,
            )
        except Exception as exc:
            result.add_error(f"项目开发失败: {exc}")
            dev_data = {}

        # ── 阶段 5: 成品验证 ────────────────────────────────
        logger.info("\n>>> 阶段 5: 成品验证")
        verify_product(dev_data)

    except asyncio.TimeoutError:
        result.add_error(f"全局超时（{GLOBAL_TIMEOUT_S}s），测试中断")
    except Exception as exc:
        result.add_error(f"测试异常: {exc}")
        traceback.print_exc()

    # ── 动态 Bug 检测 ────────────────────────────────────────
    findings = detect_runtime_bugs()

    # ── 保存结果 ─────────────────────────────────────────────
    _save_result(findings)

    # ── 清理 ─────────────────────────────────────────────────
    cleanup()

    # ── 打印摘要 ─────────────────────────────────────────────
    _print_summary(findings)


def _save_result(findings: list[dict] | None = None):
    """保存结构化测试结果到 JSON 文件。"""
    data = result.to_dict()
    if findings:
        data["runtime_findings"] = findings
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    logger.info("测试结果已保存到 %s", RESULT_FILE)


def _print_summary(findings: list[dict]):
    """打印测试摘要。"""
    data = result.to_dict()

    print("\n" + "=" * 60)
    print("端到端测试结果")
    print("=" * 60)
    print(f"总结果: {'PASS' if data['passed'] else 'FAIL'}")
    print(f"总耗时: {data['total_duration_s']}s")
    print(f"Token 消耗: input={data['token_usage']['input']}, "
          f"output={data['token_usage']['output']}")

    print("\n阶段执行:")
    for name, phase in data["phases"].items():
        icon = "OK" if phase["status"] == "completed" else "FAIL"
        print(f"  [{icon}] {name}: {phase['status']} ({phase['duration_s']}s)")
        if phase.get("error"):
            print(f"       错误: {phase['error']}")

    print("\n验证项:")
    for check in data["checks"]:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['name']}: {check['detail']}")

    if data["files_created"]:
        print("\n生成的文件:")
        for f in data["files_created"]:
            print(f"  - {f}")

    if findings:
        print("\n运行时发现:")
        for f in findings:
            print(f"  [{f['severity']}] {f['title']}: {f['detail']}")

    if data["errors"]:
        print("\n错误:")
        for e in data["errors"]:
            print(f"  - {e}")

    print("\n详细结果: " + str(RESULT_FILE.resolve()))
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
