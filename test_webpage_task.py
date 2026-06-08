"""复杂端到端测试 —— 制作「世界咖啡地图」交互式网页，模拟 LangGraph Agent 全流程。

任务：搜集世界各产地咖啡信息，制作一个交互式网页，展示咖啡产地地图、风味轮、冲泡指南。

使用方式：
    python test_webpage_task.py

前置条件：
    - .env 文件中已配置 DEEPSEEK_API_KEY
    - 依赖已安装

全流程模拟：
    1. 需求分析（直接 LLM 调用）
    2. PRD 设计（直接 LLM 调用）
    3. 成本估算（直接 LLM 调用）
    4. 开发阶段（LGAgent 全流程：P3 阶段推进 + interrupt 审批 + 子代理派遣）
    5. 成品验证（文件检查 + HTML 质量 + 交互功能检测）

监控维度：
    1. Plan-then-Execute 规划是否合理
    2. P3 自适应工具选择（gather→modify→verify）是否正常推进
    3. 并行子代理派遣是否生效
    4. interrupt 审批门在无 checkpointer 时的降级行为
    5. 消息压缩 / ContextView 裁剪是否丢失关键信息
    6. 多文件产出是否完整（HTML + CSS + JS）
    7. 最终网页质量（结构、样式、交互元素）
    8. 整体 token 消耗和迭代次数
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
        logging.FileHandler("test_webpage_task.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_webpage")

# ── 常量 ──────────────────────────────────────────────────────

TEST_USER_ID = "test_webpage_user"
TEST_TICKET_ID = "webpage_task_001"
GLOBAL_TIMEOUT_S = 900      # 15 分钟硬上限
ANALYSIS_TIMEOUT_S = 120    # 分析阶段单阶段超时
DEV_TIMEOUT_S = 600         # 开发阶段超时
RESULT_FILE = Path("test_webpage_result.json")

# 复杂任务：制作世界咖啡地图交互式网页
WEBPAGE_REQUIREMENT = {
    "project_name": "世界咖啡地图",
    "project_type": "交互式网页",
    "description": (
        "制作一个「世界咖啡地图」交互式网页，展示全球主要咖啡产区的信息。"
        "要求：\n"
        "1. 至少包含 8 个咖啡产区（埃塞俄比亚、哥伦比亚、巴西、危地马拉、肯尼亚、"
        "印尼苏门答腊、牙买加蓝山、云南）的详细信息\n"
        "2. 每个产区需包含：产地名称、风味描述、海拔范围、主要品种、处理法\n"
        "3. 页面需有交互式风味轮（用 CSS + JS 实现，点击风味可高亮对应产区）\n"
        "4. 包含冲泡指南区域（至少 3 种冲泡方式：手冲、法压、摩卡壶）\n"
        "5. 使用 data-origin 属性标记每个产区卡片，以便程序精确计数\n"
        "6. 深棕色/米色/金色的咖啡主题配色方案\n"
        "7. 响应式布局，支持移动端浏览\n"
        "8. 将所有文件写入工作区：index.html, style.css, app.js\n"
    ),
    "deadline": "",
    "budget": "",
}

# ── 分析阶段 Prompt ──────────────────────────────────────────

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
4. 创建 3 个文件：index.html, style.css, app.js

工作方式：
1. 使用 write_file 工具逐个创建项目文件
2. 先创建 HTML 结构，再创建 CSS 样式，最后创建 JS 交互
3. 每个文件写完后确认内容正确

⚠️ 重要约束：
- 不要使用 run_command 工具运行代码验证
- 不要使用 edit_file 工具（直接用 write_file 覆盖即可）
- 只使用 write_file 工具创建文件

完成所有文件后，输出一个 JSON 摘要，格式如下：
{{"project_name": "项目名", "files_created": ["文件1路径", "文件2路径"], "tech_stack": "技术栈", "setup_instructions": "安装运行步骤"}}

⚠️ 重要：先使用 write_file 工具创建所有文件，最后再输出 JSON 摘要。不要把代码内容放在 JSON 中。"""


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
        logger.info("[ToolEnd] %s", str(output)[:100])

    def on_tool_error(self, error, **kwargs):
        self._result.add_finding("ERROR", "工具执行错误", str(error))


# ── 纯 LLM 调用（分析阶段用） ──────────────────────────────

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
    """直接调用 LLM，不走 Agent 循环。用于三大分析阶段。"""
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


# ── Agent 直接调用 ────────────────────────────────────────────

async def _call_lg_agent(prompt: str, user_id: str, ticket_id: str,
                         max_iterations: int = 40,
                         phase_name: str = "") -> dict:
    """直接调用 LangGraph Agent，返回 {output} 或抛异常。

    使用 ainvoke 获取可靠结果，自动处理 interrupt。
    每次调用使用独立的 thread_id，避免不同阶段的 state 污染。
    """
    from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
    from agent_by_langgraph.lg_agent import ReasoningCollector
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

    # D2-fix: 不再拼接 phase_name 到 ticket_id，确保 Agent 工作区与验证路径一致
    agent = create_lg_agent(
        user_id=user_id,
        ticket_id=ticket_id,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations=max_iterations,
    )

    # 禁用 Checkpointer，避免 SQLite 开销和 Windows 文件锁问题
    agent.graph.checkpointer = None
    has_checkpointer = False

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

    config: RunnableConfig = {
        "callbacks": all_callbacks,
        "recursion_limit": max_iterations * 4 + 10,
        "configurable": {
            "thread_id": f"{user_id}_{phase_name}" if phase_name else user_id,
            "__has_checkpointer__": has_checkpointer,
        },
    }

    # 执行 + interrupt 自动批准循环
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
                logger.warning("[Interrupt] aget_state 失败: %s", exc)

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

    # 从 invoke_result 中提取 P3 阶段信息
    final_phase = invoke_result.get("_phase", "unknown")
    final_stall = invoke_result.get("_stall_count", 0)
    logger.info("[Agent] 最终阶段=%s, 停滞次数=%d", final_phase, final_stall)

    # 统计图节点执行
    for msg in invoke_result.get("messages", []):
        pass  # 节点统计通过 astream_events 更精确，此处仅记录最终状态

    # 清理
    try:
        reset_lg_agent(user_id, ticket_id)
    except Exception:
        pass

    return {"output": reply, "final_phase": final_phase, "stall_count": final_stall}


# ── JSON 提取工具 ─────────────────────────────────────────────

def _extract_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON，支持截断修复。"""
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

    # 快速检测截断
    stripped = candidate.rstrip()
    is_truncated = (
        stripped.endswith('...') or
        stripped.endswith('\\') or
        (stripped.count('"') % 2 == 1) or
        (stripped[-1:] not in '}"\'\\]') and not stripped.endswith('null') and not stripped.endswith('true') and not stripped.endswith('false')
    )
    if is_truncated:
        try:
            return _repair_truncated_json(candidate)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    for strategy in [
        lambda: json.loads(candidate),
        lambda: json.JSONDecoder().raw_decode(candidate)[0],
        lambda: json.loads(re.sub(r',\s*([}\]])', r'\1', candidate)),
    ]:
        try:
            return strategy()
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    try:
        return _repair_truncated_json(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def _repair_truncated_json(text: str) -> dict:
    """修复被截断的 JSON：补全缺失的括号和引号。"""
    text = re.sub(r',\s*$', '', text.strip())

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

    # 补全未闭合的字符串
    if in_string:
        text += '"'

    # 补全未闭合的括号
    for ch in reversed(stack):
        text += '}' if ch == '{' else ']'

    return json.loads(text)


# ── 三大分析阶段 ──────────────────────────────────────────────

async def run_analysis_phase(name: str, system_prompt: str, data: Any) -> dict:
    """执行分析阶段（直接 LLM 调用）。"""
    t0 = time.time()
    last_error = None

    for attempt in range(3):
        try:
            user_prompt = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data)
            raw = await _call_llm_direct(system_prompt, user_prompt)
            parsed = _extract_json(raw)

            if parsed is None:
                raise ValueError(f"JSON 解析失败，原始输出前200字: {raw[:200]}")

            duration = time.time() - t0
            result.record_phase(name, "completed", duration, data=parsed)
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning("[Analysis] %s 第 %d 次尝试失败: %s", name, attempt + 1, exc)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

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
            max_iterations=40, phase_name="development",
        )
        reply = raw["output"]
        final_phase = raw.get("final_phase", "unknown")
        stall_count = raw.get("stall_count", 0)

        # 记录 P3 阶段信息
        result.add_finding("INFO", "P3 最终阶段",
                           f"开发阶段最终处于 {final_phase} 阶段，停滞次数={stall_count}")

        # 检测迭代耗尽
        if "max iterations" in reply.lower() or "agent stopped" in reply.lower():
            raise ValueError("Agent 迭代次数耗尽")

        data = _extract_json(reply)
        if data is None:
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
    """验证开发产出的文件。"""
    project_root = Path(__file__).parent
    # D2-fix: Agent 和验证脚本现在使用同一 ticket_id，无需搜索两个目录
    workspace = project_root / "data" / "users" / TEST_USER_ID / TEST_TICKET_ID

    # 收集所有已创建的文件
    found_files = []
    if workspace.exists():
        for p in workspace.rglob("*"):
            if p.is_file() and p.name != "checkpoints.db" and "checkpoints" not in str(p):
                found_files.append(p)

    # 也扫描成品目录
    output_root = workspace / "成品"
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

    # ── 检查 HTML 文件 ──────────────────────────────────────
    html_path = None
    for f in unique_files:
        if f.suffix == ".html":
            html_path = f
            break

    # 如果没找到 .html，在文件名中搜索
    if not html_path:
        for f in unique_files:
            if "index" in f.name.lower() and f.suffix in (".htm", ".html"):
                html_path = f
                break

    result.add_check("html_exists", html_path is not None and html_path.exists(),
                     str(html_path) if html_path else "未找到 HTML 文件")

    if html_path and html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        html_size = html_path.stat().st_size
        result.add_check("html_size", html_size > 2000,
                         f"HTML 文件大小: {html_size} bytes")

        # 检查 data-origin 属性（产区卡片标记）
        origin_count = len(re.findall(r'data-origin', html_content))
        result.add_check("origin_cards", origin_count >= 8,
                         f"检测到 {origin_count} 个 data-origin 产区卡片（要求 >= 8）")
        if origin_count < 8 and origin_count > 0:
            result.add_finding("WARN", "产区卡片不足",
                               f"仅 {origin_count} 个产区卡片，要求至少 8 个")

        # 检查关键内容
        has_coffee_keywords = any(kw in html_content for kw in ["咖啡", "coffee", "Coffee"])
        result.add_check("coffee_content", has_coffee_keywords,
                         "HTML 包含咖啡相关内容")

        # 检查产区信息
        origin_keywords = ["埃塞俄比亚", "哥伦比亚", "巴西", "危地马拉", "肯尼亚",
                           "印尼", "苏门答腊", "牙买加", "蓝山", "云南"]
        found_origins = [kw for kw in origin_keywords if kw in html_content]
        result.add_check("origin_coverage", len(found_origins) >= 5,
                         f"覆盖 {len(found_origins)}/{len(origin_keywords)} 个产区: {found_origins}")

        # 检查风味描述
        flavor_keywords = ["风味", "flavor", "酸", "苦", "甜", "果香", "花香", "巧克力"]
        has_flavor = any(kw in html_content for kw in flavor_keywords)
        result.add_check("flavor_descriptions", has_flavor,
                         "包含风味描述")

        # 检查冲泡指南
        brew_keywords = ["冲泡", "手冲", "法压", "摩卡壶", "brew", "pour"]
        has_brew = any(kw in html_content for kw in brew_keywords)
        result.add_check("brew_guide", has_brew,
                         "包含冲泡指南")

        # 检查样式
        has_style = "<style" in html_content or 'rel="stylesheet"' in html_content
        result.add_check("has_style", has_style,
                         "包含样式定义")

        # 检查交互元素
        has_js = "<script" in html_content or 'onclick' in html_content
        result.add_check("has_interaction", has_js,
                         "包含交互元素")

        # 检查响应式布局
        has_responsive = "viewport" in html_content or "@media" in html_content or "responsive" in html_content.lower()
        result.add_check("responsive", has_responsive,
                         "包含响应式布局")

        # 检查配色方案（咖啡主题）
        coffee_colors = ["#3E2723", "#5D4037", "#795548", "#8D6E63", "#D7CCC8",
                         "#BCAAA4", "#A1887F", "#4E342E", "#6D4C41", "brown",
                         "coffee", "tan", "beige", "gold"]
        has_coffee_colors = any(c.lower() in html_content.lower() for c in coffee_colors)
        result.add_check("coffee_theme", has_coffee_colors,
                         "使用咖啡主题配色")

    # ── 检查 CSS 文件 ──────────────────────────────────────
    css_path = None
    for f in unique_files:
        if f.suffix == ".css":
            css_path = f
            break

    result.add_check("css_exists", css_path is not None and css_path.exists(),
                     str(css_path) if css_path else "未找到 CSS 文件")

    if css_path and css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        css_size = css_path.stat().st_size
        result.add_check("css_size", css_size > 500,
                         f"CSS 文件大小: {css_size} bytes")

        # 检查响应式媒体查询
        has_media_query = "@media" in css_content
        result.add_check("css_responsive", has_media_query,
                         "CSS 包含响应式媒体查询")

    # ── 检查 JS 文件 ──────────────────────────────────────
    js_path = None
    for f in unique_files:
        if f.suffix == ".js":
            js_path = f
            break

    result.add_check("js_exists", js_path is not None and js_path.exists(),
                     str(js_path) if js_path else "未找到 JS 文件")

    if js_path and js_path.exists():
        js_content = js_path.read_text(encoding="utf-8")
        js_size = js_path.stat().st_size
        result.add_check("js_size", js_size > 200,
                         f"JS 文件大小: {js_size} bytes")

        # 检查交互功能
        has_event_listener = "addEventListener" in js_content or "onclick" in js_content
        result.add_check("js_interaction", has_event_listener,
                         "JS 包含事件监听/交互逻辑")

    # ── 综合质量评估 ──────────────────────────────────────
    file_types = set(f.suffix for f in unique_files)
    result.add_check("multi_file", len(file_types) >= 2,
                     f"产出 {len(file_types)} 种文件类型: {file_types}")


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

    # 3. 检查 token 消耗
    total_tokens = result.token_usage["input"] + result.token_usage["output"]
    if total_tokens > 300_000:
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

    # 6. P3 阶段推进检查
    if not result.phase_transitions:
        findings.append({
            "severity": "WARN",
            "title": "未检测到 P3 阶段转换",
            "detail": "P3 自适应工具选择可能未正常工作，或 observer 未捕获到 _phase 变化",
        })

    # 7. 工具调用统计
    write_file_calls = sum(1 for tc in result.tool_calls if tc["tool"] == "write_file")
    if write_file_calls == 0:
        findings.append({
            "severity": "ERROR",
            "title": "未调用 write_file",
            "detail": "开发阶段未调用 write_file 工具，文件无法产出",
        })
    elif write_file_calls > 20:
        findings.append({
            "severity": "WARN",
            "title": "write_file 调用过多",
            "detail": f"调用了 {write_file_calls} 次 write_file，可能存在重复写入或覆盖",
        })

    # 8. LLM 调用次数
    if result.llm_calls > 50:
        findings.append({
            "severity": "WARN",
            "title": "LLM 调用次数过多",
            "detail": f"共 {result.llm_calls} 次 LLM 调用，可能存在循环或重试过多",
        })

    return findings


# ── 静态代码审查 ─────────────────────────────────────────────

def analyze_code_findings() -> list[dict]:
    """对 agent 系统核心代码进行静态审查，识别潜在 bug 和缺陷。"""
    findings = []

    # 1. lg_graph.py: _route_after_agent 混合调用路由
    findings.append({
        "severity": "WARN",
        "title": "混合调用路由可能丢失普通工具调用",
        "location": "agent_by_langgraph/lg_graph.py::_route_after_agent",
        "description": (
            "当 LLM 同时发出 dispatch_subagent_lg + 普通工具调用时，"
            "路由到 subagent_dispatcher，非子代理 tool_calls 暂存到 _pending_tool_calls。"
            "但如果 _advance_phase 未正确暂存（如 AIMessage 无 tool_calls），"
            "这些调用可能丢失。"
        ),
        "suggestion": "在 _aggregate_results 中增加防御性检查，确保 pending_tool_calls 不为空时才路由到 tools",
    })

    # 2. lg_graph.py: _advance_phase 阶段推进逻辑
    findings.append({
        "severity": "INFO",
        "title": "P3 gather→modify 推进依赖写工具调用检测",
        "location": "agent_by_langgraph/lg_graph.py::_advance_phase",
        "description": (
            "gather→modify 的推进条件是 agent 调用了写工具（write_file/edit_file/run_command）。"
            "但如果 LLM 在 gather 阶段就想写文件（如创建临时文件辅助分析），"
            "会被工具过滤阻止，导致 stall_count 增加。"
            "连续 2 次停滞后回退到 'all' 阶段，绕过了 P3 限制。"
        ),
        "suggestion": "考虑在 gather 阶段允许 run_command（只读命令如 ls、cat），减少误判停滞",
    })

    # 3. lg_agent.py: Checkpointer 延迟初始化
    findings.append({
        "severity": "WARN",
        "title": "Checkpointer 延迟初始化可能导致首次调用失败",
        "location": "agent_by_langgraph/lg_agent.py::_init_checkpointer",
        "description": (
            "Checkpointer 延迟到首次 ainvoke 时初始化（_ensure_checkpointer），"
            "但 _init_checkpointer 返回 None，导致 graph.checkpointer 为 None。"
            "如果调用方在 _ensure_checkpointer 之前检查 checkpointer 状态，"
            "会误判为无 checkpointer。"
        ),
        "suggestion": "在 _ensure_checkpointer 中增加状态标记，或在 __init__ 中同步初始化",
    })

    # 4. lg_graph.py: interrupt 无 checkpointer 降级
    findings.append({
        "severity": "WARN",
        "title": "无 checkpointer 时 interrupt 审批门自动放行",
        "location": "agent_by_langgraph/lg_graph.py::_interrupt_approval",
        "description": (
            "interrupt() 需要 checkpointer 才能暂停和恢复执行。"
            "无 checkpointer 时自动放行所有危险工具调用，"
            "这意味着测试环境中的安全审批门形同虚设。"
        ),
        "suggestion": "在无 checkpointer 时至少记录日志或提供配置选项控制是否自动放行",
    })

    # 5. lc_tools.py: _resolve 路径解析复杂度
    findings.append({
        "severity": "INFO",
        "title": "_resolve 路径解析逻辑复杂，LLM 路径格式不可预测",
        "location": "agent/lc_tools.py::_resolve",
        "description": (
            "_resolve 函数处理多种路径格式（相对、绝对、带前缀），"
            "但 LLM 生成的路径格式不可预测，可能导致路径嵌套问题。"
            "如 LLM 写入 'data/users/uid/tid/index.html' 会被剥离前缀，"
            "但 '成品/index.html' 可能不会被正确处理。"
        ),
        "suggestion": "增加路径解析的单元测试，覆盖 LLM 常见路径格式",
    })

    # 6. lg_subagent.py: 子代理无 interrupt 审批
    findings.append({
        "severity": "WARN",
        "title": "子代理子图无 interrupt 审批门",
        "location": "agent_by_langgraph/lg_subagent.py::create_subagent_graph",
        "description": (
            "子代理子图直接执行工具调用，无 interrupt_approval 节点。"
            "如果子代理调用 write_file 等危险工具，不会经过审批。"
            "虽然子代理的工具白名单通常不包含危险工具，"
            "但 spec 配置错误时可能绕过安全检查。"
        ),
        "suggestion": "在子代理子图中也加入 interrupt 审批门，或验证 spec 的工具白名单",
    })

    # 7. lg_graph.py: subagent_results 清空机制
    findings.append({
        "severity": "INFO",
        "title": "subagent_results 使用哨兵值 '__CLEAR__' 清空",
        "location": "agent_by_langgraph/lg_graph.py::_replace_results",
        "description": (
            "使用 '__CLEAR__' 哨兵值替代空列表清空语义，"
            "避免意外清空。但如果子代理输出恰好包含 '__CLEAR__' 字符串，"
            "会触发误清空。虽然概率极低，但值得注意。"
        ),
        "suggestion": "使用更独特的哨兵值（如 UUID）或改用 state 标记位",
    })

    # 8. factory.py: Agent 缓存无 TTL
    findings.append({
        "severity": "INFO",
        "title": "Agent 缓存无 TTL 过期机制",
        "location": "agent_by_langgraph/factory.py::_agent_cache",
        "description": (
            "Agent 实例缓存只有 LRU 淘汰，无 TTL 过期。"
            "长时间不活跃的 Agent 实例仍占用内存和 SQLite 连接。"
        ),
        "suggestion": "增加 TTL 过期机制，定期清理不活跃的 Agent 实例",
    })

    return findings


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

    # 等待 SQLite 连接释放
    import gc
    gc.collect()
    time.sleep(0.5)

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
                    logger.warning("[Cleanup] rmtree 失败 (尝试 %d/%d)，%0.1fs 后重试...",
                                   attempt + 1, max_retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("[Cleanup] 清理不完全，请手动删除: %s", test_dir)


# ── 主流程 ────────────────────────────────────────────────────

async def run_test():
    """运行端到端测试。"""
    global result
    result = TestResult()
    from dotenv import load_dotenv
    load_dotenv()

    # 测试环境允许无 checkpointer 时自动放行危险工具
    os.environ.setdefault("AUTO_APPROVE_WITHOUT_CHECKPOINTER", "true")

    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or \
       os.environ["DEEPSEEK_API_KEY"] == "your_api_key_here":
        result.add_error("DEEPSEEK_API_KEY 未配置")
        _save_result()
        return

    logger.info("=" * 60)
    logger.info("复杂端到端测试开始: 世界咖啡地图交互式网页")
    logger.info("=" * 60)

    try:
        # ── 阶段 1: 需求分析 ────────────────────────────────
        logger.info("\n>>> 阶段 1: 需求分析")
        try:
            analysis = await asyncio.wait_for(
                run_analysis_phase("requirement_analysis",
                                   REQUIREMENT_ANALYST_PROMPT,
                                   WEBPAGE_REQUIREMENT),
                timeout=ANALYSIS_TIMEOUT_S,
            )
        except Exception as exc:
            result.add_finding("ERROR", "需求分析失败", str(exc))
            logger.warning("[阶段1] 需求分析失败，使用降级数据: %s", exc)
            analysis = {
                "project_name": WEBPAGE_REQUIREMENT["project_name"],
                "core_features": ["咖啡产区展示", "风味轮", "冲泡指南"],
            }

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
            prd = {
                "productPositioning": "世界咖啡地图交互式网页",
                "featureList": [{"name": "产区展示", "priority": "P0"}],
                "acceptanceCriteria": ["8个产区卡片", "风味轮", "冲泡指南"],
            }

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
        slim_prd = {}
        if isinstance(prd, dict):
            slim_prd = {
                k: v for k, v in prd.items()
                if k in ("productPositioning", "featureList", "acceptanceCriteria",
                         "techComplexity", "totalFeatures")
            }
        project_data = {
            "project_name": WEBPAGE_REQUIREMENT["project_name"],
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

    # ── 静态代码审查 ────────────────────────────────────────
    code_findings = analyze_code_findings()

    # ── 保存结果 ─────────────────────────────────────────────
    _save_result(findings, code_findings)

    # ── 清理 ─────────────────────────────────────────────────
    cleanup()

    # ── 打印摘要 ─────────────────────────────────────────────
    _print_summary(findings, code_findings)


def _save_result(findings: list[dict] | None = None, code_findings: list[dict] | None = None):
    """保存结构化测试结果到 JSON 文件。"""
    data = result.to_dict()
    if findings:
        data["runtime_findings"] = findings
    if code_findings:
        data["code_findings"] = code_findings
    RESULT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    logger.info("测试结果已保存到 %s", RESULT_FILE)


def _print_summary(findings: list[dict], code_findings: list[dict]):
    """打印测试摘要。"""
    data = result.to_dict()

    print("\n" + "=" * 60)
    print("复杂端到端测试结果 — 世界咖啡地图")
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

    # 运行时发现
    if findings:
        print("\n" + "=" * 60)
        print("运行时发现:")
        print("=" * 60)
        for f in findings:
            print(f"  [{f['severity']}] {f['title']}")
            if f.get("detail"):
                print(f"       {f['detail']}")

    # 代码审查发现
    if code_findings:
        print("\n" + "=" * 60)
        print("代码审查发现:")
        print("=" * 60)
        for f in code_findings:
            print(f"\n  [{f['severity']}] {f['title']}")
            print(f"    位置: {f['location']}")
            print(f"    描述: {f['description']}")
            if f.get('suggestion'):
                print(f"    建议: {f['suggestion']}")

    print("\n" + "=" * 60)
    print(f"详细结果已保存到: {RESULT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())
