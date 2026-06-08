"""模拟生产场景测试脚本 —— 让 LangGraph Agent 完成修仙境界创作任务。

任务：搜集修仙境界设定，自创100个境界，制作大型静态页面。

使用方式：
    python test_cultivation_task.py

前置条件：
    - .env 文件中已配置 DEEPSEEK_API_KEY
    - 依赖已安装（pip install -r requirements.txt）

观察维度：
    1. Plan-then-Execute 规划是否合理
    2. P3 自适应工具选择（gather→modify→verify）是否正常推进
    3. 并行子代理派遣是否生效
    4. interrupt 审批门在无 checkpointer 时的降级行为
    5. 消息压缩 / ContextView 裁剪是否丢失关键信息
    6. 10个境界的生成是否完整（非低阶/高阶/巅峰后缀糊弄）
    7. 最终静态页面是否成功写入
    8. 整体 token 消耗和迭代次数
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import traceback
from pathlib import Path

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
        logging.FileHandler("test_cultivation_task.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("test_cultivation")


# ── 自定义回调：记录全流程 ──────────────────────────────────────

class FlowObserver:
    """流程观察器：记录每个节点的执行情况。"""

    def __init__(self):
        self.events: list[dict] = []
        self.start_time = time.time()
        self.node_counts: dict[str, int] = {}
        self.tool_calls: list[dict] = []
        self.phase_transitions: list[dict] = []
        self.errors: list[str] = []
        self.llm_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def record_event(self, event_type: str, detail: str, data: dict = None):
        elapsed = time.time() - self.start_time
        self.events.append({
            "time": elapsed,
            "type": event_type,
            "detail": detail,
            "data": data or {},
        })
        logger.info("[%.1fs] %s: %s", elapsed, event_type, detail)

    def record_node(self, node_name: str):
        self.node_counts[node_name] = self.node_counts.get(node_name, 0) + 1
        self.record_event("NODE", node_name)

    def record_tool_call(self, tool_name: str, args_summary: str):
        self.tool_calls.append({"tool": tool_name, "args": args_summary})
        self.record_event("TOOL", f"{tool_name}({args_summary[:80]})")

    def record_phase(self, old_phase: str, new_phase: str):
        self.phase_transitions.append({"from": old_phase, "to": new_phase})
        self.record_event("PHASE", f"{old_phase} → {new_phase}")

    def record_error(self, error: str):
        self.errors.append(error)
        self.record_event("ERROR", error)

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        lines = [
            "=" * 60,
            "流程观察报告",
            "=" * 60,
            f"总耗时: {elapsed:.1f}s",
            f"LLM 调用次数: {self.llm_calls}",
            f"Token 消耗: input={self.total_tokens_in}, output={self.total_tokens_out}",
            f"节点执行次数: {self.node_counts}",
            f"工具调用次数: {len(self.tool_calls)}",
            f"工具调用详情:",
        ]
        for tc in self.tool_calls:
            lines.append(f"  - {tc['tool']}: {tc['args'][:100]}")
        lines.append(f"阶段转换: {self.phase_transitions}")
        if self.errors:
            lines.append(f"错误({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


observer = FlowObserver()


# ── LangChain 回调：拦截 LLM 调用 ──────────────────────────────

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class TestCallbackHandler(BaseCallbackHandler):
    """测试回调：记录 LLM 调用、token 用量、工具调用。"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        observer.llm_calls += 1
        observer.record_event("LLM_START", f"call #{observer.llm_calls}")

    def on_llm_end(self, response, **kwargs):
        # 统计 token
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
        if not usage:
            try:
                usage = response.generations[0][0].message.usage_metadata
            except Exception:
                usage = {}
        if usage:
            if hasattr(usage, "input_tokens"):
                observer.total_tokens_in += usage.input_tokens or 0
                observer.total_tokens_out += usage.output_tokens or 0
            else:
                observer.total_tokens_in += usage.get("input_tokens", 0)
                observer.total_tokens_out += usage.get("output_tokens", 0)

        # 记录工具调用
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    if isinstance(gen.message, AIMessage) and gen.message.tool_calls:
                        for tc in gen.message.tool_calls:
                            args_str = str(tc.get("args", ""))[:200]
                            observer.record_tool_call(tc["name"], args_str)
        except Exception:
            pass

    def on_llm_error(self, error, **kwargs):
        observer.record_error(f"LLM error: {error}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        observer.record_event("TOOL_START", f"{tool_name}")

    def on_tool_end(self, output, **kwargs):
        observer.record_event("TOOL_END", str(output)[:100])

    def on_tool_error(self, error, **kwargs):
        observer.record_error(f"Tool error: {error}")


# ── 主测试流程 ──────────────────────────────────────────────────

TASK_PROMPT = """\
搜集尽可能多的修仙境界设定（参考凡人修仙传、遮天、完美世界、一念永恒、仙逆等经典修仙小说），然后自创出100个境界。

要求：
1. 每个境界必须是一个独立、完整的名称，不能用"低阶"、"高阶"、"巅峰"、"大圆满"这种后缀来糊弄就当一个境界了
2. 境界名称要有修仙气质，如：凝气、筑基、结丹、元婴、化神、炼虚、合体、大乘、渡劫、飞升 等
3. 每个境界需要有简短设定描述（如修炼特征、寿元变化、战力描述等）
4. 将这100个境界制作成一个极其具有修仙气质的大型静态HTML页面，展示这些自创境界以及相应设定
5. HTML页面要有仙侠风格（深色背景、金色/青色文字、云纹装饰等），视觉效果要震撼
6. 每个境界条目必须使用 data-realm 属性标记（如 <div data-realm>或<h2 data-realm>），以便程序精确计数
7. 将HTML文件写入 index.html
"""


async def run_test():
    """运行测试：创建 LangGraph Agent 并执行修仙境界任务。"""
    from dotenv import load_dotenv
    load_dotenv()

    # 清理残留数据，确保干净启动
    test_user_id = "test_cultivation_user"
    test_ticket_id = "cultivation_task_001"
    data_dir = Path(__file__).parent / "data" / "users" / test_user_id
    if data_dir.exists():
        import shutil
        logger.info(f"清理残留数据: {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)

    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY") or os.environ["DEEPSEEK_API_KEY"] == "your_api_key_here":
        logger.error("DEEPSEEK_API_KEY 未配置！请在 .env 文件中设置。")
        logger.error("参考 .env.example 创建 .env 文件。")
        sys.exit(1)

    observer.record_event("TEST_START", "修仙境界创作任务")

    # 创建 Agent（使用工厂模式，模拟生产环境）
    from agent_by_langgraph.factory import create_lg_agent

    observer.record_event("AGENT_CREATE", f"user_id={test_user_id}, ticket_id={test_ticket_id}")

    try:
        agent = create_lg_agent(
            user_id=test_user_id,
            ticket_id=test_ticket_id,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            max_iterations=80,  # 100个境界任务较复杂，提高上限
        )
    except Exception as exc:
        observer.record_error(f"Agent 创建失败: {exc}")
        traceback.print_exc()
        print(observer.summary())
        return

    observer.record_event("AGENT_CREATED", "Agent 创建成功")

    # 构造输入
    from langchain_core.messages import HumanMessage, SystemMessage

    has_checkpointer = agent.will_have_checkpointer
    is_first_turn = agent._first_turn

    if is_first_turn or not has_checkpointer:
        initial_messages = [SystemMessage(content=agent._system_prompt)]
        initial_messages.extend(agent.memory_store.messages)
        user_msg = HumanMessage(content=TASK_PROMPT)
        user_msg.metadata = {"milestone": True}
        initial_messages.append(user_msg)
        input_state = {"messages": initial_messages}
        if has_checkpointer:
            agent._first_turn = False
    else:
        user_msg = HumanMessage(content=TASK_PROMPT)
        user_msg.metadata = {"milestone": True}
        input_state = {"messages": [user_msg]}

    # 配置
    from langchain_core.runnables import RunnableConfig
    test_callback = TestCallbackHandler()
    all_callbacks = list(getattr(agent.graph, '_lg_llm_callbacks', []))
    all_callbacks.append(test_callback)

    # 延迟初始化 checkpointer（确保在正确的事件循环中）
    await agent._ensure_checkpointer()
    has_checkpointer = agent.checkpointer_ready

    config: RunnableConfig = {
        "callbacks": all_callbacks,
        "recursion_limit": 80 * 5 + 10,
        "configurable": {
            "thread_id": test_user_id,
            "__has_checkpointer__": has_checkpointer,
        },
    }

    observer.record_event("INVOKE_START", f"checkpointer={has_checkpointer}, first_turn={is_first_turn}")

    # 执行（使用 astream_events 实现节点级观察 + interrupt 自动批准）
    from langgraph.types import Command

    result = None
    max_retries = 20  # 最多处理20次 interrupt
    for retry in range(max_retries):
        try:
            # 使用 astream_events 获取节点级事件流
            async for event in agent.graph.astream_events(
                input_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                # 记录图节点执行
                if kind == "on_chain_start" and name in {
                    "agent", "advance_phase", "tools", "interrupt_approval",
                    "aggregate_results", "subagent_dispatcher", "subagent_worker",
                    "planner", "route_after_agent", "route_after_aggregate",
                    "route_after_approval",
                }:
                    observer.record_node(name)

                # 记录阶段转换（从 advance_phase 输出中捕获）
                if kind == "on_chain_end" and name == "advance_phase":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        new_phase = output.get("_phase")
                        if new_phase:
                            # 推断旧阶段（从上一次记录的阶段）
                            last_phase = observer.phase_transitions[-1]["to"] if observer.phase_transitions else "gather"
                            if new_phase != last_phase:
                                observer.record_phase(last_phase, new_phase)

            # astream_events 完成后获取最终 state
            # astream_events 不直接返回最终 state，需要通过 get_state 获取
            if has_checkpointer:
                snapshot = await agent.graph.aget_state(config)
                result = {"messages": snapshot.values.get("messages", [])}
            else:
                # 无 checkpointer 时，astream_events 的最后一个 on_chain_end 包含最终输出
                # 此处用 ainvoke 作为降级
                result = await agent.graph.ainvoke(input_state, config=config)

        except Exception as exc:
            observer.record_error(f"Agent 执行失败: {exc}")
            traceback.print_exc()
            print(observer.summary())
            try:
                agent.close()
            except Exception:
                pass
            return

        # 检查是否有 pending interrupt（需要人工审批）
        has_interrupt = False
        if has_checkpointer:
            try:
                snapshot = await agent.graph.aget_state(config)
                if snapshot.next:  # 有待执行的节点 → 图被 interrupt 暂停
                    has_interrupt = True
                    interrupt_info = getattr(snapshot, "tasks", [])
                    observer.record_event("INTERRUPT", f"检测到 interrupt，自动批准: {interrupt_info}")
                    input_state = Command(resume="approve")
                    continue
            except Exception as exc:
                observer.record_event("INTERRUPT_CHECK", f"get_state 失败: {exc}")

        if not has_interrupt:
            break
    else:
        observer.record_error("interrupt 重试次数耗尽")

    observer.record_event("INVOKE_END", "Agent 执行完成")

    # ── 分析结果 ──────────────────────────────────────────────

    messages = result.get("messages", [])
    observer.record_event("RESULT", f"消息总数: {len(messages)}")

    # 提取最终回复
    final_reply = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            final_reply = content if isinstance(content, str) else str(content)
            break

    # 检查文件是否生成
    workspace = agent.root / "data" / "users" / test_user_id / test_ticket_id
    html_path = workspace / "index.html"
    html_exists = html_path.exists()
    observer.record_event("FILE_CHECK", f"index.html exists={html_exists}, path={html_path}")

    # 验证境界数量
    realm_count = 0
    realm_issues: list[str] = []
    html_content = ""
    if html_exists:
        html_content = html_path.read_text(encoding="utf-8")
        import re
        # 优先使用 data-realm 属性精确计数（TASK_PROMPT 中已要求标记）
        data_realm_matches = re.findall(r'data-realm', html_content)
        if data_realm_matches:
            realm_count = len(data_realm_matches)
            observer.record_event("REALM_COUNT", f"精确检测到 {realm_count} 个 data-realm 境界条目")
        else:
            # 降级：统计 HTML 中的境界条目（粗略估计）
            h_tags = re.findall(r'<h[23][^>]*>(.*?)</h[23]>', html_content, re.DOTALL)
            li_tags = re.findall(r'<li[^>]*>(.*?)</li>', html_content, re.DOTALL)
            # 检查是否有低阶/高阶/巅峰后缀糊弄
            forbidden_suffixes = ["低阶", "高阶", "巅峰", "大圆满", "极境", "半步"]
            for tag_content in h_tags + li_tags:
                text = re.sub(r'<[^>]+>', '', tag_content).strip()
                if text:
                    realm_count += 1
                    for suffix in forbidden_suffixes:
                        if suffix in text and len(text) < 10:
                            realm_issues.append(f"疑似糊弄后缀: {text}")

            observer.record_event("REALM_COUNT", f"粗略检测到约 {realm_count} 个境界条目")
            if realm_issues:
                observer.record_event("REALM_ISSUES", f"发现 {len(realm_issues)} 个疑似糊弄: {realm_issues[:5]}")

    # 检查 HTML 质量
    html_quality_issues: list[str] = []
    if html_exists and html_content:
        html_size = html_path.stat().st_size
        observer.record_event("HTML_SIZE", f"{html_size} bytes")
        if html_size < 5000:
            html_quality_issues.append("HTML 文件过小，可能内容不完整")
        if "仙" not in html_content and "修" not in html_content:
            html_quality_issues.append("HTML 中未找到修仙相关文字")
        if "<style" not in html_content:
            html_quality_issues.append("HTML 缺少样式，可能不够震撼")
        if realm_count < 50:
            html_quality_issues.append(f"境界数量不足50个（仅{realm_count}个）")
        if realm_count < 100:
            html_quality_issues.append(f"境界数量不足100个（仅{realm_count}个），任务未完全完成")

    # ── 输出报告 ──────────────────────────────────────────────

    print("\n")
    print(observer.summary())

    print("\n" + "=" * 60)
    print("最终回复（前2000字）:")
    print("=" * 60)
    print(final_reply[:2000])

    if html_quality_issues:
        print("\n" + "=" * 60)
        print("HTML 质量问题:")
        print("=" * 60)
        for issue in html_quality_issues:
            print(f"  - {issue}")

    # ── 阶段转换验证 ──────────────────────────────────────────

    print("\n" + "=" * 60)
    print("P3 阶段转换验证:")
    print("=" * 60)

    phase_transitions = observer.phase_transitions
    if phase_transitions:
        for pt in phase_transitions:
            print(f"  {pt['from']} → {pt['to']}")

        # 验证是否经历了完整的 gather→modify→verify 链路
        phases_seen = set()
        for pt in phase_transitions:
            phases_seen.add(pt["from"])
            phases_seen.add(pt["to"])

        if "gather" in phases_seen:
            print("  [OK] 经历了 gather 阶段（信息收集）")
        else:
            print("  [WARN] 未经历 gather 阶段，可能跳过了信息收集")

        if "modify" in phases_seen:
            print("  [OK] 经历了 modify 阶段（代码修改）")
        else:
            print("  [WARN] 未经历 modify 阶段，可能未执行修改操作")

        if "verify" in phases_seen:
            print("  [OK] 经历了 verify 阶段（验证总结）")
        else:
            print("  [INFO] 未经历 verify 阶段（复杂任务可能需要，简单任务可跳过）")

        # 检查异常回退
        rollback_count = sum(
            1 for pt in phase_transitions
            if pt["to"] == "all" and pt["from"] != "all"
        )
        if rollback_count > 0:
            print(f"  [WARN] 发生 {rollback_count} 次回退到 'all' 阶段（P3 停滞兜底）")
    else:
        print("  [INFO] 未记录到阶段转换（可能 observer 未捕获到 _phase 变化）")

    # ── Bug/缺陷/优化分析 ─────────────────────────────────────

    print("\n" + "=" * 60)
    print("代码审查发现:")
    print("=" * 60)

    findings = analyze_code_findings()
    for finding in findings:
        print(f"\n  [{finding['severity']}] {finding['title']}")
        print(f"    位置: {finding['location']}")
        print(f"    描述: {finding['description']}")
        if finding.get('suggestion'):
            print(f"    建议: {finding['suggestion']}")

    # 清理
    try:
        agent.close()
    except Exception:
        pass

    # 保存完整日志
    log_path = Path("test_cultivation_result.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(observer.summary())
        f.write("\n\n最终回复:\n")
        f.write(final_reply)
        if html_exists:
            f.write(f"\n\nHTML 文件大小: {html_path.stat().st_size} bytes")
            f.write(f"\n检测到境界条目: 约 {realm_count} 个")
    logger.info("完整日志已保存到 %s", log_path)


def analyze_code_findings() -> list[dict]:
    """基于代码审查，返回发现的 bug/缺陷/优化点。"""
    findings = []

    # 1. _should_plan 路由逻辑问题
    findings.append({
        "severity": "BUG",
        "title": "_should_plan 在 aggregate_results 后误判为新请求",
        "location": "lg_graph.py:_should_plan()",
        "description": (
            "_should_plan 检查最后一条消息是否为 HumanMessage 来判断是否为新请求。"
            "已修复：现在优先检查 metadata.milestone 标记，无 milestone 的 HumanMessage "
            "在 plan 非空时按已有计划执行，避免循环中误触发重新规划。"
        ),
        "suggestion": "[已修复] 使用 milestone 标记精确识别新用户请求。",
    })

    # 2. 混合调用时非子代理 tool_calls 丢失
    findings.append({
        "severity": "BUG",
        "title": "混合调用（子代理+普通工具）时非子代理 tool_calls 可能丢失",
        "location": "lg_graph.py:_route_after_agent() + _aggregate_results()",
        "description": (
            "当 LLM 同时发出 dispatch_subagent_lg 和普通/危险工具调用时，"
            "_route_after_agent 优先路由到 subagent_dispatcher。"
            "已修复：_route_after_agent 现在将非子代理 tool_calls 暂存到 "
            "_pending_tool_calls_store（模块级变量），_aggregate_results 从中精确读取，"
            "而非反向遍历 state['messages']（可能匹配到错误的 AIMessage）。"
        ),
        "suggestion": "[已修复] 使用 _pending_tool_calls_store 暂存，_aggregate_results 精确恢复。",
    })

    # 3. _advance_phase 在 interrupt 拒绝后的行为
    findings.append({
        "severity": "DEFECT",
        "title": "_advance_phase 在 interrupt 拒绝后仍可能推进阶段",
        "location": "lg_graph.py:_advance_phase()",
        "description": (
            "当危险工具被审批拒绝后，_interrupt_approval 将 phase 回退到 gather "
            "并重置 _stall_count=0。下一轮 agent 经过 _advance_phase 时，"
            "如果 agent 返回文本回复（无 tool_calls），stall_count 从 0 开始增加。"
            "行为已合理：拒绝后 stall_count 重置，不会因拒绝直接导致回退到 'all'。"
        ),
        "suggestion": "[行为已合理] _interrupt_approval 拒绝时已重置 _stall_count=0，无需额外修改。",
    })

    # 4. 子代理结果压缩可能丢失关键信息
    findings.append({
        "severity": "DEFECT",
        "title": "子代理结果压缩截断可能丢失关键文件路径或代码",
        "location": "lg_graph.py:_compress_subagent_result()",
        "description": (
            "已修复：_MAX_SUBAGENT_CHARS 从 1500 提升到 2000，"
            "_MAX_CONCLUSION_CHARS 从 800 提升到 1200，"
            "减少关键信息被截断的风险。"
        ),
        "suggestion": "[已修复] 压缩上限已调大。",
    })

    # 5. P3 gather 阶段阻止 dispatch_subagent_lg 但工具列表包含它
    findings.append({
        "severity": "DEFECT",
        "title": "P3 gather 阶段包含 dispatch_subagent_lg 但 gather 语义上不应派遣子代理修改",
        "location": "lg_graph.py:_GATHER_TOOLS",
        "description": (
            "_GATHER_TOOLS 包含 dispatch_subagent_lg，意味着 gather 阶段可以派遣子代理。"
            "但子代理可能执行写操作（write_file, edit_file），与 gather 阶段'只读'的语义矛盾。"
            "虽然子代理有独立的工具白名单控制，但主图的 P3 阶段语义不够一致。"
        ),
        "suggestion": "考虑在 gather 阶段移除 dispatch_subagent_lg，或区分子代理类型（只读子代理 vs 写子代理）。",
    })

    # 6. Checkpointer SQLite 并发写入
    findings.append({
        "severity": "OPTIMIZE",
        "title": "SQLite Checkpointer 在高并发下可能成为瓶颈",
        "location": "lg_agent.py:_init_checkpointer() + factory.py",
        "description": (
            "SqliteSaver 使用 SQLite 文件数据库，在高并发场景下（多个用户同时请求），"
            "SQLite 的写锁可能导致性能瓶颈。当前每个 user_id 一个数据库文件，"
            "缓解了部分压力，但同一用户的并发请求仍会竞争。"
        ),
        "suggestion": "考虑使用 PostgreSQL 替代 SQLite（langgraph-checkpoint-postgres），或增加连接池和重试机制。",
    })

    # 7. 并行子代理的 ContextVar 传播
    findings.append({
        "severity": "BUG",
        "title": "并行子代理 worker 中 ContextVar 可能未正确传播",
        "location": "lg_graph.py:_subagent_worker()",
        "description": (
            "已修复：_subagent_worker 入口处添加了防御性 ContextVar 快照/恢复，"
            "使用 try/finally 确保所有路径都恢复原始上下文，"
            "防止子代理内部修改 ContextVar 影响其他并行 worker。"
        ),
        "suggestion": "[已修复] 使用 context_var_manager.snapshot/restore 防御性隔离。",
    })

    # 8. planner 使用无工具 LLM 但未绑定工具
    findings.append({
        "severity": "OPTIMIZE",
        "title": "planner 节点使用原始 LLM 而非 llm_with_tools 的无工具版本",
        "location": "lg_graph.py:_plan_node()",
        "description": (
            "_plan_node 使用 _ctx_llm_ref.get() 获取的原始 LLM 做规划。"
            "这个 LLM 没有绑定工具，所以不会产生 tool_calls，这是正确的。"
            "但如果 LLM 实例被其他地方修改（如绑定了工具），planner 可能意外调用工具。"
            "更安全的做法是显式使用 llm.bind_tools([]) 或 llm.with_config(...) 确保无工具。"
        ),
        "suggestion": "虽然当前实现安全（_ctx_llm_ref 是原始 LLM），但建议添加注释或显式保护。",
    })

    # 9. 100个境界任务的迭代次数可能不足
    findings.append({
        "severity": "OPTIMIZE",
        "title": "复杂任务可能因 recursion_limit 不足而中断",
        "location": "lg_agent.py + lg_graph.py",
        "description": (
            "已修复：recursion_limit 公式从 max_iterations*2+5 调整为 max_iterations*4+10。"
            "一次工具调用循环消耗 4-5 个 recursion step（agent→advance_phase→route→tools→agent），"
            "新公式确保 max_iterations 次工具调用不会触及限制。"
        ),
        "suggestion": "[已修复] recursion_limit 公式已调整。",
    })

    # 10. _extract_file_paths 正则匹配过于宽泛
    findings.append({
        "severity": "DEFECT",
        "title": "_extract_file_paths 正则可能误匹配非文件路径",
        "location": "lg_graph.py:_extract_file_paths()",
        "description": (
            "已修复：当前代码已使用文件扩展名白名单（_EXT_WHITELIST）过滤，"
            "不会匹配 '凝气期:1' 这种非文件路径。原描述中的正则已过时。"
        ),
        "suggestion": "[已修复] 扩展名白名单已生效，无需额外修改。",
    })

    return findings


if __name__ == "__main__":
    asyncio.run(run_test())
