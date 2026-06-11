"""6号对照测试 —— 验证 Planner 派遣决策树的合理性。

两个对照任务：
  A. 简单任务（1个文件）→ 不应派遣子代理
  B. 复杂任务（3+个文件）→ 应派遣子代理

验证点：
  - 简单任务：dispatch_subagent_lg 不被调用
  - 复杂任务：dispatch_subagent_lg 被调用
  - 两个任务的 Planner 输出都遵循决策树

使用方式：
    python test/test_dispatch_decision.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 常量 ──────────────────────────────────────────────────────

TEST_USER_ID = "test_dispatch_001"
TEST_TICKET_ID_A = "simple_task"
TEST_TICKET_ID_B = "complex_task"

SIMPLE_PROMPT = """\
请在当前工作目录创建一个 Python 脚本 calc.py，实现一个简单的四则运算计算器：
1. 支持加减乘除
2. 命令行输入表达式，如: python calc.py "2+3*4"
3. 输出计算结果
4. 只使用 Python 标准库

创建后用 run_command 运行 python calc.py "2+3*4" 验证结果为 14。
"""

COMPLEX_PROMPT = """\
你需要开发一个 Python CLI 待办管理应用。这是一个完整的项目，包含主程序、测试和文档。

## 项目需求

### 主程序 todo_app.py
实现一个命令行待办管理工具，功能如下：
1. 添加待办：python todo_app.py add "买菜"
2. 完成待办：python todo_app.py done 1
3. 列出所有待办：python todo_app.py list
4. 删除待办：python todo_app.py delete 1
5. 数据存储在 todo_data.json 文件中
6. 只使用 Python 标准库

### 测试文件 test_todo_app.py
使用 unittest 编写测试：
1. 测试添加待办
2. 测试完成待办
3. 测试列出待办
4. 测试删除待办
5. 至少 6 个测试用例
6. 使用 python test_todo_app.py 即可运行

### 文档 README.md
1. 项目简介
2. 功能列表
3. 使用示例

## 工作要求
1. 先规划再编码
2. 创建文件后用 run_command 运行测试验证
3. 如果测试失败，用 edit_file 修复
"""

TIMEOUT_S = 360

# ── 日志 ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            str(Path(__file__).parent.parent / "test_dispatch_decision.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("test_dispatch")


# ── 事件监听 ──────────────────────────────────────────────────

class TaskMonitor:
    """监控单个任务的执行过程。"""

    def __init__(self, label: str):
        self.label = label
        self.planner_output: str = ""
        self.planner_phase: str = ""
        self.tool_calls: list[dict] = []
        self.dispatch_called: bool = False
        self.dispatch_details: list[dict] = []
        self.node_counts: dict[str, int] = {}
        self.llm_calls: int = 0
        self.total_tokens: int = 0
        self.phase_transitions: list[dict] = []
        self.interrupt_count: int = 0
        self.subagent_worker_count: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "planner_output": self.planner_output,
            "planner_phase": self.planner_phase,
            "tool_calls": self.tool_calls,
            "dispatch_called": self.dispatch_called,
            "dispatch_details": self.dispatch_details,
            "node_counts": self.node_counts,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "phase_transitions": self.phase_transitions,
            "interrupt_count": self.interrupt_count,
            "subagent_worker_count": self.subagent_worker_count,
        }


async def _run_task(prompt: str, ticket_id: str, monitor: TaskMonitor):
    """执行单个任务并监听事件。复用5号测试的 agent 创建方式。"""
    from agent_by_langgraph.factory import create_lg_agent, reset_lg_agent
    from agent_by_langgraph.lg_agent import ReasoningCollector
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langgraph.types import Command

    agent = create_lg_agent(
        user_id=TEST_USER_ID,
        ticket_id=ticket_id,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        max_iterations=40,
    )

    has_checkpointer = agent.will_have_checkpointer
    logger.info("[%s] checkpointer 初始状态: %s", monitor.label, has_checkpointer)

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

    collector = ReasoningCollector()
    await agent._ensure_checkpointer()
    has_checkpointer = agent.checkpointer_ready

    config = {
        "callbacks": [collector],
        "recursion_limit": 170,
        "configurable": {
            "thread_id": f"{TEST_USER_ID}_{ticket_id}",
            "__has_checkpointer__": has_checkpointer,
        },
    }

    _TRACKED_NODES = {
        "agent", "advance_phase", "tools", "interrupt_approval",
        "aggregate_results", "subagent_dispatcher", "subagent_worker",
        "planner", "route_after_agent", "route_after_aggregate",
        "route_after_approval",
    }

    last_phase = "all"
    current_input = input_state
    max_interrupt_retries = 30

    for retry in range(max_interrupt_retries):
        try:
            async for event in agent.graph.astream_events(
                current_input, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                # 节点计数
                if kind == "on_chain_start" and name in _TRACKED_NODES:
                    monitor.node_counts[name] = monitor.node_counts.get(name, 0) + 1

                # P3 阶段转换
                if kind == "on_chain_end" and name == "advance_phase":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        new_phase = output.get("_phase")
                        if new_phase and new_phase != last_phase:
                            monitor.phase_transitions.append({
                                "from": last_phase, "to": new_phase
                            })
                            logger.info("[%s] P3 阶段: %s → %s",
                                        monitor.label, last_phase, new_phase)
                            last_phase = new_phase

                # Planner 输出
                if kind == "on_chain_end" and name == "planner":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        plan = output.get("plan", "")
                        phase_hint = output.get("_phase", "")
                        if plan:
                            monitor.planner_output = plan
                            monitor.planner_phase = phase_hint
                        logger.info("[%s] Planner: phase=%s, plan=%s",
                                    monitor.label, phase_hint,
                                    plan[:300] if plan else "空")

                # 工具调用
                if kind == "on_chat_model_end":
                    monitor.llm_calls += 1
                    msg = event.get("data", {}).get("output", {})
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        monitor.total_tokens += usage.get("total_tokens", 0)

                    tool_calls = getattr(msg, "tool_calls", None) or []
                    for tc in tool_calls:
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("args", {})
                        monitor.tool_calls.append({
                            "tool": tool_name, "args": tool_args
                        })
                        logger.info("[%s] 工具: %s(%s)",
                                    monitor.label, tool_name,
                                    json.dumps(tool_args, ensure_ascii=False)[:120])

                        if tool_name == "dispatch_subagent_lg":
                            monitor.dispatch_called = True
                            monitor.dispatch_details.append(tool_args)
                            logger.info("[%s] ★ 子代理派遣: %s",
                                        monitor.label,
                                        json.dumps(tool_args, ensure_ascii=False))

                # Interrupt
                if kind == "on_chain_start" and name == "interrupt_approval":
                    monitor.interrupt_count += 1
                    logger.info("[%s] interrupt (第%d次)",
                                monitor.label, monitor.interrupt_count)

                # 子代理 worker
                if kind == "on_chain_start" and name == "subagent_worker":
                    monitor.subagent_worker_count += 1
                    logger.info("[%s] subagent_worker 启动 (第%d次)",
                                monitor.label, monitor.subagent_worker_count)

        except Exception as exc:
            logger.error("[%s] 执行异常: %s", monitor.label, exc)
            try:
                reset_lg_agent(TEST_USER_ID, ticket_id)
            except Exception:
                pass
            return

        # 检查 interrupt
        has_interrupt = False
        if has_checkpointer:
            try:
                snapshot = await agent.graph.aget_state(config)
                if snapshot and snapshot.next:
                    for task in snapshot.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            has_interrupt = True
                            logger.info("[%s] interrupt 自动批准", monitor.label)
                            current_input = Command(resume="approve")
                            break
            except Exception as exc:
                logger.warning("[%s] aget_state 失败: %s", monitor.label, exc)

        if not has_interrupt:
            break

    # 清理
    try:
        reset_lg_agent(TEST_USER_ID, ticket_id)
    except Exception:
        pass

    logger.info("[%s] 完成: dispatch=%s, tools=%s, tokens=%d, nodes=%s",
                monitor.label, monitor.dispatch_called,
                set(tc["tool"] for tc in monitor.tool_calls),
                monitor.total_tokens, monitor.node_counts)


# ── 评估 ──────────────────────────────────────────────────────

def evaluate(simple: TaskMonitor, complex_: TaskMonitor) -> dict:
    """评估决策树的合理性。"""
    checks = []

    # 1. 简单任务不应派遣
    checks.append({
        "name": "simple_no_dispatch",
        "passed": not simple.dispatch_called,
        "detail": f"简单任务 dispatch 调用: {len(simple.dispatch_details)} 次",
    })

    # 2. 复杂任务应派遣
    checks.append({
        "name": "complex_has_dispatch",
        "passed": complex_.dispatch_called,
        "detail": f"复杂任务 dispatch 调用: {len(complex_.dispatch_details)} 次",
    })

    # 3. Planner 计划一致性
    simple_plan_has_dispatch = "派遣" in simple.planner_output
    complex_plan_has_dispatch = "派遣" in complex_.planner_output
    checks.append({
        "name": "planner_simple_no_dispatch",
        "passed": not simple_plan_has_dispatch,
        "detail": f"简单任务计划含'派遣': {simple_plan_has_dispatch}",
    })
    checks.append({
        "name": "planner_complex_has_dispatch",
        "passed": complex_plan_has_dispatch,
        "detail": f"复杂任务计划含'派遣': {complex_plan_has_dispatch}",
    })

    # 4. Planner 计划与实际执行一致
    simple_consistent = simple_plan_has_dispatch == simple.dispatch_called
    complex_consistent = complex_plan_has_dispatch == complex_.dispatch_called
    checks.append({
        "name": "plan_execution_consistency",
        "passed": simple_consistent and complex_consistent,
        "detail": f"简单: 计划={simple_plan_has_dispatch}/实际={simple.dispatch_called}, "
                  f"复杂: 计划={complex_plan_has_dispatch}/实际={complex_.dispatch_called}",
    })

    # 5. Token 效率
    if simple.total_tokens > 0:
        token_ratio = complex_.total_tokens / simple.total_tokens
        checks.append({
            "name": "token_reasonable",
            "passed": token_ratio < 8,
            "detail": f"复杂/简单 token 比: {token_ratio:.1f}x "
                      f"(简单={simple.total_tokens}, 复杂={complex_.total_tokens})",
        })
    else:
        checks.append({
            "name": "token_reasonable",
            "passed": False,
            "detail": "简单任务 token 为 0，无法计算比率",
        })

    # 6. 简单任务步骤少
    checks.append({
        "name": "simple_fewer_steps",
        "passed": len(simple.tool_calls) < len(complex_.tool_calls),
        "detail": f"简单任务工具调用: {len(simple.tool_calls)} 次, "
                  f"复杂任务工具调用: {len(complex_.tool_calls)} 次",
    })

    return {
        "checks": checks,
        "pass_count": sum(1 for c in checks if c["passed"]),
        "total_count": len(checks),
    }


# ── 主流程 ────────────────────────────────────────────────────

async def run_test():
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("AUTO_APPROVE_WITHOUT_CHECKPOINTER", "true")

    if not os.environ.get("DEEPSEEK_API_KEY") or \
       os.environ["DEEPSEEK_API_KEY"] == "your_api_key_here":
        print("ERROR: DEEPSEEK_API_KEY 未配置！")
        return

    logger.info("=" * 60)
    logger.info("6号对照测试: Planner 派遣决策树验证")
    logger.info("=" * 60)

    # ── 任务A: 简单任务 ─────────────────────────────────
    logger.info("── 任务A: 简单任务（1个文件，不应派遣）──")
    monitor_a = TaskMonitor("简单任务")
    try:
        await asyncio.wait_for(
            _run_task(SIMPLE_PROMPT, TEST_TICKET_ID_A, monitor_a),
            timeout=TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("任务A 超时")
    except Exception as e:
        logger.error("任务A 异常: %s", e)

    # ── 任务B: 复杂任务 ─────────────────────────────────
    logger.info("── 任务B: 复杂任务（3+文件，应派遣）──")
    monitor_b = TaskMonitor("复杂任务")
    try:
        await asyncio.wait_for(
            _run_task(COMPLEX_PROMPT, TEST_TICKET_ID_B, monitor_b),
            timeout=TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("任务B 超时")
    except Exception as e:
        logger.error("任务B 异常: %s", e)

    # ── 评估 ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)

    eval_result = evaluate(monitor_a, monitor_b)

    for c in eval_result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        logger.info("  [%s] %s: %s", status, c["name"], c["detail"])

    logger.info("总计: %d/%d 通过", eval_result["pass_count"], eval_result["total_count"])

    # ── Planner 输出对比 ────────────────────────────────
    logger.info("")
    logger.info("── Planner 输出对比 ──")
    logger.info("[简单任务 Planner]:\n%s", monitor_a.planner_output or "(无)")
    logger.info("[复杂任务 Planner]:\n%s", monitor_b.planner_output or "(无)")

    # ── 保存结果 ────────────────────────────────────────
    result_path = Path(__file__).parent.parent / "test_dispatch_decision_result.json"
    result_data = {
        "simple_task": monitor_a.to_dict(),
        "complex_task": monitor_b.to_dict(),
        "evaluation": eval_result,
    }
    result_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    logger.info("结果已保存: %s", result_path)

    return eval_result


if __name__ == "__main__":
    asyncio.run(run_test())
