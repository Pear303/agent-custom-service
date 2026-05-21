"""子 Agent 工具级并发执行。

恢复旧系统 `AgentRunner._execute_tool_blocks()` 中的 ThreadPoolExecutor 模式：
当 LLM 在同一帧回复中发出多个只读工具调用（如 web_fetch、read_file、grep），
用线程池并行执行而非顺序执行。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

from langchain_classic.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish, AgentStep


# 只读工具列表，可安全并发
_READ_ONLY_TOOLS = {
    "web_fetch",
    "read_file",
    "glob_tool",
    "grep_tool",
    "load_skill",
}


class ParallelAgentExecutor(AgentExecutor):
    """AgentExecutor 子类，同一帧内的只读工具调用并发执行。

    覆盖 _iter_next_step()，将连续出现的只读工具分组，
    用 ThreadPoolExecutor 并行执行，保持原始顺序返回结果。
    """

    # 同一帧内多个只读工具 → 并发执行
    def _iter_next_step(
        self,
        name_to_tool_map: dict[str, Any],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: Any = None,
    ) -> Iterator[AgentFinish | AgentAction | AgentStep]:
        """与父类相同，但只读工具并发执行。"""
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)
            output = self._action_agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except Exception as e:
            from langchain_classic.agents import OutputParserException
            from langchain_core.tools import InvalidTool

            if not isinstance(e, OutputParserException):
                raise
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(str(e)) from e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`") from e
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            observation = InvalidTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            yield AgentStep(action=output, observation=observation)
            return

        if isinstance(output, AgentFinish):
            yield output
            return

        actions: list[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output

        # 第一遍：先 yield 所有 action（供日志/回调使用）
        for agent_action in actions:
            yield agent_action

        # 第二遍：将只读工具分组并发执行，非只读工具顺序执行
        # 按连续出现的只读工具分组（保持旧系统行为）
        i = 0
        while i < len(actions):
            agent_action = actions[i]

            if agent_action.tool not in name_to_tool_map:
                # 工具不存在 → 顺序执行（单步）
                yield self._perform_agent_action(
                    name_to_tool_map, color_mapping, agent_action, run_manager,
                )
                i += 1
                continue

            tool = name_to_tool_map[agent_action.tool]
            is_read_only = agent_action.tool in _READ_ONLY_TOOLS

            if not is_read_only:
                # 非只读工具 → 顺序执行（单步）
                yield self._perform_agent_action(
                    name_to_tool_map, color_mapping, agent_action, run_manager,
                )
                i += 1
                continue

            # 收集连续出现的只读工具
            group = []
            while i < len(actions):
                a = actions[i]
                if a.tool not in _READ_ONLY_TOOLS or a.tool not in name_to_tool_map:
                    break
                group.append(a)
                i += 1

            if len(group) == 1:
                # 只有一个，顺序执行即可
                yield self._perform_agent_action(
                    name_to_tool_map, color_mapping, group[0], run_manager,
                )
                continue

            # 多个只读工具 → 并发执行
            if self.verbose:
                names = ", ".join(a.tool for a in group)
                print(f"\n[子代理并发执行 {len(group)} 个工具]: {names}\n")

            tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
            with ThreadPoolExecutor(max_workers=len(group)) as pool:

                def _run(a: AgentAction) -> AgentStep:
                    t = name_to_tool_map[a.tool]
                    obs = t.run(
                        a.tool_input,
                        verbose=self.verbose,
                        color=color_mapping.get(a.tool, ""),
                        callbacks=run_manager.get_child() if run_manager else None,
                        **tool_run_kwargs,
                    )
                    return AgentStep(action=a, observation=obs)

                results = list(pool.map(_run, group))

            for step in results:
                yield step
