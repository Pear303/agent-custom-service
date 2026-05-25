"""Agent 服务：Dify 路由 + 兜底 Agent 逻辑 + 需求分析"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .session_manager import SessionManager, Session
from ..clients.dify import DifyChatflowClient
from ..utils.file_manager import _clean_output_path
from agent.factory import create_agent
from agent.lc_tools import set_workspace, set_skills_loader, set_todo_store, set_subagent_deps, set_user_id, set_ticket_id, clear_context, _build_workspace

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent

_AGENT_CONCURRENCY_LIMIT = 5
_agent_semaphore = asyncio.Semaphore(_AGENT_CONCURRENCY_LIMIT)

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0


def _parse_json_safe(content: str, _debug_label: str = "") -> dict | None:
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()

    strategies = [
        lambda c: json.loads(c),
        lambda c: json.loads(re.sub(r',\s*([}\]])', r'\1', c)),
        lambda c: _try_truncated_json(c),
        lambda c: json.loads(re.sub(r'(?m)^```(?:json)?\s*\n?|^\s*```\s*$', '', c).strip()),
        lambda c: json.loads(re.sub(r"(?<!\\)'", '"', c)),
        lambda c: _try_truncated_json(re.sub(r"(?<!\\)'", '"', c)),
    ]
    for _ in strategies:
        try:
            return _(content)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    _dump_debug_json(content, _debug_label)
    return None


def _dump_debug_json(content: str, label: str = "") -> None:
    debug_dir = _PROJECT_ROOT / "data" / "_json_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_label = re.sub(r"[^\w一-龥]", "_", label)[:40] if label else "unknown"
    filepath = debug_dir / f"parse_fail_{ts}_{safe_label}.json"
    filepath.write_text(content, encoding="utf-8")
    logger.warning("废弃 JSON 已保存到 %s (%d 字节)", filepath, len(content))


def _try_truncated_json(content: str) -> dict:
    start = content.find('{')
    if start == -1:
        start = content.find('[')
        close_char = ']'
        open_char = '['
    else:
        close_char = '}'
        open_char = '{'
    bracket_open = '['
    bracket_close = ']'
    if start == -1:
        raise json.JSONDecodeError("No opening brace/bracket found", content, 0)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if e.pos > 0 and "Extra data" in str(e):
            try:
                return json.loads(content[:e.pos])
            except json.JSONDecodeError:
                pass

    for end in range(len(content) - 1, start, -1):
        if content[end] == close_char:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                continue

    depth = {close_char: 0, bracket_close: 0}
    for ch in content[start:]:
        if ch == open_char:
            depth[close_char] += 1
        elif ch == close_char:
            depth[close_char] = max(0, depth[close_char] - 1)
        elif ch == bracket_open:
            depth[bracket_close] += 1
        elif ch == bracket_close:
            depth[bracket_close] = max(0, depth[bracket_close] - 1)
    suffix = bracket_close * depth[bracket_close] + close_char * depth[close_char]
    if suffix:
        return json.loads(content[start:] + suffix)

    raise json.JSONDecodeError("No valid JSON found in truncated content", content, 0)


def _invoke_in_thread(workspace, user_id, ticket_id, executor, input_data, output_subdir=None,
                       skills_loader=None, todo_store=None, sub_reg=None):
    """在线程中设置上下文变量 (ContextVar) 后执行调用。

    asyncio.to_thread 不会自动复制上下文变量，因此必须在新线程中显式设置。
    workspace 参数代表项目根目录，此函数会将其重置为完整的目标路径。
    output_subdir: 非空时使用 _build 临时目录作为 workspace，
                   避免 LLM 的 write_file 与最终输出目录冲突导致路径嵌套。
    """
    effective_ws = _build_workspace(workspace, user_id, ticket_id)
    if output_subdir:
        # 使用 _build 临时目录：LLM 在此自由操作，不会污染最终输出目录
        effective_ws = effective_ws / "_build"
    effective_ws.mkdir(parents=True, exist_ok=True)
    set_workspace(effective_ws)
    set_user_id(user_id)
    set_ticket_id(ticket_id)
    if skills_loader is not None:
        set_skills_loader(skills_loader)
    if todo_store is not None:
        set_todo_store(todo_store)
    if sub_reg is not None:
        # sub_reg 是一个元组 (llm, registry) 或 registry 对象
        if isinstance(sub_reg, tuple) and len(sub_reg) == 2:
            set_subagent_deps(sub_reg[0], sub_reg[1])
        else:
            set_subagent_deps(None, sub_reg)
    try:
        return _invoke_with_retry(executor, input_data)
    finally:
        clear_context()


def _invoke_with_retry(executor, input_data, max_retries: int = _MAX_RETRIES):
    """带有重试机制的同步执行器调用。

    Args:
        executor: AgentExecutor 实例
        input_data: 输入数据字典
        max_retries: 最大重试次数

    Returns:
        执行结果字典

    Raises:
        若所有重试均失败，则抛出最后一次捕获的异常
    """
    import time
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return executor.invoke(input_data)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning("第 %d 次尝试失败，%0.1f 秒后重试: %s", attempt + 1, _RETRY_DELAY, exc)
                time.sleep(_RETRY_DELAY)
    raise last_exc

REQUIREMENT_ANALYST_PROMPT = """你是需求分析师，负责将客户模糊的原始需求转化为结构化的需求简报。

收到客户需求后，按以下维度分析：

1. **项目概述**：一句话概括项目本质
2. **目标用户**：谁会使用这个产品
3. **核心功能**：3-5 个最关键的功能点
4. **非功能需求**：性能、安全、兼容性等
5. **约束条件**：预算、时间、技术限制
6. **风险点**：可能的技术或业务风险
7. **待澄清问题**：需要客户补充的信息

最后输出结构化需求简报（JSON 格式），并给出复杂度评估（简单/中等/复杂）。"""

PRODUCT_MANAGER_PROMPT = """你是产品经理，负责将需求分析转化为完整的产品需求文档（PRD）。

收到需求分析后，按以下维度设计：

1. **产品定位**：一句话描述产品价值和差异化
2. **功能清单**：按优先级排序（P0 核心/P1 重要/P2 锦上添花）
3. **用户故事**：3-5 个核心用户场景的完整描述
4. **信息架构**：主要页面和导航结构
5. **数据模型**：核心数据实体和关系
6. **验收标准**：每个 P0 功能的完成定义

最后输出 PRD（JSON 格式），包含功能总数、核心场景数和技术复杂度评估。"""

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
- 售后支持期限（月）"""


class AgentService:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self._dify: DifyChatflowClient | None = None
        self._llm: AsyncOpenAI | None = None

    async def _get_dify(self) -> DifyChatflowClient:
        if self._dify is None:
            self._dify = DifyChatflowClient()
        return self._dify

    def _get_llm(self) -> AsyncOpenAI:
        if self._llm is None:
            self._llm = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        return self._llm

    async def chat(self, user_id: str, message: str) -> dict:
        session = await self.session_manager.get_or_create_async(user_id)
        dify = await self._get_dify()

        try:
            resp = await dify.chat(
                query=message,
                user_id=user_id,
                conversation_id=session.conversation_id,
            )
            session.conversation_id = resp.get("conversation_id")
            answer = resp.get("answer", "")
            source = "dify"
        except Exception as exc:
            logger.warning("Dify 调用失败，使用兜底 Agent: %s", exc)
            fallback = await self._fallback(user_id, message)
            answer = fallback["answer"]
            source = fallback["source"]

        session.history.append({"role": "user", "content": message, "timestamp": int(time.time())})
        session.history.append({"role": "assistant", "content": answer, "source": source, "timestamp": int(time.time())})
        await self.session_manager._save_session(session)
        return {
            "user_id": user_id,
            "answer": answer,
            "conversation_id": session.conversation_id,
            "source": source,
        }

    async def chat_stream(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        session = await self.session_manager.get_or_create_async(user_id)
        dify = await self._get_dify()
        full_answer = ""

        try:
            async for chunk in dify.chat_stream(
                query=message,
                user_id=user_id,
                conversation_id=session.conversation_id,
            ):
                event = chunk.get("event")
                if event == "message":
                    chunk_text = chunk.get("answer", "")
                    full_answer += chunk_text
                    yield json.dumps({
                        "event": "message",
                        "answer": chunk_text,
                        "source": "dify",
                    }) + "\n"
                elif event == "message_end":
                    session.conversation_id = chunk.get("conversation_id")
                    session.history.append({"role": "user", "content": message, "timestamp": int(time.time())})
                    session.history.append({"role": "assistant", "content": full_answer, "source": "dify", "timestamp": int(time.time())})
                    await self.session_manager._save_session(session)
                    yield json.dumps({
                        "event": "message_end",
                        "conversation_id": chunk.get("conversation_id"),
                    }) + "\n"
                    return
                elif event == "error":
                    raise RuntimeError(chunk.get("message", "Dify stream error"))
        except Exception as exc:
            logger.warning("Dify 流式调用失败，使用兜底 Agent: %s", exc)
            fallback_answer = await self._fallback(user_id, message)
            full_answer = fallback_answer["answer"]
            yield json.dumps({
                "event": "message",
                "answer": full_answer,
                "source": "agent",
            }) + "\n"
            session.history.append({"role": "user", "content": message, "timestamp": int(time.time())})
            session.history.append({"role": "assistant", "content": full_answer, "source": "agent", "timestamp": int(time.time())})
            await self.session_manager._save_session(session)
            yield json.dumps({"event": "message_end"}) + "\n"

    async def analyze_requirement(self, user_id: str, requirement: dict, ticket_id: str | None = None) -> dict:
        """需求分析：将客户需求转化为结构化需求简报"""
        async with _agent_semaphore:
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            prompt = f"""基于以下客户需求，按要求输出 JSON 格式的需求分析结果：

{json.dumps(requirement, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{REQUIREMENT_ANALYST_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("需求分析响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "需求分析响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("需求分析 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"需求分析格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("需求分析失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    async def design_prd(self, user_id: str, analysis: dict, ticket_id: str | None = None) -> dict:
        async with _agent_semaphore:
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            prompt = f"""基于以下需求分析结果，按要求输出 JSON 格式的 PRD：

{json.dumps(analysis, ensure_ascii=False, indent=2)}

请严格按照以下系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{PRODUCT_MANAGER_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容（处理 LLM 可能返回的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("PRD 响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "PRD 响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("PRD JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"PRD 格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("PRD 设计失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    async def estimate_cost(self, user_id: str, prd: dict, analysis: dict, ticket_id: str | None = None) -> dict:
        async with _agent_semaphore:
            agent = create_agent(user_id=user_id, ticket_id=ticket_id)

            combined = {**prd, **analysis}
            prompt = f"""基于以下 PRD 和需求分析，按要求输出 JSON 格式的成本估算：

{json.dumps(combined, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{COST_ESTIMATOR_PROMPT}"""

            try:
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容（处理 LLM 可能返回的额外文本）
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("成本估算响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "成本估算响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("成本估算 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"成本估算格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("成本估算失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    DEVELOPER_PROMPT = """你是全栈开发工程师，负责根据产品需求文档（PRD）生成完整的项目代码。

收到 PRD 后，按以下维度生成代码：

1. **项目结构**：合理的目录组织
2. **核心代码**：实现所有 P0/P1 功能
3. **配置文件**：package.json / requirements.txt 等
4. **README**：项目说明和运行指南

最后输出开发结果（JSON 格式），包含：
- project_structure: 项目目录结构（树形文本）
- files: 生成的文件列表（路径 + 内容）
- tech_stack: 使用的技术栈
- setup_instructions: 安装和运行步骤"""

    async def develop_project(self, user_id: str, project_data: dict, ticket_id: str | None = None) -> dict:
        async with _agent_semaphore:
            agent = create_agent(user_id=user_id, ticket_id=ticket_id, max_iterations=80)

            prompt = f"""基于以下项目数据，按要求输出 JSON 格式的开发结果：

{json.dumps(project_data, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。

系统提示词：
{self.DEVELOPER_PROMPT}"""

            try:
                # 使用 asyncio.to_thread 避免阻塞事件循环
                result = await asyncio.to_thread(
                    _invoke_in_thread,
                    agent.root, user_id, ticket_id,
                    agent.executor,
                    {"input": prompt, "chat_history": []},
                    "成品",
                    skills_loader=agent.skills,
                    todo_store=agent.todo_store,
                    sub_reg=(agent.llm, agent.sub_reg),
                )
                content = result["output"]

                # 检测 Agent 迭代耗尽
                if "max iterations" in content.lower() or "Agent stopped" in content:
                    logger.error("Agent 迭代次数耗尽，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "Agent 迭代次数耗尽，项目过于复杂未能完成。可重试或简化需求。"}
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                
                # 提取 JSON 内容
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    content = content[json_start:json_end]
                else:
                    logger.error("开发响应中未找到 JSON 内容，原始输出: %s", content[:200])
                    return {"status": "failed", "error": "开发响应格式无效"}
                
                data = _parse_json_safe(content)
                if data is None:
                    raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)

                # 将 LLM 生成的代码文件写入 data/users/{user_id}/{ticket_id}/成品/
                files = data.get("files")
                if files and isinstance(files, list) and ticket_id:
                    output_root = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "成品"
                    saved_count = 0
                    for entry in files:
                        if not isinstance(entry, dict):
                            continue
                        file_path = entry.get("path") or entry.get("file") or entry.get("name")
                        file_content = entry.get("content") or entry.get("code") or ""
                        if not file_path or not file_content:
                            continue
                        safe_path = _clean_output_path(file_path, user_id, ticket_id)
                        target = output_root / safe_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(file_content, encoding="utf-8")
                        saved_count += 1
                    if saved_count > 0:
                        logger.info("开发成品已保存 %d 个文件到 %s", saved_count, output_root)
                        data["_output_dir"] = str(output_root)
                        data["_file_count"] = saved_count

                # 开发完成后清理 _build 临时目录
                _build_dir = _PROJECT_ROOT / "data" / "users" / user_id / ticket_id / "_build"
                if ticket_id and _build_dir.exists():
                    import shutil
                    try:
                        shutil.rmtree(_build_dir)
                        logger.info("已清理 _build 临时目录: %s", _build_dir)
                    except Exception as e:
                        logger.warning("清理 _build 目录失败: %s", e)

                return {"status": "completed", "data": data}
            except json.JSONDecodeError as exc:
                logger.error("开发 JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
                return {"status": "failed", "error": f"开发格式错误: {str(exc)}"}
            except Exception as exc:
                logger.error("开发失败: %s", exc)
                return {"status": "failed", "error": str(exc)}
            finally:
                clear_context()

    async def _call_llm(self, system_prompt: str, input_data: dict) -> dict:
        llm = self._get_llm()
        prompt = f"""基于以下输入数据，按要求输出 JSON 格式结果：

{json.dumps(input_data, ensure_ascii=False, indent=2)}

请严格按照系统提示词的要求输出 JSON 格式，不要其他内容。"""

        try:
            resp = await llm.chat.completions.create(
                model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content.strip()
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            
            # 提取 JSON 内容（处理 LLM 可能返回的额外文本）
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                logger.error("LLM 响应中未找到 JSON 内容，原始输出: %s", content[:200])
                return {"status": "failed", "error": "LLM 响应格式无效"}
            
            result = _parse_json_safe(content)
            if result is None:
                raise json.JSONDecodeError("所有 JSON 修复策略均失败", content, 0)
            return {"status": "completed", "data": result}
        except json.JSONDecodeError as exc:
            logger.error("LLM JSON 解析失败: %s\n原始内容(前2000字符): %s", exc, content[:2000])
            return {"status": "failed", "error": f"LLM 格式错误: {str(exc)}"}
        except Exception as exc:
            logger.error("LLM 调用失败: %s", exc)
            return {"status": "failed", "error": str(exc)}

    async def _fallback(self, user_id: str, message: str) -> dict:
        return {
            "user_id": user_id,
            "answer": f"抱歉，智能客服系统暂时不可用。您的问题是：「{message}」，已转人工处理。",
            "conversation_id": None,
            "source": "agent",
        }
