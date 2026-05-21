# OPC Smart Customer Service System

基于 LangChain 的多轮对话 AI 服务系统。异步任务队列处理多会话并发、API 层与 Agent 核心分离解耦、内置 Web UI 支持流式对话展示。提供 CLI 与 REST API 双入口，可作为远程服务独立部署。

## 核心特性

- **异步任务处理**：FastAPI + asyncio 实现非阻塞并发，支持多会话并行处理
- **开发与业务分离**：API 层（`api/`）负责接口与路由，Agent 核心（`agent/`）专注智能决策
- **前端交互界面**：内置 Web UI（`static/index.html`），支持实时流式对话展示

## 快速开始

```bash
# 安装
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env       # 填入 DEEPSEEK_API_KEY

# 启动服务 → http://localhost:8080
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# 或使用 CLI 交互模式
python agent.py
```

## 项目结构

```
agent.py                  CLI 入口
api/                      FastAPI 服务层（开发与业务分离）
├── main.py               API 路由与 SSE 流式输出
├── agent_service.py      Agent 调用封装
├── session_manager.py    会话管理
├── task_queue.py         异步任务队列
├── database.py           SQLite 数据库管理
├── config.py             服务配置
├── schemas.py            请求/响应模型
└── tools/                API 层专用工具（helpdesk、通知、产品目录等）
agent/                    Agent 核心逻辑
├── lc_agent.py           主 Agent 循环（LCAgent）
├── lc_tools.py           工具定义（@tool 函数）
├── subagent_parallel.py  子代理只读工具并发执行器
├── memory.py             三层记忆存储
├── compactor.py          历史压缩 → 情景记忆 + MEMORY.md
├── context.py            system prompt 构建
├── skills.py             技能加载器
├── telemetry.py          token 用量追踪
├── factory.py            Agent 工厂（共享资源缓存）
├── todo.py               update_todos 实现
├── embeddings/           向量嵌入（ChromaDB 存储与检索）
└── subagents/
    ├── registry.py       子代理注册表（工具白名单 + max_turns）
    └── spec.py           子代理规格定义

static/                   前端资源
└── index.html            Web UI 界面（实时对话展示）
templates/                身份/引导与提示词模板
├── SOUL.md               Agent 身份引导
├── SOUL_CS.md            客服身份引导
├── USER.md               用户偏好档案
├── agent/                Agent 系统提示词模板
└── subagents/            子代理身份模板（11 种）
services/                 外部服务集成
└── dify/                 Dify 智能客服客户端
scripts/                  工具脚本（数据库迁移、数据校验等）
tests/                    测试套件
Dockerfile                Docker 容器镜像
docker-compose.yml        Docker Compose 编排
skills/                   可插拔技能包
data/                     数据存储（数据库文件、用户数据等）
```

## 三层记忆

| 层 | 载体 | 何时写 | 何时读 |
|----|------|--------|--------|
| 工作记忆 | `history` 列表（内存） | 每轮追加 | 全量传给 LLM |
| 情景记忆 | `memory/YYYY-MM-DD.md` | 压缩触发时 | 后续压缩时读取旧摘要 |
| 长期记忆 | `memory/MEMORY.md` | 压缩/启动归档时 | 每轮注入 system prompt |

**自动压缩**：当 input_tokens 超过 200K × 50% = 100K 时，将较旧的历史浓缩为情景摘要并更新长期记忆，保留最近 10 轮。

## 内置工具

| 工具 | 说明 |
|------|------|
| `run_command` | 执行 shell 命令 |
| `web_fetch` | 抓取网页 |
| `read_file` / `write_file` / `edit_file` | 文件读写编辑 |
| `glob` / `grep` | 工作区搜索 |
| `load_skill` | 按需加载技能包 |
| `update_todos` | 任务规划 todolist |
| `dispatch_subagent` | 派遣子代理（11 种身份，独立上下文） |

## 子代理

派遣后拥有独立的运行上下文，办完只回传一段总结。**主Agent** 按顺序逐个派遣，但**子代理内部**的连续只读工具（web_fetch、read_file、glob、grep）会通过 `ThreadPoolExecutor` 并发执行。

身份定义在 `templates/subagents/{name}.md`，工具白名单和最大轮数写在 `registry.py`（安全设置不放模板）。

可用子代理分为两组：
- **开发者子代理**：`quick_helper`、`doc_analyzer`、`web_researcher`、`validator`、`engine_executor`、`skill_manager`、`document_processor`、`system_maintainer`
- **业务子代理**：`requirement_analyst`（需求分析）、`product_manager`（PRD 设计）、`cost_estimator`（成本估算）

子代理不能递归派遣子代理，不能修改主 Agent 的 todolist。

## 内置技能

基础：clawhub（技能安装）、github、skill-creator、summarize、weather  
浏览器：agent-browser（基于 Rust，需 node/npm）  
文档：pdf、word-docx、pptx、xlsx  
设计：ui-ux-pro-max  
知识：ontology、self-improving-agent、ddg-web-search、find-skills  
维护：auto-updater

> **注意**：上述 skill 中除 `clawhub`、`github`、`skill-creator`、`summarize`、`weather` 外，
> 均为第三方版权内容，**本仓库不包含**。详见下方"第三方技能声明"。

## 第三方技能声明

### 版权归属

以下技能因版权/许可证限制，**不在本仓库中分发**，需要时请自行通过 ClawHub 安装：

| 技能 | 来源 | 许可证 | 原因 |
|------|------|--------|------|
| `pdf` | Anthropic | 专有许可 | 明确禁止复制、分发、创建衍生作品 |
| `pptx` | Anthropic | 专有许可 | 同上 |
| `xlsx` | Anthropic | 专有许可 | 同上 |
| `agent-browser` | ClawHub 发布 | 无开源许可证 | All Rights Reserved，版权归原作者 |
| `auto-updater` | ClawHub 发布 | 无开源许可证 | 同上 |
| `ddg-web-search` | ClawHub 发布 | 无开源许可证 | 同上 |
| `find-skills` | ClawHub 发布 | 无开源许可证 | 同上 |
| `ontology` | ClawHub 发布 | 无开源许可证 | 同上 |
| `self-improving-agent` | ClawHub 发布 | 无开源许可证 | 同上 |
| `ui-ux-pro-max` | ClawHub 发布 | 无开源许可证 | 同上 |
| `word-docx` | ClawHub 发布 | 无开源许可证 | 同上 |

### 缺失技能对子代理的影响

本仓库开源版保留了 5 个安全技能（`clawhub`、`github`、`skill-creator`、`summarize`、`weather`），
其余 11 个技能已被移除。以下说明各子代理在缺失状态下的能力变化：

| 子代理 | 仍拥有的技能 | 失去的技能 | 剩余能力 |
|--------|------------|-----------|---------|
| `quick_helper` | （无 skill 映射） | 无 | 🟢 完全不受影响。短命令、快速查询照常 |
| `doc_analyzer` | `summarize` | `pdf`、`pptx`、`ontology`、`word-docx`、`xlsx` | 🟡 仍可读文件/代码分析，失去专用文档格式指南和知识图谱 |
| `web_researcher` | `summarize` | `agent-browser`、`ddg-web-search` | 🟡 仍可用 `web_fetch` 搜索网页，失去浏览器自动化和备用搜索方案 |
| `validator` | `summarize` | `pdf`、`xlsx` | 🟡 仍可做常规文件校验，失去专用格式校验指南 |
| `engine_executor` | `github` | `agent-browser`、`pdf`、`pptx`、`ui-ux-pro-max`、`word-docx`、`xlsx` | 🟡 读写文件/执行命令能力完全保留，失去浏览器/GitHub/文档/设计专用知识 |
| `skill_manager` | `clawhub`、`skill-creator` | `find-skills` | 🟡 仍可安装和管理技能，失去主动搜索发现能力 |
| `document_processor` | （无） | `pdf`、`pptx`、`ui-ux-pro-max`、`word-docx`、`xlsx` | 🟠 完全失去文档格式专项知识。需要处理文档时建议手动安装对应 skill |
| `system_maintainer` | （无） | `auto-updater`、`ontology`、`self-improvement` | 🟠 完全失去运维知识。建议手动安装 |

> **重要提示**：子代理仍拥有 `read_file`、`run_command`、`web_fetch` 等通用工具，不会瘫痪。
> 失去的是"最佳实践指南"而非基础能力。LLM 自身的领域知识（如 PDF 操作、Excel 公式）仍然可用，
> 只是不如 skill 提供的高度定制化指南准确。建议按需手动安装。

### 建议安装命令

```
# 安装前请确认各技能的传播协议，尊重原作者版权
npx clawhub install agent-browser     # 浏览器自动化
npx clawhub install auto-updater      # 自动更新
npx clawhub install ddg-web-search    # DuckDuckGo 搜索
npx clawhub install find-skills       # 技能发现
npx clawhub install ontology          # 知识图谱
npx clawhub install self-improving-agent  # 自我改进
npx clawhub install ui-ux-pro-max     # UI/UX 设计
npx clawhub install word-docx         # Word 文档

# Anthropic 专有技能（pdf/pptx/xlsx）不可通过 ClawHub 安装，
# 它们是 AI 工具内建功能的一部分，仅在原服务中使用。
```

> **注意**：`pdf`、`pptx`、`xlsx` 三个技能属于 Anthropic 专有内容，
> 只能在 Claude/OpenCode 等原 AI 工具中自动使用，不可提取到外部环境。

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | API 地址（默认 https://api.deepseek.com） |
| `DEEPSEEK_MODEL` | 使用的模型（默认 deepseek-v4-flash） |

其余环境变量请查看 `.env.example`。