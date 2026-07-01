# OPC Smart Customer Service System

LangGraph 状态图驱动的 AI Agent 服务（底层基于 langchain-core 类型系统 + langchain-openai LLM 封装）。FastAPI + SSE 流式，三层记忆自动压缩，11 种子代理派遣，可插拔技能包，Vue 3 前端。CLI 与 REST API 双入口。

## 核心特性

- **异步任务处理**：FastAPI + asyncio 实现非阻塞并发，支持多会话并行处理
- **开发与业务分离**：API 层（`api/`）负责接口与路由，Agent 核心（`agent_by_langgraph/` + `agent_core/`）专注智能决策
- **前后端分离**：前端 Vue 3 项目（`frontend/`）独立构建部署，后端提供 REST API + SSE 流式

## 快速开始

```bash
# 后端安装
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
cp .env.example .env       # 填入 DEEPSEEK_API_KEY

# 前端构建（生产模式必需）
cd frontend
npm install
npm run build              # 产出 frontend/dist/
cd ..

# 启动服务 → http://localhost:8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# 启动服务 → http://localhost:8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 前端开发模式（另开终端）
cd frontend; npm run dev   # Vite → http://localhost:5173
$env:DEV_MODE="true"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 或使用 CLI 交互模式
python agent.py
```

> **Dify 集成说明**：本项目可通过 `DIFY_BASE_URL` / `DIFY_API_KEY` 连接到单独部署的 Dify 服务。
> 如需搭建 Dify，请参考 [Dify 官方文档](https://docs.dify.ai) 部署，不包含在本仓库中。

## RAG 知识库（可选）

`requirement_analyst` 子代理支持基于知识库的检索增强生成（RAG）。首次启用 RAG 功能时，
系统会自动从 HuggingFace 下载 Embedding 模型（`BAAI/bge-small-zh-v1.5`，约 100MB）。

> 国内用户已配置 `hf-mirror.com` 镜像，无需额外设置。

### 构建知识库

```bash
# 1. 将需求文档（.txt / .md / .pdf / .docx）放入 data/raw/ 目录
mkdir -p data/raw/需求分析
cp your_docs/*.md data/raw/需求分析/

# 2. 运行索引脚本（分块 → 向量化 → 入库 ChromaDB）
python -c "from agent_core.rag.indexer import index_knowledge_base; index_knowledge_base()"
```

索引后向量数据存储在 `data/chroma/`（已加入 `.gitignore`），模型缓存于 `agent/embeddings/models/`。

### 数据库目录

以下目录由系统首次运行时自动创建：

| 目录 | 说明 |
|------|------|
| `data/chroma/` | ChromaDB 向量库持久化 |
| `data/users/` | 用户会话数据（checkpoints.db、history.jsonl 等） |
| `agent/embeddings/models/` | HuggingFace Embedding 模型缓存 |
| `memory/` | 三层记忆存储（MEMORY.md、情景记忆、history.jsonl） |

以上目录均已加入 `.gitignore`，不随仓库分发。

## 项目结构

```
agent.py                  CLI 入口（LangGraph StateGraph Agent）
agent_lg.py               CLI 入口（同 agent.py，LangGraph Agent）
api/                      FastAPI 服务层
├── main.py               应用入口、SSE 流式输出、路由挂载
├── core/                 核心基础设施
│   ├── config.py         环境变量配置
│   ├── database.py       异步 SQLite 数据库
│   ├── lifespan.py       应用生命周期
│   └── rate_limit.py     速率限制中间件
├── routers/              路由模块（health、chat、session、task、dify_tools）
├── services/             业务逻辑层
│   ├── agent_service.py  Agent 服务编排
│   └── session_manager.py 会话管理
├── repositories/         数据访问层（会话、工单）
├── clients/              外部服务客户端（Dify）
├── tools/                客服工具集（工单、转人工、商品目录、通知）
├── task_queue.py          异步任务队列（Worker Pool）
├── schemas/              请求/响应模型（Pydantic）
└── utils/                工具函数（文件管理、进度计算）
agent_core/               Agent 核心基础设施（LLM 封装、三层记忆、压缩、子代理注册、工具、RAG）
├── compactor.py          历史压缩 → 情景记忆 + MEMORY.md
├── context.py            system prompt 构建
├── context_view.py       上下文视图
├── decision_summary.py   决策摘要
├── in_context_compactor.py  上下文内压缩
├── llm.py                LLM 调用封装
├── memory.py             三层记忆存储
├── observation_masker.py  观察掩码
├── skills.py             技能加载器
├── telemetry.py          token 用量追踪
├── todo.py               update_todos 实现
├── rag/                  RAG 模块
│   ├── chains.py         RAG 链路
│   ├── embeddings.py     向量嵌入
│   ├── indexer.py        索引器
│   ├── reranker.py       重排序
│   ├── retriever.py      检索器
│   └── vectorstore.py    向量存储
├── subagents/
│   ├── registry.py       子代理注册表（工具白名单 + max_turns）
│   └── spec.py           子代理规格定义
└── tools/                工具定义（文件、搜索、shell、技能、todo、web、workspace）
agent_by_langgraph/       LangGraph Agent 引擎（状态图定义、Agent 循环、子代理并发派遣）
├── lg_agent.py           LangGraph Agent 循环
├── lg_graph.py           状态图定义
├── lg_subagent.py        子代理并发执行
├── lg_parallel_tools.py  并发工具执行
├── lg_rag_subagent.py    RAG 子代理
├── lg_tools.py           工具定义
├── context_var_manager.py 上下文变量管理
├── level_router.py       级别路由
└── factory.py            Agent 工厂
frontend/                 Vue 3 前端（前后端分离，独立构建）
├── src/
│   ├── main.ts           Vue 应用入口
│   ├── App.vue           根组件（Header + Tabs + Router View）
│   ├── api/              API 层（chat.ts SSE 流式引擎、task.ts 工单 CRUD）
│   ├── stores/           Pinia 状态管理（user、chat、ticket、toast）
│   ├── router/           Vue Router（/、/requirement、/ticket/:id）
│   ├── views/            页面组件（ChatView、RequirementView、TicketDetailView）
│   ├── components/       UI 组件（ChatMessage、ChatInput、TicketCard 等）
│   ├── types/            TypeScript 类型定义
│   └── assets/           CSS 设计系统
├── vite.config.ts        Vite 构建配置（@ 别名、API 代理）
└── package.json          依赖清单（Vue 3、Pinia、Vue Router、Axios）
templates/                身份/引导与提示词模板
├── SOUL.md               Agent 身份引导
├── SOUL_CS.md            客服身份引导
├── USER.md               用户偏好档案
├── agent/                Agent 系统提示词模板
└── subagents/            子代理身份模板（11 种）
Dockerfile                Docker 容器镜像
docker-compose.yml        Docker Compose 编排
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

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek 模型名 |
| `ZHIPU_API_KEY` | — | 智谱 AI API Key |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 智谱 AI API 地址 |
| `ZHIPU_MODEL` | `GLM-4.7-Flash` | 智谱 AI 模型名 |
| `DIFY_BASE_URL` | `http://127.0.0.1:80` | Dify 服务地址 |
| `DIFY_API_KEY` | — | Dify API Key |
| `SERVICE_PORT` | `8000` | 服务监听端口 |
| `SESSION_TIMEOUT_MINUTES` | `30` | 会话超时时间（分钟） |
| `MAX_SESSIONS` | `1000` | 最大并发会话数 |

## Docker 部署

```bash
docker-compose up -d
# 服务运行在 http://localhost:8000
```

## 技术栈

| 层 | 技术 |
|----|------|
| Agent 引擎 | LangGraph + LangChain Core |
| LLM | DeepSeek / 智谱 AI |
| 后端框架 | FastAPI + SSE 流式 |
| 数据存储 | SQLite (aiosqlite) |
| 向量存储 | ChromaDB |
| RAG | sentence-transformers |
| 前端 | Vue 3 + Pinia + Vue Router + Vite |

## 运行测试

```bash
pytest test/ -v
```

## 许可证

本项目采用 [MIT License](LICENSE)。

## 贡献

欢迎提交 Issue 和 Pull Request。请确保代码通过现有测试。

## Docker 部署

```bash
docker-compose up -d
# 服务运行在 http://localhost:8000
```

## 技术栈

| 层 | 技术 |
|----|------|
| Agent 引擎 | LangGraph + LangChain Core |
| LLM | DeepSeek / 智谱 AI |
| 后端框架 | FastAPI + SSE 流式 |
| 数据存储 | SQLite (aiosqlite) |
| 向量存储 | ChromaDB |
| RAG | sentence-transformers |
| 前端 | Vue 3 + Pinia + Vue Router + Vite |

## 运行测试

```bash
pytest test/ -v
```

## 许可证

本项目采用 [MIT License](LICENSE)。

## 贡献

欢迎提交 Issue 和 Pull Request。请确保代码通过现有测试。