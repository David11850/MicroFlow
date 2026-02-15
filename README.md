# MicroFlow

这是一个轻量级的 **本地 Agent 网关** 示例，目标是让上下文窗口只有 `4096` 的本地模型也能稳定跑多轮对话，并支持自然语言触发终端命令。

## 已实现能力

- OpenAI 协议兼容入口：`POST /v1/chat/completions`
- 自动会话管理：支持 `conversation_id` 维持多轮上下文
- 上下文压缩：超过预算后自动把旧消息滚动摘要，避免爆上下文
- 流式转发：把下游本地模型的 SSE chunk 转发给上游客户端
- 非流式兼容：`stream=false` 时自动聚合输出
- **Terminal Skill**：通过自然语言触发命令行操作（带安全规则）

## 架构

```text
Client(OpenAI SDK)
      |
      v
MicroFlow Agent Gateway (:9000)
      |               \
      |                \--> Terminal Skill (local shell)
      v
Local LLM OpenAI API (:8000)
```

你给出的本地 OpenAI API 脚本可以作为下游模型服务（例如运行在 `0.0.0.0:8000`）。
本项目新增的 `agent/local_agent_service.py` 作为上游 Agent 层，默认连到 `http://127.0.0.1:8000`。

## 快速开始

### 1) 启动你的本地模型 OpenAI API（下游）

```bash
python your_local_llm_openai_api.py
```

### 2) 启动 Agent 网关（上游）

```bash
pip install -r requirements-agent.txt
python -m agent.local_agent_service
```

### 3) 调用测试

#### 普通问答（流式）

```bash
curl -N http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"deepseek-local",
    "conversation_id":"demo-1",
    "stream":true,
    "messages":[{"role":"user","content":"帮我规划一个一周健身计划"}]
  }'
```

#### 终端技能（自然语言）

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id":"ops-1",
    "stream":false,
    "messages":[{"role":"user","content":"列出当前目录"}]
  }'
```

也支持显式模式：

```text
执行命令: pwd
!ls -la
查看文件 README.md
```

#### 查询当前可用技能

```bash
curl http://127.0.0.1:9000/v1/skills
```

## 配置项

通过环境变量配置：

- `LOCAL_MODEL_BASE_URL`：下游本地模型地址（默认 `http://127.0.0.1:8000`）
- `LOCAL_MODEL_NAME`：模型名（默认 `deepseek-local`）
- `AGENT_SYSTEM_PROMPT`：系统提示词
- `AGENT_WORKDIR`：终端技能执行目录（默认当前工作目录）

示例：

```bash
export LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000
export LOCAL_MODEL_NAME=DeepSeek_R1_Distill_Qwen_1.5B_4096
export AGENT_WORKDIR=/workspace/MicroFlow
python -m agent.local_agent_service
```

## 上下文策略说明

`agent/context_window.py` 中实现了 `ContextWindowManager`：

- 总预算默认 `4096` token
- 预留默认 `512` token 给模型输出
- 超预算时，最旧消息会被压缩进 `summary`
- 后续请求会自动把 `summary` 作为 system 消息附加

## Terminal Skill 安全策略

在 `agent/terminal_skill.py` 里：

- 仅允许执行白名单命令（如 `pwd`, `ls`, `cat`, `rg`, `pytest`, `git`）
- 拦截危险模式（如 `rm -rf`, `shutdown`, `mkfs`, `dd`）
- 默认超时 30 秒，输出截断防止响应过大

这样你可以通过自然语言发起命令行任务，同时尽量降低误操作风险。
