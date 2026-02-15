# MicroFlow

这是一个轻量级的 **本地 Agent 网关** 示例，目标是让上下文窗口只有 `4096` 的本地模型也能稳定跑多轮对话。

## 已实现能力

- OpenAI 协议兼容入口：`POST /v1/chat/completions`
- 自动会话管理：支持 `conversation_id` 维持多轮上下文
- 上下文压缩：超过预算后自动把旧消息滚动摘要，避免爆上下文
- 流式转发：把下游本地模型的 SSE chunk 转发给上游客户端
- 非流式兼容：`stream=false` 时自动聚合输出

## 架构

```text
Client(OpenAI SDK)
      |
      v
MicroFlow Agent Gateway (:9000)
      |
      v
Local LLM OpenAI API (:8000)
```

你给出的本地 OpenAI API 脚本可以作为下游模型服务（例如运行在 `0.0.0.0:8000`）。
本项目新增的 `agent/local_agent_service.py` 作为上游 Agent 层，默认连到 `http://127.0.0.1:8000`。

## 快速开始

### 1) 启动你的本地模型 OpenAI API（下游）

你已有脚本可直接启动：

```bash
python your_local_llm_openai_api.py
```

### 2) 启动 Agent 网关（上游）

```bash
pip install -r requirements-agent.txt
python -m agent.local_agent_service
```

### 3) 调用测试

#### 流式

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

#### 非流式

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"deepseek-local",
    "conversation_id":"demo-1",
    "stream":false,
    "messages":[{"role":"user","content":"继续，把饮食也补充上"}]
  }'
```

## 配置项

通过环境变量配置：

- `LOCAL_MODEL_BASE_URL`：下游本地模型地址（默认 `http://127.0.0.1:8000`）
- `LOCAL_MODEL_NAME`：模型名（默认 `deepseek-local`）
- `AGENT_SYSTEM_PROMPT`：系统提示词

示例：

```bash
export LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000
export LOCAL_MODEL_NAME=DeepSeek_R1_Distill_Qwen_1.5B_4096
python -m agent.local_agent_service
```

## 上下文策略说明

`agent/context_window.py` 中实现了 `ContextWindowManager`：

- 总预算默认 `4096` token
- 预留默认 `512` token 给模型输出
- 超预算时，最旧消息会被压缩进 `summary`
- 后续请求会自动把 `summary` 作为 system 消息附加

这套机制适合你当前这种 **模型上下文较小** 的部署。
