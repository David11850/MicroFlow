from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent.context_window import ContextWindowManager
from agent.terminal_skill import TerminalSkill, format_terminal_reply

app = FastAPI(title="MicroFlow Agent Gateway", version="0.1.0")

MODEL_BASE_URL = os.getenv("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "deepseek-local")
SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "你是 MicroFlow Agent，一个可以调用本地模型的中文助手。回答要简洁，必要时分步骤。",
)

context_manager = ContextWindowManager(max_tokens=4096, reserve_tokens=512)
terminal_skill = TerminalSkill()


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return ""


async def _stream_from_local_model(payload: Dict[str, Any]) -> AsyncIterator[str]:
    url = f"{MODEL_BASE_URL.rstrip('/')}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                text = await response.aread()
                raise HTTPException(status_code=response.status_code, detail=text.decode("utf-8", errors="ignore"))
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    yield line + "\n\n"


def _chunk_text(text: str, size: int = 120) -> List[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _stream_text_response(text: str, conversation_id: str) -> AsyncIterator[str]:
    async def _gen() -> AsyncIterator[str]:
        for chunk in _chunk_text(text):
            payload = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "conversation_id": conversation_id,
                "choices": [{"index": 0, "delta": {"content": chunk}}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        done_payload = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "conversation_id": conversation_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return _gen()


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "model_base_url": MODEL_BASE_URL}


@app.get("/v1/skills")
async def list_skills() -> Dict[str, Any]:
    return {
        "skills": [
            {
                "name": "terminal",
                "description": "通过自然语言触发命令行执行（带安全限制）",
                "trigger_examples": ["执行命令: pwd", "列出当前目录", "查看文件 README.md"],
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream", True))
    conversation_id = body.get("conversation_id") or f"conv-{uuid.uuid4().hex[:8]}"
    incoming_messages = body.get("messages", [])
    user_prompt = _extract_last_user_message(incoming_messages)

    if not user_prompt:
        raise HTTPException(status_code=400, detail="messages 里必须包含用户输入")

    context_manager.append_user_message(conversation_id, user_prompt)

    terminal_command = terminal_skill.extract_command_from_text(user_prompt)
    if terminal_command:
        terminal_result = terminal_skill.execute(terminal_command)
        assistant_reply = format_terminal_reply(terminal_result)
        context_manager.append_assistant_message(conversation_id, assistant_reply)

        if stream:
            return StreamingResponse(_stream_text_response(assistant_reply, conversation_id), media_type="text/event-stream")

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL_NAME),
            "conversation_id": conversation_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": assistant_reply}, "finish_reason": "stop"}],
        }
        return JSONResponse(response)

    upstream_messages = context_manager.build_prompt_messages(conversation_id, SYSTEM_PROMPT)

    upstream_payload = {
        "model": body.get("model", MODEL_NAME),
        "messages": upstream_messages,
        "temperature": body.get("temperature", 0.4),
        "stream": True,
    }

    async def proxy_stream() -> AsyncIterator[str]:
        collected = []
        async for line in _stream_from_local_model(upstream_payload):
            if line.strip() == "data: [DONE]":
                break
            try:
                payload = json.loads(line.removeprefix("data: "))
                delta = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    collected.append(delta)
            except json.JSONDecodeError:
                pass
            yield line

        assistant_reply = "".join(collected).strip()
        if assistant_reply:
            context_manager.append_assistant_message(conversation_id, assistant_reply)

        done_payload = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "conversation_id": conversation_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return StreamingResponse(proxy_stream(), media_type="text/event-stream")

    aggregated = []
    async for line in proxy_stream():
        if not line.startswith("data: "):
            continue
        raw = line.removeprefix("data: ").strip()
        if raw == "[DONE]":
            break
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        delta = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta:
            aggregated.append(delta)

    content = "".join(aggregated)
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL_NAME),
        "conversation_id": conversation_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent.local_agent_service:app", host="0.0.0.0", port=9000, reload=False)
