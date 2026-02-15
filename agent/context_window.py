from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


Message = Dict[str, str]


@dataclass
class ConversationState:
    summary: str = ""
    messages: List[Message] = field(default_factory=list)


class ContextWindowManager:
    """Keep chat history under a token budget with rolling summary."""

    def __init__(self, max_tokens: int = 4096, reserve_tokens: int = 512):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self._store: Dict[str, ConversationState] = {}

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Rough estimate for Chinese/English mixed text.
        return max(1, len(text) // 2)

    def get_state(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationState()
        return self._store[conversation_id]

    def append_user_message(self, conversation_id: str, content: str) -> ConversationState:
        state = self.get_state(conversation_id)
        state.messages.append({"role": "user", "content": content})
        self._trim_state(state)
        return state

    def append_assistant_message(self, conversation_id: str, content: str) -> ConversationState:
        state = self.get_state(conversation_id)
        state.messages.append({"role": "assistant", "content": content})
        self._trim_state(state)
        return state

    def build_prompt_messages(self, conversation_id: str, system_prompt: str) -> List[Message]:
        state = self.get_state(conversation_id)
        output: List[Message] = [{"role": "system", "content": system_prompt}]
        if state.summary:
            output.append(
                {
                    "role": "system",
                    "content": f"历史摘要（由系统自动压缩）: {state.summary}",
                }
            )
        output.extend(state.messages)
        return output

    def _trim_state(self, state: ConversationState) -> None:
        budget = self.max_tokens - self.reserve_tokens
        while self._state_tokens(state) > budget and len(state.messages) > 2:
            dropped = state.messages.pop(0)
            state.summary = self._merge_summary(state.summary, dropped)

    def _state_tokens(self, state: ConversationState) -> int:
        payload = state.summary + "\n" + "\n".join(msg["content"] for msg in state.messages)
        return self.estimate_tokens(payload)

    @staticmethod
    def _merge_summary(existing: str, dropped: Message) -> str:
        prefix = f"[{dropped['role']}]"
        clipped = dropped["content"].strip().replace("\n", " ")[:120]
        merged = (existing + " " + prefix + clipped).strip()
        return merged[-800:]
