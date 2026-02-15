from agent.context_window import ContextWindowManager


def test_context_trim_and_summary():
    manager = ContextWindowManager(max_tokens=60, reserve_tokens=10)
    conversation_id = "demo"

    manager.append_user_message(conversation_id, "你好" * 30)
    manager.append_assistant_message(conversation_id, "收到" * 30)
    manager.append_user_message(conversation_id, "请给我总结" * 20)

    state = manager.get_state(conversation_id)
    assert state.summary
    assert len(state.messages) <= 2


def test_prompt_contains_summary_system_prompt():
    manager = ContextWindowManager(max_tokens=40, reserve_tokens=10)
    conversation_id = "demo2"

    for _ in range(3):
        manager.append_user_message(conversation_id, "A" * 40)

    prompt = manager.build_prompt_messages(conversation_id, "system")
    roles = [item["role"] for item in prompt]
    assert roles[0] == "system"
    assert "历史摘要" in prompt[1]["content"]
