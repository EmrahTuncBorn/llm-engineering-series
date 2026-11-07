from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm_series.chat import tool_calls_in, trim_history, visible_messages


def conversation(turns: int):
    msgs = []
    for i in range(turns):
        msgs += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
    return msgs


def test_short_history_is_unchanged():
    msgs = conversation(3)
    assert trim_history(msgs, limit=20) == msgs


def test_long_history_keeps_recent_messages_and_starts_with_user():
    trimmed = trim_history(conversation(30), limit=7)
    assert len(trimmed) <= 7
    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[-1].content == "a29"


def test_window_never_starts_with_a_dangling_tool_result():
    msgs = [
        HumanMessage(content="q0"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "1"}]),
        ToolMessage(content="2", tool_call_id="1"),
        AIMessage(content="It is 2."),
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
    ]
    trimmed = trim_history(msgs, limit=4)
    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[0].content == "q1"


def test_visible_messages_hide_tool_plumbing():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}]),
        ToolMessage(content="2", tool_call_id="1"),
        AIMessage(content=[{"type": "text", "text": "Answer"}]),  # Gemini-style content blocks
    ]
    assert list(visible_messages(msgs)) == [("user", "hi"), ("assistant", "Answer")]
    assert [c["name"] for c in tool_calls_in(msgs)] == ["calculator"]
