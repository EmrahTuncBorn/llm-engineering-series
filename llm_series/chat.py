"""
Helpers shared by the Streamlit chat apps (02-05)
=================================================
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# How many past messages are sent to the model. LLMs are stateless, so the
# whole history is re-sent on every turn: without a limit the context window
# eventually overflows and the cost grows quadratically with the length of
# the conversation (turn n re-sends all n-1 previous turns).
MAX_HISTORY_MESSAGES = 20


def trim_history(messages: list[BaseMessage], limit: int = MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
    """Keep the most recent messages (sliding window).

    The window always starts at a user message, so the model never receives
    a dangling tool result or an assistant turn without its question.
    """
    if len(messages) <= limit:
        return list(messages)
    window = messages[-limit:]
    for i, msg in enumerate(window):
        if isinstance(msg, HumanMessage):
            return window[i:]
    return window[-1:]


def visible_messages(messages: list[BaseMessage]):
    """Yield (role, text) pairs worth rendering.

    Tool results and assistant messages that only contain tool calls (no text)
    are part of the history the model needs, but not of what the user sees.
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            yield "user", msg.text
        elif isinstance(msg, AIMessage) and msg.text.strip():
            yield "assistant", msg.text
        elif isinstance(msg, ToolMessage):
            continue


def tool_calls_in(messages: list[BaseMessage]) -> list[dict]:
    """All tool calls made in a list of messages, in order."""
    return [call for msg in messages if isinstance(msg, AIMessage) for call in msg.tool_calls]
