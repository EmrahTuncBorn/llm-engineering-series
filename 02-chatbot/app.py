"""
02 · Multi-turn chatbot
=======================

LLMs are stateless: the model remembers nothing between calls. "Memory" lives
in the application, which re-sends the conversation on every turn.

Run:  streamlit run 02-chatbot/app.py
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_series.chat import trim_history, visible_messages
from llm_series.config import ConfigError, get_chat_model

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("💬 Multi-turn LLM chatbot")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Reply in the language the user writes in."
)


@st.cache_resource
def load_model():
    # Created once per server process instead of on every Streamlit rerun.
    return get_chat_model()


try:
    llm = load_model()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

# Streamlit re-runs this whole script on every interaction; ordinary
# variables are reset each time. st.session_state survives reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("🗑️ New chat"):
        st.session_state.messages = []
        st.rerun()

for role, text in visible_messages(st.session_state.messages):
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # The system prompt is prepended at call time rather than stored in the
    # history, so it is never trimmed away and never rendered.
    history = trim_history(st.session_state.messages + [HumanMessage(content=user_input)])
    prompt = [SystemMessage(content=SYSTEM_PROMPT)] + history

    with st.chat_message("assistant"):
        try:
            # Stream tokens as they arrive; keep the chunks so the complete
            # message (including provider metadata) can be stored.
            chunks = []

            def token_stream():
                for chunk in llm.stream(prompt):
                    chunks.append(chunk)
                    yield chunk.text

            st.write_stream(token_stream())
            full = chunks[0]
            for chunk in chunks[1:]:
                full = full + chunk
            reply = AIMessage(content=full.content, additional_kwargs=full.additional_kwargs)
        except Exception as exc:  # network error, quota, invalid key...
            st.error(f"The model call failed: {exc}")
            st.stop()

    # Both messages are stored only after a successful reply. If the call
    # fails, the history stays consistent (no question without an answer).
    st.session_state.messages += [HumanMessage(content=user_input), reply]
