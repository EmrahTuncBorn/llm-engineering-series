"""
04 · Tool-calling agent
=======================

The model gets three tools (calculator, web search, Wikipedia) and decides on
its own which one to call and with which arguments. `create_agent` builds a
ReAct loop as a LangGraph graph:

    user ─► model ──(tool calls?)── yes ─► ToolNode runs the tool ─► ToolMessage ─┐
              ▲                                                                   │
              └───────────────────────────────────────────────────────────────────┘
                          └── no ─► final answer

The model never executes anything. It emits a structured request (tool name,
arguments, id) that matches the tool's JSON schema; the application runs the
tool and returns the result as a ToolMessage with the same id.

Run:  streamlit run 04-tool-calling/app.py
"""

from __future__ import annotations

import streamlit as st
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from llm_series.chat import tool_calls_in, trim_history, visible_messages
from llm_series.config import ConfigError, get_chat_model, tavily_available
from llm_series.safe_math import calculate

st.set_page_config(page_title="Tool-calling agent", page_icon="🛠️")
st.title("🛠️ Tool-calling agent")

# Upper bound on model/tool steps per question, so a model that keeps calling
# tools cannot loop forever (LangGraph raises GraphRecursionError beyond it).
RECURSION_LIMIT = 15

# This project deliberately uses a *directive* prompt. Project 05 replaces it
# with a neutral one and leaves tool choice entirely to the model; comparing
# the two is part of the series.
SYSTEM_PROMPT = """You are a helpful assistant. Follow these rules:
- For any calculation, use the 'calculator' tool instead of computing in your head.
- For encyclopedic facts (people, events, dates, places), verify with the 'wikipedia' tool.
- For current events or news, use the web search tool.
Reply in the language the user writes in."""


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Supports + - * / // % and ** (or ^) for powers, parentheses, the constants
    pi and e, and the functions sqrt, log, log10, exp, sin, cos, tan, abs,
    round, floor, ceil. Example: '2 + 2 * 5' or 'sqrt(2) ^ 4'."""
    return calculate(expression)


def build_tools() -> list:
    # The docstring of a tool is effectively a prompt: it is the only thing the
    # model reads when deciding whether and how to call the tool.
    tools = [
        calculator,
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)),
    ]
    if tavily_available():
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=2))
    return tools


@st.cache_resource
def load_agent():
    return create_agent(get_chat_model(), build_tools(), system_prompt=SYSTEM_PROMPT)


try:
    agent = load_agent()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("**Tools**")
    st.markdown("- calculator\n- wikipedia")
    if tavily_available():
        st.markdown("- web search (Tavily)")
    else:
        st.caption("Web search is off: add TAVILY_API_KEY to .env to enable it.")
    if st.button("🗑️ New chat"):
        st.session_state.messages = []
        st.rerun()

for role, text in visible_messages(st.session_state.messages):
    with st.chat_message(role):
        st.markdown(text)

user_input = st.chat_input("Ask anything (math, encyclopedic facts, current news)")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    history = trim_history(st.session_state.messages + [HumanMessage(content=user_input)])

    with st.chat_message("assistant"):
        try:
            with st.status("Thinking...", expanded=False) as status:
                result = agent.invoke({"messages": history}, config={"recursion_limit": RECURSION_LIMIT})
                new_messages = result["messages"][len(history):]
                calls = tool_calls_in(new_messages)
                for call in calls:
                    st.markdown(f"🔧 **{call['name']}** — `{call['args']}`")
                status.update(label=f"Used {len(calls)} tool call(s)" if calls else "Answered without tools")
        except Exception as exc:
            st.error(f"The agent failed: {exc}")
            st.stop()

        st.markdown(result["messages"][-1].text)

    # The full message list is kept, including tool calls and tool results:
    # follow-up questions ("multiply the number you found by 2") can refer to
    # earlier results, and Gemini expects its own messages back unmodified
    # (they carry thought signatures needed for multi-turn tool calling).
    st.session_state.messages = st.session_state.messages + [HumanMessage(content=user_input)] + new_messages
