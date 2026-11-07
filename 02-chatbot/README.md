# 02 · Multi-turn Chatbot

A Streamlit chat app on top of Gemini. The point of the project is one fact
about LLMs: **they are stateless**. The model remembers nothing between
calls; the application re-sends the conversation on every turn.

```bash
streamlit run 02-chatbot/app.py
```

## How it works

```
user message ─► [system prompt + recent history + message] ─► model.stream()
                                                                   │
        render tokens as they arrive ◄─────────────────────────────┘
        store question + answer in st.session_state (only on success)
```

Streamlit re-runs the whole script on every interaction, which resets
ordinary variables. `st.session_state` survives reruns and holds the
history as LangChain `HumanMessage` / `AIMessage` objects.

## Design decisions

| Decision | Why |
|----------|-----|
| Sliding window of the last 20 messages | Without a limit the context window eventually overflows, and cost grows quadratically: turn *n* re-sends all previous turns. |
| Window always starts at a user message | The model never receives an answer without its question. |
| System prompt added at call time, not stored | It can never be trimmed away or rendered as a chat bubble. |
| Streaming (`llm.stream` + `st.write_stream`) | The user sees the answer forming instead of a blank screen. |
| Question and answer stored together, after success | If the API fails, the history stays consistent. The original version stored the question first and left a dangling message on error. |
| Model created once (`@st.cache_resource`) | Not rebuilt on every rerun. |
| Temperature not set | Google recommends default sampling settings for Gemini 3 models. |

## Known limitations

- History lives in memory and disappears on page reload; persistence would
  need a database.
- Old messages are dropped, not summarized. A summary memory would keep
  more context in the same budget.
