"""
End-to-end tests of the Streamlit apps with scripted fake models.

Streamlit's AppTest runs each script exactly as `streamlit run` would, in the
same process, so the model factories in llm_series.config can be patched.
Projects 04 and 05 run the real LangGraph agent loop, including the
ToolNode thread pool, with a fake model that requests tool calls.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import streamlit as st
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from conftest import FailingModel, scripted_model
from llm_series.documents import build_vectorstore

ROOT = Path(__file__).resolve().parent.parent
FAKE_EMB = DeterministicFakeEmbedding(size=64)
TIMEOUT = 30


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def app(path: str) -> AppTest:
    return AppTest.from_file(str(ROOT / path), default_timeout=TIMEOUT)


def patch_models(chat_model):
    return (
        mock.patch("llm_series.config.get_chat_model", return_value=chat_model),
        mock.patch("llm_series.config.get_embeddings", return_value=FAKE_EMB),
    )


def call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


# ----------------------------------------------------------------------
# 01 · embeddings
# ----------------------------------------------------------------------
def test_01_embeddings_demo_runs(capsys):
    spec = importlib.util.spec_from_file_location("embeddings_demo", ROOT / "01-embeddings/embeddings_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class TaskTypeFakeEmbedding(DeterministicFakeEmbedding):
        # Like Gemini's embeddings, accept (and record) a task_type argument.
        def embed_documents(self, texts, task_type=None, **kwargs):
            seen.append(task_type)
            return super().embed_documents(texts)

    seen = []
    with mock.patch.object(module, "get_embeddings", return_value=TaskTypeFakeEmbedding(size=64)):
        module.main()
    assert seen == ["SEMANTIC_SIMILARITY"]
    out = capsys.readouterr().out
    assert "Dimension         : 64" in out
    assert "[0] vs [3]" in out and "Similarity matrix" in out

    assert module.cosine_similarity([1, 0], [0, 0]) == 0.0           # zero vector guard
    assert module.cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    matrix = module.similarity_matrix([[1, 0], [0, 3], [1, 1]])
    assert matrix[0, 1] == pytest.approx(0.0) and matrix[1, 1] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 02 · chatbot
# ----------------------------------------------------------------------
def test_02_missing_key_shows_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    at = app("02-chatbot/app.py").run()
    assert not at.exception
    assert "GOOGLE_API_KEY" in at.error[0].value


def test_02_multi_turn_history_is_sent_to_the_model():
    model = scripted_model("Hello Emrah!", "You said your name is Emrah.")
    p1, p2 = patch_models(model)
    with p1, p2:
        at = app("02-chatbot/app.py").run()
        at.chat_input[0].set_value("Hi, I am Emrah").run()
        at.chat_input[0].set_value("What is my name?").run()

    assert not at.exception
    assert len(at.session_state.messages) == 4
    second_prompt = model.received[-1]
    assert isinstance(second_prompt[0], SystemMessage)
    assert [m.text for m in second_prompt[1:]] == [
        "Hi, I am Emrah", "Hello Emrah!", "What is my name?",
    ]


def test_02_api_failure_keeps_history_consistent():
    p1, p2 = patch_models(FailingModel(messages=iter([])))
    with p1, p2:
        at = app("02-chatbot/app.py").run()
        at.chat_input[0].set_value("Hello").run()
    assert "simulated API outage" in at.error[0].value
    assert at.session_state.messages == []   # no question without an answer


# ----------------------------------------------------------------------
# 03 · RAG
# ----------------------------------------------------------------------
def test_03_rag_answers_with_sources_and_rewrites_follow_ups(make_pdf):
    store, _ = build_vectorstore(
        make_pdf(["Darboux frame: T, V, N along a curve on a surface.", "Geodesic curvature is k_g."]),
        FAKE_EMB,
    )
    model = scripted_model(
        "The Darboux frame is {T, V, N} (p. 1).",
        "What is the geodesic curvature in the Darboux frame?",   # rewritten query
        "It is k_g (p. 2).",
    )
    p1, p2 = patch_models(model)
    with p1, p2:
        at = app("03-rag/app.py")
        at.session_state["vectorstore"] = store
        at.session_state["doc_name"] = "notes.pdf"
        at.run()
        at.chat_input[0].set_value("What is the Darboux frame?").run()
        at.chat_input[0].set_value("And its geodesic curvature?").run()

    assert not at.exception
    # First question: no history, so no rewrite call; context is in the prompt.
    first_answer_prompt = model.received[0]
    assert "<context>" in first_answer_prompt[-1].text and "[page" in first_answer_prompt[-1].text
    # Follow-up: one rewrite call, then the answer call with history.
    assert len(model.received) == 3
    assert any("Search query" in c.value for c in at.caption)
    assert [m.text for m in at.session_state.messages][-1] == "It is k_g (p. 2)."
    assert len(at.expander) >= 1


def test_03_question_without_document_warns():
    p1, p2 = patch_models(scripted_model("unused"))
    with p1, p2:
        at = app("03-rag/app.py").run()
        at.chat_input[0].set_value("Hello?").run()
    assert "Upload a document first" in at.warning[0].value
    assert at.session_state.messages == []


# ----------------------------------------------------------------------
# 04 · tool calling
# ----------------------------------------------------------------------
def test_04_agent_calls_calculator_and_keeps_tool_results():
    model = scripted_model(call("calculator", {"expression": "2^10"}), "2^10 is 1024.")
    p1, p2 = patch_models(model)
    with p1, p2:
        at = app("04-tool-calling/app.py").run()
        at.chat_input[0].set_value("What is 2^10?").run()

    assert not at.exception
    msgs = at.session_state.messages
    tool_results = [m for m in msgs if isinstance(m, ToolMessage)]
    assert tool_results and tool_results[0].content == "1024"
    assert isinstance(msgs[0], HumanMessage) and msgs[-1].text == "2^10 is 1024."
    assert any("2^10 is 1024." in m.value for m in at.markdown)


def test_04_agent_failure_keeps_history_consistent():
    p1, p2 = patch_models(FailingModel(messages=iter([])))
    with p1, p2:
        at = app("04-tool-calling/app.py").run()
        at.chat_input[0].set_value("Hello").run()
    assert "simulated API outage" in at.error[0].value
    assert at.session_state.messages == []


# ----------------------------------------------------------------------
# 05 · unified agent
# ----------------------------------------------------------------------
def test_05_analyze_data_runs_in_worker_thread_via_snapshot():
    """The thread-safety story: the tool runs inside LangGraph's ToolNode
    (a worker thread) and must still see the uploaded DataFrame."""
    df = pd.DataFrame({"city": ["Izmir", "Manisa"], "temp": [21.0, 19.0]})
    model = scripted_model(call("analyze_data", {"operation": "mean", "column": "temp"}), "The mean is 20.")
    p1, p2 = patch_models(model)
    with p1, p2:
        at = app("05-unified-agent/app.py")
        at.session_state["dataframe"] = df
        at.session_state["csv_name"] = "weather.csv"
        at.run()
        at.chat_input[0].set_value("Average temperature?").run()

    assert not at.exception
    tool_results = [m for m in at.session_state.messages if isinstance(m, ToolMessage)]
    assert tool_results[0].content == "20.0"


def test_05_document_search_tool_reads_uploaded_pdf(make_pdf):
    store, _ = build_vectorstore(make_pdf(["Tzitzeica curves satisfy tau / d^2 = const."]), FAKE_EMB)
    model = scripted_model(call("document_search", {"query": "Tzitzeica condition"}), "See page 1.")
    p1, p2 = patch_models(model)
    with p1, p2:
        at = app("05-unified-agent/app.py")
        at.session_state["vectorstore"] = store
        at.run()
        at.chat_input[0].set_value("What is the Tzitzeica condition?").run()

    assert not at.exception
    tool_results = [m for m in at.session_state.messages if isinstance(m, ToolMessage)]
    assert "Tzitzeica" in tool_results[0].content and "[page 1]" in tool_results[0].content
