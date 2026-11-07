"""
Smoke test against the REAL Gemini API
======================================

The offline suite (pytest) checks the logic with fake models. This script
checks what only the real service can confirm:

    1. the configured model names exist
    2. chat and streaming work
    3. embeddings: dimension, unit norm, semantic ordering, query/document asymmetry
    4. Matryoshka truncation is re-normalized
    5. multi-turn tool calling (thought signatures survive the history)
    6. analyze_data's enum schema is accepted and used
    7. agentic RAG: document_search with real embeddings
    8. optional: Tavily web search

Run from the repository root (needs GOOGLE_API_KEY in .env):

    python scripts/smoke_test.py

It makes roughly 15 API calls, well inside the free tier. The key is never
printed. Paste the report it prints when asking for help.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "05-unified-agent"))

RESULTS: list[tuple[str, str, str]] = []
PAUSE_SECONDS = 2  # be gentle with free-tier rate limits


def record(name: str, status: str, note: str = "") -> None:
    RESULTS.append((name, status, note))
    print(f"[{status}] {name}" + (f" — {note}" if note else ""), flush=True)


def check(name: str):
    """Decorator: run a check, record PASS/WARN/FAIL, never stop the script."""

    def wrap(fn):
        def run():
            try:
                outcome = fn()
                status, note = outcome if isinstance(outcome, tuple) else ("PASS", outcome or "")
                record(name, status, note)
            except Exception as exc:  # report and continue with the next check
                detail = f"{type(exc).__name__}: {exc}".replace("\n", " ")
                record(name, "FAIL", detail[:400])
                if os.getenv("SMOKE_DEBUG"):
                    traceback.print_exc()
            time.sleep(PAUSE_SECONDS)

        return run

    return wrap


def cos(u, v) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))


def norm(u) -> float:
    return math.sqrt(sum(a * a for a in u))


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
from llm_series import config  # noqa: E402  (loads .env)

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", config.DEFAULT_CHAT_MODEL)
EMBED_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", config.DEFAULT_EMBEDDING_MODEL)


@check("0. API key and settings")
def check_env():
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")
    dim = os.getenv("GEMINI_EMBEDDING_DIM") or "3072 (default)"
    return f"chat={CHAT_MODEL}, embedding={EMBED_MODEL}, dim={dim}, tavily={'yes' if config.tavily_available() else 'no'}"


@check("1. Model names exist")
def check_models():
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    names = {m.name.removeprefix("models/") for m in client.models.list()}
    missing = [m for m in (CHAT_MODEL, EMBED_MODEL) if m.removeprefix("models/") not in names]
    if missing:
        flash = sorted(n for n in names if "flash" in n and "tts" not in n and "image" not in n)[-8:]
        embeds = sorted(n for n in names if "embedding" in n)
        raise RuntimeError(
            f"Not available for this key: {missing}. Some available flash models: {flash}. "
            f"Embedding models: {embeds}. Set GEMINI_CHAT_MODEL / GEMINI_EMBEDDING_MODEL in .env."
        )
    return f"{len(names)} models visible"


@check("2. Chat (invoke + stream)")
def check_chat():
    llm = config.get_chat_model()
    reply = llm.invoke("Reply with the single word: pong").text.strip()
    if not reply:
        raise RuntimeError("empty reply")
    chunks = [c.text for c in llm.stream("Count from 1 to 5, separated by spaces.")]
    streamed = "".join(chunks)
    if "3" not in streamed:
        return "WARN", f"invoke ok ({reply!r}); unexpected stream: {streamed[:80]!r}"
    return f"invoke → {reply!r}; stream → {len(chunks)} chunks"


@check("3. Embeddings: size, norm, semantics, asymmetry")
def check_embeddings():
    sys.path.insert(0, str(ROOT / "01-embeddings"))
    from embeddings_demo import SENTENCES

    emb = config.get_embeddings()
    vecs = emb.embed_documents(SENTENCES, task_type="SEMANTIC_SIMILARITY")
    dim = len(vecs[0])
    norms = [norm(v) for v in vecs]
    dogs_puppies, dogs_cats, dogs_stock = cos(vecs[0], vecs[2]), cos(vecs[0], vecs[1]), cos(vecs[0], vecs[3])
    notes = f"dim={dim}, norms≈{min(norms):.3f}-{max(norms):.3f}, dogs~puppies={dogs_puppies:.3f}, dogs~cats={dogs_cats:.3f}, dogs~stock={dogs_stock:.3f}"
    if not all(abs(n - 1) < 1e-2 for n in norms):
        return "WARN", "vectors are not unit length; " + notes
    if not dogs_puppies > dogs_stock:
        return "FAIL", "semantic ordering is wrong; " + notes

    # Asymmetric retrieval: a question should land nearest the right document.
    doc_vecs = emb.embed_documents(SENTENCES)              # RETRIEVAL_DOCUMENT
    q = emb.embed_query("Which animal is a loyal companion?")  # RETRIEVAL_QUERY
    best = max(range(len(SENTENCES)), key=lambda i: cos(q, doc_vecs[i]))
    if best not in (0, 2):
        return "WARN", f"query matched sentence [{best}]; " + notes
    return notes + f", query→[{best}]"


@check("4. Matryoshka truncation (768) is re-normalized")
def check_truncation():
    emb = config.RetrievalEmbeddings(model=EMBED_MODEL, output_dimensionality=768)
    v = emb.embed_documents(["Geodesics on a sphere are great circles."])[0]
    if len(v) != 768:
        raise RuntimeError(f"expected 768 dimensions, got {len(v)}")
    if abs(norm(v) - 1) > 1e-6:
        raise RuntimeError(f"norm is {norm(v):.4f}, expected 1")
    return "dim=768, norm=1.000"


def agent_with(dataframe=None, vectorstore=None):
    from langchain.agents import create_agent

    from agent_tools import build_tools

    return create_agent(
        config.get_chat_model(),
        build_tools(vectorstore, dataframe),
        system_prompt="Use the tools when they help. Use the calculator for arithmetic.",
    )


def calls_in(messages):
    from llm_series.chat import tool_calls_in

    return tool_calls_in(messages)


@check("5. Multi-turn tool calling (thought signatures)")
def check_tool_calling():
    from langchain_core.messages import HumanMessage

    agent = agent_with()
    history = [HumanMessage("Use the calculator: what is 37 * 91 + 2^5?")]
    first = agent.invoke({"messages": history}, config={"recursion_limit": 10})["messages"]
    turn1_calls = [c["name"] for c in calls_in(first[len(history):])]
    if "calculator" not in turn1_calls:
        return "WARN", f"turn 1 answered without the calculator: {first[-1].text[:100]!r}"
    if "3399" not in first[-1].text.replace(",", ""):
        return "WARN", f"turn 1 answer does not contain 3399: {first[-1].text[:100]!r}"

    # Turn 2 re-sends turn 1 unmodified, including the tool call and result.
    history = first + [HumanMessage("Now divide that result by 3 with the calculator.")]
    second = agent.invoke({"messages": history}, config={"recursion_limit": 10})["messages"]
    answer = second[-1].text.replace(",", "")
    if "1133" not in answer:
        return "WARN", f"turn 2 answer does not contain 1133: {second[-1].text[:100]!r}"
    return f"turn 1 tools={turn1_calls}, turn 2 tools={[c['name'] for c in calls_in(second[len(history):])]}"


@check("6. analyze_data (enum schema) via the agent")
def check_analyze_data():
    import pandas as pd
    from langchain_core.messages import HumanMessage

    df = pd.DataFrame({"city": ["Izmir", "Manisa", "Izmir", "Aydin"], "temp": [21.5, 19.0, 23.5, 22.0]})
    agent = agent_with(dataframe=df)
    q = [HumanMessage("Using the uploaded dataset, what is the average temp for Izmir?")]
    msgs = agent.invoke({"messages": q}, config={"recursion_limit": 10})["messages"]
    calls = calls_in(msgs[1:])
    used = [c for c in calls if c["name"] == "analyze_data"]
    if not used:
        return "WARN", f"analyze_data not used: {msgs[-1].text[:100]!r}"
    if "22.5" not in msgs[-1].text:
        return "WARN", f"calls={[c['args'] for c in used]}, answer lacks 22.5: {msgs[-1].text[:100]!r}"
    return f"calls={[c['args'] for c in used]}"


@check("7. Agentic RAG: document_search with real embeddings")
def check_rag():
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage

    from llm_series.documents import index_documents

    docs = [
        Document("The Darboux frame of a curve on a surface is {T, V, N}.", metadata={"page": 0}),
        Document("A Tzitzeica curve satisfies tau / d^2 = constant, where d is the distance "
                 "from the origin to the osculating plane.", metadata={"page": 1}),
        Document("Our office plants are watered every Tuesday.", metadata={"page": 2}),
    ]
    store = index_documents(docs, config.get_embeddings())
    try:
        agent = agent_with(vectorstore=store)
        q = [HumanMessage("According to the uploaded document, what condition does a Tzitzeica curve satisfy?")]
        msgs = agent.invoke({"messages": q}, config={"recursion_limit": 10})["messages"]
    finally:
        store.delete_collection()
    used = [c for c in calls_in(msgs[1:]) if c["name"] == "document_search"]
    if not used:
        return "WARN", f"document_search not used: {msgs[-1].text[:100]!r}"
    if "tau" not in msgs[-1].text.lower() and "τ" not in msgs[-1].text:
        return "WARN", f"answer may be ungrounded: {msgs[-1].text[:120]!r}"
    return f"query={used[0]['args'].get('query')!r}"


@check("8. Tavily web search (optional)")
def check_tavily():
    if not config.tavily_available():
        return "SKIP", "TAVILY_API_KEY not set"
    from langchain_tavily import TavilySearch

    out = TavilySearch(max_results=1).invoke({"query": "Gemini API"})
    return f"{len(str(out))} characters returned"


def main() -> None:
    print(f"Python {platform.python_version()} on {platform.system()}\n")
    for fn in (check_env, check_models, check_chat, check_embeddings, check_truncation,
               check_tool_calling, check_analyze_data, check_rag, check_tavily):
        fn()
        if RESULTS[-1][0].startswith("0.") and RESULTS[-1][1] == "FAIL":
            break  # nothing else can work without a key

    counts = {s: sum(1 for _, st, _ in RESULTS if st == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
    print("\n" + "=" * 70)
    print("SUMMARY  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("=" * 70)
    print("WARN = it ran, but the model behaved unexpectedly (often fine, worth a look).")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
