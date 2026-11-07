"""
Offline test setup: no API key and no network are needed.

Real models are replaced with LangChain's fake models. A fake key is set only
so that config.py's "missing key" guard passes; it is never sent anywhere,
because every test that reaches a model patches it first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "05-unified-agent"))


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-offline-tests")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_DIM", raising=False)


class FakeToolModel(GenericFakeChatModel):
    """A scripted chat model that also accepts tools, so it can drive a real
    LangGraph agent loop. It returns the given messages in order and records
    every prompt it receives."""

    received: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted_model(*replies: AIMessage | str) -> FakeToolModel:
    msgs = [r if isinstance(r, AIMessage) else AIMessage(content=r) for r in replies]
    return FakeToolModel(messages=iter(msgs), received=[])


class FailingModel(GenericFakeChatModel):
    """Simulates an API outage."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise ConnectionError("simulated API outage")

    def _stream(self, *args, **kwargs):
        raise ConnectionError("simulated API outage")


@pytest.fixture
def make_pdf(tmp_path):
    """Create a small text PDF with one string per page."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    def _make(pages: list[str]) -> bytes:
        path = tmp_path / "doc.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        for text in pages:
            y = 800
            for line in text.split("\n"):
                c.drawString(50, y, line)
                y -= 15
            c.showPage()
        c.save()
        return path.read_bytes()

    return _make
