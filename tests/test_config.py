from unittest import mock

import numpy as np
import pytest

from llm_series import config


def test_missing_key_gives_a_readable_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    with pytest.raises(config.ConfigError, match="GOOGLE_API_KEY"):
        config.get_chat_model()


def test_gemini_api_key_is_accepted_as_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    config.get_chat_model()


def test_chat_model_reads_name_from_env_and_leaves_temperature_unset(monkeypatch):
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-test-model")
    llm = config.get_chat_model()
    assert llm.model.endswith("gemini-test-model")
    assert "temperature" not in llm.model_fields_set


@pytest.mark.parametrize("raw", ["abc", "64", "4096"])
def test_invalid_embedding_dim(monkeypatch, raw):
    monkeypatch.setenv("GEMINI_EMBEDDING_DIM", raw)
    with pytest.raises(config.ConfigError):
        config.get_embeddings()


def _patched(emb, doc_vectors, query_vector, seen):
    def fake_docs(self, texts, **kw):
        seen["documents"] = kw
        return doc_vectors

    def fake_query(self, text, **kw):
        seen["query"] = kw
        return query_vector

    base = config.GoogleGenerativeAIEmbeddings
    return mock.patch.object(base, "embed_documents", fake_docs), mock.patch.object(base, "embed_query", fake_query)


def test_task_types_are_asymmetric_and_overridable():
    emb = config.get_embeddings()
    seen = {}
    p1, p2 = _patched(emb, [[1.0, 0.0]], [0.0, 1.0], seen)
    with p1, p2:
        emb.embed_documents(["doc"])
        emb.embed_query("question")
        assert seen["documents"]["task_type"] == "RETRIEVAL_DOCUMENT"
        assert seen["query"]["task_type"] == "RETRIEVAL_QUERY"
        emb.embed_documents(["a"], task_type="SEMANTIC_SIMILARITY")
        assert seen["documents"]["task_type"] == "SEMANTIC_SIMILARITY"


def test_reduced_dimension_vectors_are_normalized(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_DIM", "768")
    emb = config.get_embeddings()
    p1, p2 = _patched(emb, [[3.0, 4.0], [0.0, 0.0]], [0.0, 2.0], {})
    with p1, p2:
        docs = emb.embed_documents(["a", "b"])
        assert np.allclose(docs[0], [0.6, 0.8])      # now on the unit circle
        assert docs[1] == [0.0, 0.0]                 # zero vector: no division by zero
        assert np.allclose(emb.embed_query("q"), [0.0, 1.0])


def test_full_dimension_vectors_are_left_untouched():
    emb = config.get_embeddings()
    p1, p2 = _patched(emb, [[3.0, 4.0]], [0.0, 2.0], {})
    with p1, p2:
        assert emb.embed_documents(["a"]) == [[3.0, 4.0]]

