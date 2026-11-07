import pandas as pd
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from llm_series.documents import (
    EmptyDocumentError,
    build_vectorstore,
    fingerprint,
    format_context,
    load_pdf,
    page_label,
    read_csv_robust,
    replace_vectorstore,
)

EMB = DeterministicFakeEmbedding(size=64)


def test_fingerprint_depends_on_content_not_name():
    assert fingerprint(b"same") == fingerprint(b"same")
    assert fingerprint(b"one") != fingerprint(b"two")


def test_pdf_pages_are_loaded_with_page_numbers(make_pdf):
    docs = load_pdf(make_pdf(["First page text", "Second page text"]))
    assert len(docs) == 2
    assert [page_label(d) for d in docs] == ["1", "2"]
    assert "Second page" in docs[1].page_content
    assert "[page 2]" in format_context(docs)


def test_pdf_without_text_raises_a_clear_error(make_pdf):
    with pytest.raises(EmptyDocumentError, match="OCR"):
        build_vectorstore(make_pdf([""]), EMB)


def test_new_document_replaces_old_one_without_mixing(make_pdf):
    """Regression test for the original bug: a second upload was ignored, and
    with a shared default collection old chunks could leak into new answers."""
    store_a, _ = build_vectorstore(make_pdf(["ALPHA document about apples"]), EMB)
    store_b, _ = build_vectorstore(make_pdf(["BETA document about bananas"]), EMB)
    old_name = store_a._collection.name
    assert old_name != store_b._collection.name

    current = replace_vectorstore(store_a, store_b)
    texts = [d.page_content for d in current.similarity_search("apples", k=10)]
    assert texts and all("BETA" in t for t in texts)

    # The old collection was deleted from the shared in-memory client.
    names = [c.name for c in store_b._client.list_collections()]
    assert old_name not in names


def test_replace_with_no_previous_store():
    sentinel = object()
    assert replace_vectorstore(None, sentinel) is sentinel


def test_turkish_excel_csv_is_parsed():
    # Semicolon separator, decimal comma, Windows-1254 encoding.
    raw = "Şehir;Sıcaklık\nİzmir;21,5\nManisa;19,0\n".encode("cp1254")
    df = read_csv_robust(raw)
    assert list(df.columns) == ["Şehir", "Sıcaklık"]
    assert pd.api.types.is_numeric_dtype(df["Sıcaklık"])
    assert df["Sıcaklık"].sum() == pytest.approx(40.5)


def test_standard_csv_is_parsed():
    df = read_csv_robust(b"name,score\nada,3.5\nbob,4\n")
    assert df.shape == (2, 2)
    assert df["score"].mean() == pytest.approx(3.75)
