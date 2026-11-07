"""
Document ingestion for RAG (used by 03-rag and 05-unified-agent)
================================================================

The indexing half of a RAG system:

    PDF bytes ──► pages ──► chunks ──► embeddings ──► Chroma collection

Design notes
    * **File identity by content, not by name.** A SHA-256 fingerprint of the
      bytes decides whether an upload is new. Same name + different content is
      re-indexed; different name + same content is not re-embedded (saves cost).
    * **One collection per document, old one deleted.** An in-memory Chroma
      client is shared by the whole Python process. Reusing the default
      collection name would silently mix chunks from the previous document
      into the new one. Each upload therefore gets a fresh, uniquely named
      collection and the previous collection is dropped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import uuid

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000      # characters, not tokens
CHUNK_OVERLAP = 200    # 20 % overlap keeps sentences on a boundary intact


class EmptyDocumentError(ValueError):
    """The PDF has no extractable text (e.g. a scanned image without OCR)."""


def fingerprint(data: bytes) -> str:
    """Content-based identity for an uploaded file."""
    return hashlib.sha256(data).hexdigest()


def load_pdf(data: bytes) -> list[Document]:
    """Parse PDF bytes into one Document per page.

    PyPDFLoader expects a path on disk while Streamlit keeps uploads in
    memory, so the bytes go to a temporary file that is always removed,
    even if parsing fails (try/finally).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(data)
        tmp.close()
        return PyPDFLoader(tmp.name).load()
    finally:
        os.unlink(tmp.name)


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise EmptyDocumentError(
            "No text could be extracted. The PDF may be a scanned image; "
            "it would need OCR before it can be searched."
        )
    return chunks


def index_documents(chunks: list[Document], embeddings: Embeddings) -> Chroma:
    """Embed chunks into a brand-new, uniquely named Chroma collection."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"doc-{uuid.uuid4().hex}",
    )


def replace_vectorstore(old: Chroma | None, new: Chroma) -> Chroma:
    """Drop the previous document's collection so nothing leaks into the new one."""
    if old is not None:
        try:
            old.delete_collection()
        except Exception:  # already gone — nothing to clean up
            pass
    return new


def build_vectorstore(data: bytes, embeddings: Embeddings) -> tuple[Chroma, int]:
    """Full indexing pipeline. Returns the store and the number of chunks."""
    chunks = split_documents(load_pdf(data))
    return index_documents(chunks, embeddings), len(chunks)


def page_label(doc: Document) -> str:
    """Human-readable page number for citations (PyPDFLoader pages are 0-based)."""
    label = doc.metadata.get("page_label")
    if label:
        return str(label)
    page = doc.metadata.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


def format_context(docs: list[Document]) -> str:
    """Join retrieved chunks, each tagged with its page so answers can cite it."""
    return "\n\n".join(f"[page {page_label(d)}]\n{d.page_content}" for d in docs)


# ----------------------------------------------------------------------
# CSV loading (05-unified-agent)
# ----------------------------------------------------------------------
def read_csv_robust(data: bytes) -> pd.DataFrame:
    """Read a CSV regardless of common regional formats.

    CSVs exported from Excel in Turkey (and much of Europe) usually use ';'
    as the separator, ',' as the decimal mark and Windows-1254 encoding.
    With pandas defaults such a file is read as one single column.
    """
    text = None
    for encoding in ("utf-8-sig", "cp1254", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # latin-1 never fails, but keep the type checker happy
        raise ValueError("Could not decode the file.")

    try:
        delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    decimal = "," if delimiter == ";" else "."
    return pd.read_csv(io.StringIO(text), sep=delimiter, decimal=decimal)
