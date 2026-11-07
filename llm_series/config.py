"""
Shared configuration
====================

Every project in the series gets its chat model and embedding model from
here. The goals:

1. **Secrets stay out of the code.** API keys and model names are read only
   from environment variables (a local ``.env`` file).
2. **Single source of truth.** Changing a model name or the embedding size
   happens in one place; all five projects stay consistent.
3. **Provider-specific details are encapsulated.** Gemini's task types and
   normalization behaviour do not leak into the projects.
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ----------------------------------------------------------------------
# Defaults, used when the .env file does not set a value
# ----------------------------------------------------------------------
DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# Full output size of gemini-embedding-001. The API returns unit-length
# (normalized) vectors only at this size.
FULL_EMBEDDING_DIM = 3072


class ConfigError(RuntimeError):
    """Readable error for missing or invalid configuration."""


def _require_api_key() -> None:
    """Fail early with a clear message instead of a deep library error."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ConfigError(
            "No Gemini API key found. Copy '.env.example' to '.env' and fill in "
            "GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey)."
        )


def _embedding_dim() -> int | None:
    """None (full size) if GEMINI_EMBEDDING_DIM is empty, otherwise an int."""
    raw = os.getenv("GEMINI_EMBEDDING_DIM", "").strip()
    if not raw:
        return None
    try:
        dim = int(raw)
    except ValueError as exc:
        raise ConfigError(f"GEMINI_EMBEDDING_DIM must be an integer, got '{raw}'.") from exc
    if not 128 <= dim <= FULL_EMBEDDING_DIM:
        raise ConfigError(f"GEMINI_EMBEDDING_DIM must be between 128 and {FULL_EMBEDDING_DIM}.")
    return dim


# ----------------------------------------------------------------------
# Chat model
# ----------------------------------------------------------------------
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return the Gemini chat model.

    Temperature is deliberately *not* set. Google recommends keeping the
    default sampling parameters (temperature, top_p, top_k) for Gemini 3
    models because their reasoning is tuned for them; langchain-google-genai
    omits temperature for Gemini 3+ when it is not passed explicitly.
    """
    _require_api_key()
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL))


# ----------------------------------------------------------------------
# Embedding model
# ----------------------------------------------------------------------
class RetrievalEmbeddings(GoogleGenerativeAIEmbeddings):
    """Thin wrapper that guarantees two things:

    1. **Asymmetric task types.** Documents are embedded with
       RETRIEVAL_DOCUMENT and queries with RETRIEVAL_QUERY. Vector stores
       such as Chroma already call embed_documents / embed_query separately,
       so every project gets the right behaviour with no extra code.
       A caller can still override it, e.g. task_type="SEMANTIC_SIMILARITY".

    2. **Points on the unit sphere.** When the size is reduced with
       Matryoshka truncation (output_dimensionality < 3072) the API does not
       normalize the vectors, so they are L2-normalized here. All vectors
       therefore lie on S^(d-1) in every configuration, and
       cosine similarity = dot product holds.
    """

    def _maybe_normalize(self, vectors: list[list[float]]) -> list[list[float]]:
        dim = self.output_dimensionality
        if dim is None or dim == FULL_EMBEDDING_DIM:
            return vectors
        arr = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero for a zero vector
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        kwargs.setdefault("task_type", "RETRIEVAL_DOCUMENT")
        return self._maybe_normalize(super().embed_documents(texts, **kwargs))

    def embed_query(self, text: str, **kwargs) -> list[float]:
        kwargs.setdefault("task_type", "RETRIEVAL_QUERY")
        return self._maybe_normalize([super().embed_query(text, **kwargs)])[0]


def get_embeddings() -> RetrievalEmbeddings:
    """Return the Gemini embedding model (task types and normalization included)."""
    _require_api_key()
    kwargs = {"model": os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)}
    dim = _embedding_dim()
    if dim is not None:
        kwargs["output_dimensionality"] = dim
    return RetrievalEmbeddings(**kwargs)


def tavily_available() -> bool:
    """Whether the web search tool can be enabled (projects 04 and 05)."""
    return bool(os.getenv("TAVILY_API_KEY"))
