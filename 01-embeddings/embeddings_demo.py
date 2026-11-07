"""
01 · Embeddings and cosine similarity
=====================================

Turns four sentences into vectors with gemini-embedding-001 and measures how
similar they are. Cosine similarity is implemented by hand on purpose: it is
exactly the operation the retrieval step of the RAG project (03) relies on.

The sentences form a tiny experiment with a control group:
    [0] dogs  vs [2] puppies -> same concept, almost no shared words  -> highest
    [0] dogs  vs [1] cats    -> shared parent concept (pets)           -> medium
    [3] stock market vs rest -> unrelated topic                         -> lowest

Run:  python 01-embeddings/embeddings_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from llm_series.config import get_embeddings

SENTENCES = [
    "Dogs are loyal and friendly companions.",      # 0
    "Cats are independent and enjoy their space.",  # 1
    "Puppies love to play fetch in the park.",      # 2
    "The stock market fell sharply today.",         # 3
]


def cosine_similarity(vec1, vec2) -> float:
    """cos θ = (u · v) / (‖u‖ ‖v‖)

    Dividing by the norms makes this correct for any vectors, including
    non-normalized ones. A zero vector has no direction, so 0.0 is returned
    instead of dividing by zero.
    """
    u = np.asarray(vec1, dtype=np.float64)
    v = np.asarray(vec2, dtype=np.float64)
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    if norm_product == 0:
        return 0.0
    return float(np.dot(u, v) / norm_product)


def similarity_matrix(vectors) -> np.ndarray:
    """All pairwise similarities at once.

    After normalizing every row to unit length, cosine similarity is just a
    dot product, so the full matrix is a single matrix product V Vᵀ instead
    of an O(n²) Python loop.
    """
    V = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    return V @ V.T


def main() -> None:
    # SEMANTIC_SIMILARITY: here every sentence plays the same role (we compare
    # them with each other), unlike RAG where queries and documents differ.
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(SENTENCES, task_type="SEMANTIC_SIMILARITY")

    print(f"Number of vectors : {len(vectors)}")
    print(f"Dimension         : {len(vectors[0])}")
    print(f"First 5 values    : {np.round(vectors[0][:5], 4).tolist()}")
    norms = [np.linalg.norm(v) for v in vectors]
    print(f"Vector norms      : {np.round(norms, 4).tolist()}  (≈1 → points on the unit sphere)")

    print("\n--- Pairwise cosine similarity (hand-written function) ---")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            score = cosine_similarity(vectors[i], vectors[j])
            # On the unit sphere the geodesic (great-circle) distance between
            # two points is the angle θ = arccos(u · v).
            angle = math.degrees(math.acos(max(-1.0, min(1.0, score))))
            print(f"[{i}] vs [{j}]: cos = {score:.4f}   angle = {angle:5.1f}°")
            print(f'    "{SENTENCES[i]}"')
            print(f'    "{SENTENCES[j]}"\n')

    print("--- Similarity matrix (vectorized, V Vᵀ) ---")
    matrix = similarity_matrix(vectors)
    print("      " + "".join(f"[{j}]    " for j in range(len(SENTENCES))))
    for i, row in enumerate(matrix):
        print(f"[{i}]  " + "  ".join(f"{x:.3f}" for x in row))

    print(
        "\nNote: unrelated sentences rarely score near 0. Embeddings occupy a narrow"
        "\ncone of the sphere (anisotropy), so the ranking matters, not the absolute value."
    )


if __name__ == "__main__":
    main()
