# 01 · Embeddings and Cosine Similarity

Turns four sentences into vectors with `gemini-embedding-001` and measures
how similar they are, with a hand-written cosine similarity function. This
is the mechanism the retrieval step of [03-rag](../03-rag/) depends on.

```bash
python 01-embeddings/embeddings_demo.py
```

## The experiment

The sentences are chosen to test a hypothesis, with a control group:

| Pair | Relationship | Expected |
|------|--------------|----------|
| [0] dogs ↔ [2] puppies | same concept, almost no shared words | highest |
| [0] dogs ↔ [1] cats | shared parent concept (pets) | medium |
| [3] stock market ↔ others | unrelated | lowest |

The first pair matters most: a high score despite almost no lexical
overlap shows that the model captures meaning, not word matching.

## The geometry

`gemini-embedding-001` returns 3072-dimensional vectors of unit length, so
every sentence is a point on the sphere $S^{3071} \subset \mathbb{R}^{3072}$.

- With $\|u\| = \|v\| = 1$, cosine similarity reduces to the dot product:
  $\cos\theta = u \cdot v$. Vector databases rely on this.
- Euclidean distance carries the same information:
  $\|u - v\|^2 = 2 - 2\cos\theta$. On the sphere, cosine and Euclidean
  distance give **the same ranking**.
- The geodesic (great-circle) distance between two sentences is the angle
  $\theta = \arccos(u \cdot v)$. The demo prints it in degrees.
- **Matryoshka truncation.** The model can return shorter vectors (e.g. 768
  dimensions) with little quality loss, but the API normalizes only full-size
  outputs. A truncated vector is no longer on the sphere, and
  $\cos\theta = u \cdot v$ silently breaks. `llm_series/config.py`
  re-normalizes truncated vectors, so the identity holds in every
  configuration. The hand-written function divides by the norms and would be
  correct either way.
- **Anisotropy.** Embeddings occupy a narrow cone of the sphere, so unrelated
  sentences rarely score near 0. The ranking carries the meaning, not the
  absolute value.

## Design decisions

- **Cosine similarity by hand.** It shows the formula, and it is exactly the
  operation retrieval performs in project 03.
- **Zero-vector guard.** A zero vector has no direction; the function
  returns 0 instead of dividing by zero.
- **Vectorized matrix.** After row normalization the whole similarity matrix
  is one matrix product $VV^\top$ instead of an $O(n^2)$ Python loop.
- **Task type.** All four sentences play the same role here, so they are
  embedded with `SEMANTIC_SIMILARITY`. RAG is asymmetric and uses
  `RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for questions; see
  project 03.

## Scaling beyond four sentences

Comparing a query with every stored vector (brute force) does not scale to
millions of documents. Vector databases use approximate nearest neighbour
(ANN) indexes such as HNSW, trading a little accuracy for a large speed-up.
Project 03 uses Chroma for this.
