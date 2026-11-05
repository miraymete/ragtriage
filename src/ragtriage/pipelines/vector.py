"""Exact cosine search. Fast enough for evaluation sets, with no index to tune."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragtriage.models import Chunk, RetrievedChunk
from ragtriage.pipelines.embeddings import Embedder, Vector


def _normalize(vectors: Vector) -> Vector:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # A text with no tokens embeds to all zeros. It must score 0 rather than divide by zero.
    normalized: Vector = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return normalized


class VectorRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self._chunks = list(chunks)
        self._embedder = embedder
        self._matrix = _normalize(embedder.embed_documents([chunk.text for chunk in self._chunks]))
        if self._matrix.ndim != 2 or len(self._matrix) != len(self._chunks):
            raise ValueError(
                f"embedder returned shape {self._matrix.shape} for {len(self._chunks)} chunks"
            )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` most similar chunks, ties broken by index."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        query_vector = _normalize(self._embedder.embed_query(query))
        if not np.any(query_vector):
            return []
        scores = self._matrix @ query_vector
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievedChunk(**self._chunks[i].model_dump(), score=float(scores[i])) for i in order
        ]
