"""Embedders. Add a model by writing one class with `embed_documents` and `embed_query`."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import blake2b
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ragtriage.pipelines.text import turkish_tokens

Vector = NDArray[np.float32]


class Embedder(Protocol):
    """Queries and documents are separate because models such as E5 prefix them differently."""

    def embed_documents(self, texts: Sequence[str]) -> Vector: ...

    def embed_query(self, text: str) -> Vector: ...


class HashingEmbedder:
    """Hashed character n-grams: a deterministic baseline with no download and no API key.

    Shared n-grams let inflected Turkish words match, but this has no notion of meaning, so it
    cannot connect synonyms. Use it to check a pipeline and as a floor for real models to beat.
    """

    def __init__(self, dim: int = 16384, ngram_min: int = 3, ngram_max: int = 5) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not 1 <= ngram_min <= ngram_max:
            raise ValueError(f"invalid n-gram range: {ngram_min}..{ngram_max}")
        self._dim = dim
        self._ngram_min = ngram_min
        self._ngram_max = ngram_max

    def _ngrams(self, text: str) -> list[str]:
        grams: list[str] = []
        for token in turkish_tokens(text):
            padded = f" {token} "
            for n in range(self._ngram_min, self._ngram_max + 1):
                grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
        return grams

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self._dim, dtype=np.float32)
        for gram in self._ngrams(text):
            # blake2b rather than hash(): Python randomizes str hashes between runs.
            digest = int.from_bytes(blake2b(gram.encode(), digest_size=8).digest(), "big")
            vector[digest % self._dim] += 1.0 if digest >> 63 else -1.0
        return np.sign(vector) * np.log1p(np.abs(vector))

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """Any Sentence Transformers model. Install with `pip install "ragtriage[embeddings]"`."""

    def __init__(
        self,
        model: str,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError('this embedder needs: pip install "ragtriage[embeddings]"') from error
        self._model = SentenceTransformer(model)
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> Vector:
        encoded: Vector = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return encoded

    def embed_documents(self, texts: Sequence[str]) -> Vector:
        return self._encode([self._document_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._encode([self._query_prefix + text])[0]
        return vector
