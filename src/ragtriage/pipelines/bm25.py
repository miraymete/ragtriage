"""A dependency-free BM25 retriever: a baseline that needs no API key."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from math import log

from ragtriage.models import Chunk, RetrievedChunk
from ragtriage.pipelines.text import turkish_tokens

Tokenizer = Callable[[str], list[str]]


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Tokenizer = turkish_tokens,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("cannot index an empty chunk list")
        self._chunks = list(chunks)
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        tokenized = [tokenizer(chunk.text) for chunk in self._chunks]
        self._term_counts = [Counter(tokens) for tokens in tokenized]
        self._lengths = [len(tokens) for tokens in tokenized]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) or 1.0
        document_frequency = Counter(term for counts in self._term_counts for term in counts)
        total = len(self._chunks)
        self._idf = {
            term: log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = self._term_counts[index]
        norm = 1 - self._b + self._b * self._lengths[index] / self._avg_length
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency:
                score += (
                    self._idf[term] * frequency * (self._k1 + 1) / (frequency + self._k1 * norm)
                )
        return score

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks sharing a term with the query, ties broken by index."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        terms = self._tokenizer(query)
        scored = [(self._score(terms, i), i) for i in range(len(self._chunks))]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(**self._chunks[i].model_dump(), score=score)
            for score, i in scored[:top_k]
        ]
