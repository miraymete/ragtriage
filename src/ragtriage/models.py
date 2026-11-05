"""Core data model.

Gold evidence is a character span inside a document, not a chunk id. Chunk ids change whenever the
chunking changes, so id-based labels cannot compare a 256-character run with a 512-character run.
Spans stay valid across every configuration.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Span(_Frozen):
    doc_id: str
    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"span must be non-empty, got [{self.start}, {self.end})")
        return self


class Document(_Frozen):
    id: str
    text: str


class Evidence(_Span):
    """A passage that supports the answer. `grade` is its relevance, used only by nDCG."""

    grade: int = Field(default=1, ge=1)


class Chunk(_Span):
    text: str

    @model_validator(mode="after")
    def _text_matches_span(self) -> Self:
        if len(self.text) != self.end - self.start:
            raise ValueError("chunk text length must equal end - start")
        return self


class RetrievedChunk(Chunk):
    score: float


class QAExample(_Frozen):
    """A test question. Empty `evidence` marks a question the documents cannot answer."""

    id: str
    question: str
    evidence: tuple[Evidence, ...] = ()
    reference_answer: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


class RetrievalRun(_Frozen):
    """What a retriever returned for one example, best first."""

    example_id: str
    chunks: tuple[RetrievedChunk, ...]
