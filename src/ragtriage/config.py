"""The YAML experiment file. Typos and impossible settings fail here, before any model loads."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragtriage.pipelines.chunking import validate_fixed


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FixedChunker(_Spec):
    type: Literal["fixed"]
    size: int
    overlap: int = 0

    @model_validator(mode="after")
    def _check(self) -> Self:
        validate_fixed(self.size, self.overlap)
        return self


class RecursiveChunker(FixedChunker):
    type: Literal["recursive"]  # type: ignore[assignment]


class SentenceChunker(_Spec):
    type: Literal["sentence"]
    max_chars: int = Field(ge=1)
    overlap_sentences: int = Field(default=0, ge=0)


ChunkerSpec = Annotated[
    FixedChunker | RecursiveChunker | SentenceChunker, Field(discriminator="type")
]


class HashingEmbedderSpec(_Spec):
    type: Literal["hashing"]
    dim: int = Field(default=16384, ge=1)
    ngram_min: int = Field(default=3, ge=1)
    ngram_max: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.ngram_min > self.ngram_max:
            raise ValueError(f"ngram_min {self.ngram_min} exceeds ngram_max {self.ngram_max}")
        return self


class SentenceTransformerSpec(_Spec):
    type: Literal["sentence_transformers"]
    model: str
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = Field(default=32, ge=1)


EmbedderSpec = Annotated[HashingEmbedderSpec | SentenceTransformerSpec, Field(discriminator="type")]


class BM25Spec(_Spec):
    type: Literal["bm25"]
    tokenizer: Literal["naive", "turkish", "turkish_prefix"] = "turkish"
    k1: float = Field(default=1.5, ge=0)
    b: float = Field(default=0.75, ge=0, le=1)


class VectorSpec(_Spec):
    type: Literal["vector"]
    embedder: EmbedderSpec


RetrieverSpec = Annotated[BM25Spec | VectorSpec, Field(discriminator="type")]


class Experiment(_Spec):
    name: str = Field(min_length=1)
    chunker: ChunkerSpec
    retriever: RetrieverSpec


class EvalConfig(_Spec):
    documents: Path
    dataset: Path
    k: int = Field(default=5, ge=1)
    depth: int = 20
    min_overlap: float = Field(default=0.5, gt=0, le=1)
    experiments: tuple[Experiment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.depth <= self.k:
            raise ValueError(
                f"depth ({self.depth}) must exceed k ({self.k}), "
                "or ranking failures cannot be told apart from retrieval failures"
            )
        names = [experiment.name for experiment in self.experiments]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate experiment names: {duplicates}")
        return self


def load_config(path: str | Path) -> EvalConfig:
    """Read a config file. Relative paths inside it are relative to the file, not the shell."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    config = EvalConfig.model_validate(raw)
    base = path.resolve().parent
    return config.model_copy(
        update={"documents": base / config.documents, "dataset": base / config.dataset}
    )
