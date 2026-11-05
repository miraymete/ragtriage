"""Turn a config into pipelines, run every experiment, and collect the results."""

from __future__ import annotations

from statistics import fmean
from time import perf_counter

from pydantic import BaseModel, ConfigDict

from ragtriage.config import (
    BM25Spec,
    ChunkerSpec,
    EmbedderSpec,
    EvalConfig,
    FixedChunker,
    HashingEmbedderSpec,
    RecursiveChunker,
    RetrieverSpec,
    SentenceChunker,
)
from ragtriage.datasets import load_documents, load_qa_examples
from ragtriage.evaluate import RetrievalReport, evaluate_retrieval
from ragtriage.models import Chunk, Document, QAExample, RetrievalRun
from ragtriage.pipelines.base import Retriever
from ragtriage.pipelines.bm25 import BM25Retriever, Tokenizer
from ragtriage.pipelines.chunking import chunk_fixed, chunk_recursive, chunk_sentences
from ragtriage.pipelines.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from ragtriage.pipelines.text import naive_tokens, turkish_prefix_tokens, turkish_tokens
from ragtriage.pipelines.vector import VectorRetriever

_TOKENIZERS: dict[str, Tokenizer] = {
    "naive": naive_tokens,
    "turkish": turkish_tokens,
    "turkish_prefix": turkish_prefix_tokens,
}


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    n_chunks: int
    report: RetrievalReport
    context_chars: float
    index_seconds: float
    retrieval_ms: float


def build_chunks(spec: ChunkerSpec, document: Document) -> list[Chunk]:
    if isinstance(spec, RecursiveChunker):
        return chunk_recursive(document, spec.size, spec.overlap)
    if isinstance(spec, FixedChunker):
        return chunk_fixed(document, spec.size, spec.overlap)
    if isinstance(spec, SentenceChunker):
        return chunk_sentences(document, spec.max_chars, spec.overlap_sentences)
    raise TypeError(f"unknown chunker: {spec!r}")


def build_embedder(spec: EmbedderSpec) -> Embedder:
    if isinstance(spec, HashingEmbedderSpec):
        return HashingEmbedder(spec.dim, spec.ngram_min, spec.ngram_max)
    return SentenceTransformerEmbedder(
        spec.model, spec.query_prefix, spec.document_prefix, spec.batch_size
    )


def build_retriever(
    spec: RetrieverSpec, chunks: list[Chunk], embedders: dict[EmbedderSpec, Embedder]
) -> Retriever:
    if isinstance(spec, BM25Spec):
        return BM25Retriever(chunks, _TOKENIZERS[spec.tokenizer], k1=spec.k1, b=spec.b)
    # Experiments often share a model, and loading one can take many seconds.
    if spec.embedder not in embedders:
        embedders[spec.embedder] = build_embedder(spec.embedder)
    return VectorRetriever(chunks, embedders[spec.embedder])


def run_experiments(config: EvalConfig) -> list[ExperimentResult]:
    documents = load_documents(config.documents)
    examples: list[QAExample] = load_qa_examples(config.dataset, documents)
    embedders: dict[EmbedderSpec, Embedder] = {}
    results = []
    for experiment in config.experiments:
        chunks = [
            chunk
            for document in documents.values()
            for chunk in build_chunks(experiment.chunker, document)
        ]
        started = perf_counter()
        retriever = build_retriever(experiment.retriever, chunks, embedders)
        index_seconds = perf_counter() - started

        runs = []
        latencies = []
        for example in examples:
            started = perf_counter()
            retrieved = retriever.retrieve(example.question, top_k=config.depth)
            latencies.append(perf_counter() - started)
            runs.append(RetrievalRun(example_id=example.id, chunks=tuple(retrieved)))

        report = evaluate_retrieval(
            examples, runs, ks=(config.k,), triage_k=config.k, min_overlap=config.min_overlap
        )
        results.append(
            ExperimentResult(
                name=experiment.name,
                n_chunks=len(chunks),
                report=report,
                context_chars=fmean(
                    sum(len(chunk.text) for chunk in run.chunks[: config.k]) for run in runs
                ),
                index_seconds=index_seconds,
                retrieval_ms=fmean(latencies) * 1000,
            )
        )
    return results
