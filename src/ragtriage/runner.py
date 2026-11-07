"""Turn a config into pipelines, run every experiment, and collect the results."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from time import perf_counter

from pydantic import BaseModel, ConfigDict

from ragtriage.config import (
    AnthropicSpec,
    BM25Spec,
    CacheSpec,
    ChunkerSpec,
    EmbedderSpec,
    EvalConfig,
    FixedChunker,
    GenerationSpec,
    HashingEmbedderSpec,
    RecursiveChunker,
    RetrieverSpec,
    SentenceChunker,
)
from ragtriage.datasets import load_documents, load_qa_examples
from ragtriage.evaluate import RetrievalReport, evaluate_retrieval
from ragtriage.generation.cache import CachedLLM, ResponseCache
from ragtriage.generation.costs import Price
from ragtriage.generation.generator import Generator
from ragtriage.generation.judge import FaithfulnessJudge
from ragtriage.generation.llm import LLMClient
from ragtriage.generation.providers.anthropic import AnthropicClient
from ragtriage.generation.relevance import RelevanceJudge
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


def build_llm(spec: AnthropicSpec) -> LLMClient:
    return AnthropicClient(
        spec.model, effort=spec.effort, max_retries=spec.max_retries, timeout_s=spec.timeout_s
    )


def build_generator(spec: GenerationSpec, k: int, min_overlap: float) -> Generator:
    """Raises CacheError for an unusable cache file and ImportError for a missing SDK, both
    before any request is sent."""
    llm = spec.generator
    # Open the cache first: if it is unusable, no client (and no connection) has been created.
    cache = ResponseCache(spec.cache.path) if spec.cache and spec.cache.enabled else None
    try:
        inner = build_llm(llm)
    except BaseException:
        if cache is not None:
            cache.close()
        raise
    client: LLMClient = CachedLLM(inner, cache) if cache is not None else inner
    price = (
        Price(
            input_per_mtok=llm.price_per_mtok.input_per_mtok,
            output_per_mtok=llm.price_per_mtok.output_per_mtok,
        )
        if llm.price_per_mtok
        else None
    )
    return Generator(client, k=k, max_tokens=llm.max_tokens, price=price, min_overlap=min_overlap)


def _judge_client(llm: AnthropicSpec, cache_spec: CacheSpec | None) -> LLMClient:
    """A judge client with its own connection to the shared cache file, if one is enabled."""
    cache = ResponseCache(cache_spec.path) if cache_spec and cache_spec.enabled else None
    try:
        inner = build_llm(llm)
    except BaseException:
        if cache is not None:
            cache.close()
        raise
    return CachedLLM(inner, cache) if cache is not None else inner


def _price(llm: AnthropicSpec) -> Price | None:
    price = llm.price_per_mtok
    if price is None:
        return None
    return Price(input_per_mtok=price.input_per_mtok, output_per_mtok=price.output_per_mtok)


def build_judge(spec: GenerationSpec) -> FaithfulnessJudge | None:
    """The faithfulness judge, or None if none is configured."""
    if spec.judge is None:
        return None
    client = _judge_client(spec.judge, spec.cache)
    return FaithfulnessJudge(client, max_tokens=spec.judge.max_tokens, price=_price(spec.judge))


def build_relevance(spec: GenerationSpec) -> RelevanceJudge | None:
    """The answer-relevance judge, or None. Builds the configured embedder explicitly."""
    if spec.relevance is None:
        return None
    embedder = build_embedder(spec.relevance.embedder)
    llm = spec.relevance.judge
    return RelevanceJudge(
        _judge_client(llm, spec.cache),
        embedder,
        embedder_settings=spec.relevance.embedder.model_dump(mode="json"),
        max_tokens=llm.max_tokens,
        price=_price(llm),
    )


@dataclass(frozen=True)
class RetrievalOutputs:
    """Everything one retrieval pass produced, kept for generation and the HTML report."""

    documents: dict[str, Document]
    examples: list[QAExample]
    experiments: list[tuple[ExperimentResult, list[RetrievalRun]]]


def run_experiments(config: EvalConfig) -> list[ExperimentResult]:
    return [result for result, _ in run_retrieval(config).experiments]


def run_retrieval(config: EvalConfig) -> RetrievalOutputs:
    """Run every experiment and keep its runs, which generation and the report need."""
    documents = load_documents(config.documents)
    examples: list[QAExample] = load_qa_examples(config.dataset, documents)
    embedders: dict[EmbedderSpec, Embedder] = {}
    results: list[tuple[ExperimentResult, list[RetrievalRun]]] = []
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
        result = ExperimentResult(
            name=experiment.name,
            n_chunks=len(chunks),
            report=report,
            context_chars=fmean(
                sum(len(chunk.text) for chunk in run.chunks[: config.k]) for run in runs
            ),
            index_seconds=index_seconds,
            retrieval_ms=fmean(latencies) * 1000,
        )
        results.append((result, runs))
    return RetrievalOutputs(documents=documents, examples=examples, experiments=results)
