"""Measure RAG pipelines and find out where the failures come from."""

from ragtriage.datasets import load_documents, load_qa_examples
from ragtriage.evaluate import RetrievalReport, evaluate_retrieval
from ragtriage.metrics.retrieval import Failure
from ragtriage.models import (
    Chunk,
    Document,
    Evidence,
    QAExample,
    RetrievalRun,
    RetrievedChunk,
)

__all__ = [
    "Chunk",
    "Document",
    "Evidence",
    "Failure",
    "QAExample",
    "RetrievalReport",
    "RetrievalRun",
    "RetrievedChunk",
    "evaluate_retrieval",
    "load_documents",
    "load_qa_examples",
]
__version__ = "0.1.0"
