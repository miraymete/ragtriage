"""Compare tokenizers and chunk sizes on a small Turkish corpus. Needs no API key.

python examples/turkish_quickstart/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import fmean

from ragtriage import Failure, RetrievalRun, evaluate_retrieval, load_documents, load_qa_examples
from ragtriage.pipelines.bm25 import BM25Retriever
from ragtriage.pipelines.chunking import chunk_fixed
from ragtriage.pipelines.text import naive_tokens, turkish_prefix_tokens, turkish_tokens

HERE = Path(__file__).parent
K = 3
DEPTH = 20  # Retrieve past K so that ranking failures can be told apart from retrieval failures.
OVERLAP = 50
TOKENIZERS = {
    "naive": naive_tokens,
    "turkish": turkish_tokens,
    "turkish+prefix5": turkish_prefix_tokens,
}
CHUNK_SIZES = (300, 600)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    documents = load_documents(HERE / "docs")
    examples = load_qa_examples(HERE / "qa.jsonl", documents)

    print(
        f"| Experiment | Recall@{K} | Precision@{K} | MRR | nDCG@{K} "
        f"| Context chars@{K} | OK | Ranking | Retrieval |"
    )
    print("|---|---|---|---|---|---|---|---|---|")

    last_failures: list[str] = []
    for name, tokenizer in TOKENIZERS.items():
        for size in CHUNK_SIZES:
            chunks = [
                chunk
                for document in documents.values()
                for chunk in chunk_fixed(document, size=size, overlap=OVERLAP)
            ]
            retriever = BM25Retriever(chunks, tokenizer=tokenizer)
            runs = [
                RetrievalRun(
                    example_id=example.id,
                    chunks=tuple(retriever.retrieve(example.question, top_k=DEPTH)),
                )
                for example in examples
            ]
            report = evaluate_retrieval(examples, runs, ks=(K,), triage_k=K)
            context = fmean(sum(len(c.text) for c in run.chunks[:K]) for run in runs)
            mean, failures = report.mean, report.failures
            print(
                f"| {name}, {size} chars "
                f"| {mean[f'recall@{K}']:.3f} | {mean[f'precision@{K}']:.3f} "
                f"| {mean['mrr']:.3f} | {mean[f'ndcg@{K}']:.3f} | {context:.0f} "
                f"| {failures.get(Failure.OK, 0)} | {failures.get(Failure.RANKING, 0)} "
                f"| {failures.get(Failure.RETRIEVAL, 0)} |"
            )
            questions = {example.id: example.question for example in examples}
            last_failures = [
                f"- [{score.failure}] {score.example_id}: {questions[score.example_id]}"
                for score in report.per_example
                if score.failure is not Failure.OK
            ]

    print(f"\nFailures in the last experiment ({name}, {size} chars):")
    print("\n".join(last_failures) or "- none")
    print(f"\nScored {len(report.per_example)} questions.", end=" ")
    print(f"Skipped {report.n_unanswerable_skipped} unanswerable.")


if __name__ == "__main__":
    main()
