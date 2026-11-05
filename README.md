# ragtriage

[![CI](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml/badge.svg)](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml)

Measure RAG pipelines, compare configurations, and find out where the failures come from, with first-class support for Turkish.

> **Status: early.** Retrieval evaluation works and is tested. Generation metrics (faithfulness, answer relevance) are next. See the [roadmap](#roadmap).

## Quickstart

Requires Python 3.11+. The demo needs no API key.

```bash
pip install -e ".[dev]"
python examples/turkish_quickstart/run.py
```

## Example: tokenizers and chunk sizes on Turkish text

The demo runs BM25 over six short Turkish documents with 23 test questions, 2 of them unanswerable. Each row is one configuration, scored at k=3:

| Experiment | Recall@3 | Precision@3 | MRR | nDCG@3 | Context chars@3 | OK | Ranking | Retrieval |
|---|---|---|---|---|---|---|---|---|
| naive, 300 chars | 0.833 | 0.317 | 0.786 | 0.786 | 702 | 17 | 1 | 3 |
| naive, 600 chars | 0.881 | 0.333 | 0.786 | 0.795 | 1138 | 18 | 1 | 2 |
| turkish, 300 chars | 0.833 | 0.333 | 0.817 | 0.804 | 696 | 17 | 2 | 2 |
| turkish, 600 chars | 0.929 | 0.349 | 0.881 | 0.877 | 1131 | 19 | 1 | 1 |
| turkish+prefix5, 300 chars | 1.000 | 0.429 | 0.952 | 0.965 | 803 | 21 | 0 | 0 |
| turkish+prefix5, 600 chars | 1.000 | 0.381 | 0.952 | 0.961 | 1268 | 21 | 0 | 0 |

What this run shows:

- **Suffixes cost more than casing.** Cutting words to five letters removed every failure, because inflected forms now share a token. For example, the question that says "peribacaları" failed at both chunk sizes until it could match "Peribacası" in the document. Correct Turkish lowercasing alone helped less.
- **Larger chunks were not worth it once suffixes were handled.** With prefix tokens, 600-character chunks sent 58% more context for the same recall.

**Read these numbers as a demo, not a benchmark.** The test set has 21 scored questions, so one question moves recall by about 0.05. The same author wrote the documents and the questions, which makes the questions easier than real user queries. A real comparison needs a larger, independent test set.

## Why evidence is a span, not a chunk id

Most RAG test sets label the correct chunk ids. Those ids change when the chunking changes, so the labels cannot fairly compare a 256-character run with a 512-character run.

ragtriage labels a position in the source document instead. A test set cites the text, and the loader resolves it to offsets:

```json
{"id": "q18", "question": "İznik çinisinin hamurunda ne bulunur?",
 "evidence": [{"doc_id": "iznik", "quote": "yüksek oranda kuvars içeren hamuru"}]}
```

A retrieved chunk counts as covering the evidence when the overlap is at least half of the evidence or half of the chunk. A quote that is missing or appears twice is an error, so a label can't silently point at the wrong passage.

## Metrics

| Metric | Answers | Definition |
|---|---|---|
| Hit Rate@k | Did anything relevant come back? | 1 if the top k contain relevant evidence, else 0 |
| Recall@k | How much of the evidence was found? | evidence covered in the top k / all evidence |
| Precision@k | How much of the context is useful? | relevant chunks in the top k / k |
| MRR | How high is the first relevant chunk? | mean of 1 / rank of the first relevant chunk |
| nDCG@k | Is the best evidence ranked first? | DCG / ideal DCG, with linear gain and each passage credited once |

Unanswerable questions are left out of these averages and counted separately, because retrieval metrics are undefined for them.

## Failure types

Each question is also classified at a cutoff k, which tells you what to fix:

| Type | Meaning | Where to look |
|---|---|---|
| `ok` | All evidence is in the top k | Nothing to fix in retrieval |
| `ranking` | Missing evidence was retrieved, but below k | Add a reranker or raise k |
| `retrieval` | Missing evidence was never retrieved | Chunking, tokenizer, or embeddings |

Retrieve deeper than k. A run that stops at k can't show a ranking failure.

## Turkish text

Python's `str.lower()` is wrong for Turkish, and the error is silent:

```python
>>> "KIRMIZI".lower()
'kirmizi'       # the correct form is "kırmızı", so the two never match
>>> "İznik".lower()
'i̇znik'         # "i" + a combining dot, which a \w+ tokenizer splits into "i" and "znik"
```

`ragtriage.pipelines.text` provides correct lowercasing and prefix truncation, an approach studied for Turkish retrieval by [Can et al. (2008)](https://doi.org/10.1002/asi.20750).

## Limitations

- **Larger chunks match more easily.** They are more likely to cover the evidence, which can flatter recall. Compare the `Context chars` column alongside it.
- **The 0.5 overlap threshold is a choice.** Change it with `min_overlap` and check whether your conclusions hold.
- **No significance testing yet.** Small differences between configurations may be noise.
- **`chunk_fixed` counts characters and can cut through a word.** Sentence-aware chunking is planned.
- **Quality of the answer isn't measured yet.** Good retrieval doesn't guarantee a good answer.

## Project layout

```text
src/ragtriage/
├── models.py             # documents, evidence spans, chunks, runs
├── matching.py           # which evidence does a chunk cover?
├── metrics/retrieval.py  # pure metric functions and failure types
├── evaluate.py           # score runs against a test set
├── datasets.py           # load documents and JSONL test sets
└── pipelines/            # chunking, tokenizers, BM25, Retriever interface
```

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy
```

Metric tests use values worked out by hand, plus property-based tests with [Hypothesis](https://hypothesis.readthedocs.io/) for invariants such as "recall never drops as k grows".

## Relation to other tools

[RAGAS](https://github.com/explodinggradients/ragas), [DeepEval](https://github.com/confident-ai/deepeval), [TruLens](https://github.com/truera/trulens), and [Phoenix](https://github.com/Arize-ai/phoenix) are more mature and cover far more. ragtriage focuses on three narrower things: labels that survive re-chunking, attributing failures to a pipeline stage, and Turkish text.

## Roadmap

- [x] Retrieval metrics, span-based evidence, failure types, BM25 baseline
- [ ] Sentence-aware chunking, embedding retrievers, YAML experiments, CLI
- [ ] Generation metrics with an LLM judge, caching, and async calls
- [ ] HTML report with the worst examples and significance testing
- [ ] Synthetic test sets, including unanswerable questions
- [ ] Judge validation against human labels, and a real Turkish case study

## License

[MIT](LICENSE)
