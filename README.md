# ragtriage

[![CI](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml/badge.svg)](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml)

Measure RAG pipelines, compare configurations, and find out where the failures come from, with first-class support for Turkish.

> **Status: early.** Retrieval evaluation works and is tested. Generation metrics (faithfulness, answer relevance) are next. See the [roadmap](#roadmap).

## Quickstart

Requires Python 3.11+. The demo needs no API key.

```bash
pip install -e ".[dev]"
ragtriage run examples/turkish_quickstart/config.yaml --failures
```

## Define experiments in YAML

Each experiment combines a chunker and a retriever, and one command compares them:

```yaml
documents: docs
dataset: qa.jsonl
k: 3
depth: 20    # retrieve past k, so ranking failures can be told apart from retrieval failures

experiments:
  - name: fixed 300, prefix
    chunker: {type: fixed, size: 300, overlap: 50}
    retriever: {type: bm25, tokenizer: turkish_prefix}

  - name: sentence 300, MiniLM
    chunker: {type: sentence, max_chars: 300}
    retriever:
      type: vector
      embedder:
        type: sentence_transformers
        model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Misspelled keys and impossible settings fail before any model loads. Add `--out results.json` to save per-question scores.

| Component | Options |
|---|---|
| Chunkers | `fixed`, `sentence`, `recursive` |
| Retrievers | `bm25` (tokenizers: `naive`, `turkish`, `turkish_prefix`), `vector` |
| Embedders | `hashing` (no download), `sentence_transformers` (needs `pip install -e ".[embeddings]"`) |

To add a model, write one class with `embed_documents` and `embed_query`.

## Example: chunking and retrieval on Turkish text

The demo runs over six short Turkish documents with 23 test questions, 2 of them unanswerable. Each row is one configuration, scored at k=3:

| Experiment | Chunks | Recall@3 | Precision@3 | MRR | nDCG@3 | Context chars@3 | OK | Ranking | Retrieval |
|---|---|---|---|---|---|---|---|---|---|
| fixed 300, naive | 21 | 0.833 | 0.317 | 0.786 | 0.786 | 702 | 17 | 1 | 3 |
| fixed 300, turkish | 21 | 0.833 | 0.333 | 0.817 | 0.804 | 696 | 17 | 2 | 2 |
| fixed 300, prefix | 21 | 1.000 | 0.429 | 0.952 | 0.965 | 803 | 21 | 0 | 0 |
| sentence 300, prefix | 24 | 0.976 | 0.349 | 0.905 | 0.911 | 621 | 20 | 1 | 0 |
| recursive 300, prefix | 27 | 0.929 | 0.413 | 0.841 | 0.860 | 639 | 19 | 1 | 1 |
| sentence 300, hashing | 24 | 1.000 | 0.365 | 0.897 | 0.926 | 677 | 21 | 0 | 0 |

The `fixed` and `recursive` rows use a 50-character overlap, and `sentence` uses none, so the chunking rows are not strictly like for like.

What this run shows:

- **Suffixes cost more than casing.** Cutting words to five letters removed all 4 failures of the fixed chunker, because inflected forms now share a token. For example, a question that says "peribacaları" can now match "Peribacası" in the document. Correct lowercasing alone left recall unchanged.
- **Neater chunks did not win here.** Sentence and recursive chunking both scored lower recall than fixed windows on this test set. One likely reason is that a smaller chunk can lose its subject: the sentence on how Van Gölü formed says only "göl", so on its own it no longer matches a question that names the lake.
- **Evaluation can catch bugs in the pipeline itself.** An early version of the recursive chunker left headings alone in tiny chunks. Its unusually low MRR (0.740) exposed the problem, and fixing it raised MRR to 0.841.

**Read these numbers as a demo, not a benchmark.** The test set has 21 scored questions, so one question moves recall by about 0.05. The same author wrote the documents and the questions, which makes the questions easier than real user queries. A real comparison needs a larger, independent test set.

## Example: the test set decides which retriever wins

This run compares keyword search with a real embedding model, [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It uses two question sets over the same documents. The first is the set above. The second has 10 questions written to avoid the documents' wording. On average, 33% of a question's word stems appear in its answer passage in the first set, and 2% in the second.

| Retriever (fixed 300) | Same wording: Recall@3 | MRR | Reworded: Recall@3 | MRR | Query ms |
|---|---|---|---|---|---|
| BM25, prefix tokens | 1.000 | 0.952 | 0.300 | 0.338 | 0.1 |
| Hashing n-grams | 1.000 | 0.968 | 0.500 | 0.355 | 0.5 |
| MiniLM | 0.857 | 0.758 | 0.900 | 0.742 | ~80 |

- **The ranking flips.** BM25 is best when questions reuse the documents' words and worst when they don't. MiniLM is the opposite.
- **A test set written from the documents favors keyword search.** Questions written by someone reading the source tend to copy its words. Real users usually don't. This is the strongest argument for testing with real queries.
- **The embedding model costs about 1000× more per query.** Latency was measured on CPU on one laptop, so treat it as a rough ratio.

The reworded set has only 10 questions, and the same author wrote it, so it shows the direction of the effect rather than its size. With almost every chunk retrieved at depth 20 on a corpus this small, the ranking and retrieval columns mean little for the vector rows. Reproduce with `config_embeddings.yaml` and `config_paraphrase.yaml`. The first run downloads about 460 MB.

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

The `sentence` chunker also handles Turkish punctuation. The period in "10. yüzyılda" or "1. Dünya Savaşı" marks an ordinal number, and a generic sentence splitter would end the sentence there.

## Limitations

- **Larger chunks match more easily.** They are more likely to cover the evidence, which can flatter recall. Compare the `Context chars` column alongside it.
- **The 0.5 overlap threshold is a choice.** Change it with `min_overlap` and check whether your conclusions hold.
- **No significance testing yet.** Small differences between configurations may be noise.
- **Chunk sizes are in characters, not tokens.** The `fixed` chunker can also cut through a word.
- **CI does not run a real embedding model.** It tests the adapter against a stub to avoid the download. The results above come from a manual run.
- **The `hashing` embedder has no notion of meaning.** It matches similar spelling and cannot connect synonyms, so it is only a baseline.
- **Quality of the answer isn't measured yet.** Good retrieval doesn't guarantee a good answer.

## Project layout

```text
src/ragtriage/
├── models.py             # documents, evidence spans, chunks, runs
├── matching.py           # which evidence does a chunk cover?
├── metrics/retrieval.py  # pure metric functions and failure types
├── evaluate.py           # score runs against a test set
├── datasets.py           # load documents and JSONL test sets
├── config.py             # validated YAML experiment file
├── runner.py             # build pipelines and run every experiment
├── cli.py                # ragtriage run config.yaml
└── pipelines/            # chunkers, tokenizers, BM25, embedders, vector search
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
- [x] Sentence and recursive chunking, vector retrieval, YAML experiments, CLI
- [x] Results from a real embedding model on the demo
- [ ] Generation metrics with an LLM judge, caching, and async calls
- [ ] HTML report with the worst examples and significance testing
- [ ] Synthetic test sets, including unanswerable questions
- [ ] Judge validation against human labels, and a real Turkish case study

## License

[MIT](LICENSE)
