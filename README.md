# ragtriage

[![CI](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml/badge.svg)](https://github.com/miraymete/ragtriage/actions/workflows/ci.yml)

Measure RAG pipelines, compare configurations, and find out where the failures come from, with first-class support for Turkish.

> **Status: early.** Retrieval evaluation works and is tested. Answer generation, abstention checks, faithfulness and answer relevance are experimental: they are tested only against a scripted fake model and have not yet been run against a real API, and the judges are not validated against human labels. See the [roadmap](#roadmap).

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

## Answer generation (experimental)

Add a `generation` block to a config to answer every question from the top `k` chunks of each experiment:

```yaml
generation:
  generator:
    provider: anthropic          # pip install -e ".[anthropic]"
    model: claude-haiku-5-5      # any model ID
    # effort: low                # optional; omitted means the model's default
    # price_per_mtok: {input_per_mtok: ..., output_per_mtok: ...}   # without it, cost is n/a
```

The model returns `{"abstained": bool, "answer": str}`. For each question, the JSON output records the answer, whether the model abstained, the context spans it saw, the model, the prompt version, token usage, latency, and cost.

So far this measures one thing: **does the model abstain when it should?**

- The rates use the test set's labels. Questions with no evidence are *labeled unanswerable from this corpus*, and abstaining on them is correct.
- Abstaining on an answerable question counts as a false abstention, even when retrieval missed the evidence. The report splits these by whether gold evidence was in the top-k chunks actually sent to the model (`evidence_in_context`). An abstention without the evidence points at retrieval more than at the model. Evidence being in context says what the model could see, not that its answer is right.
- Refusals, responses cut off at `max_tokens`, invalid JSON, and API errors are failed generations. They never enter a rate, and one failed question does not stop the others. An error that fails every request (credentials, permission, unknown model) stops the run, but the answers produced so far are still printed and saved.

**Answers are not checked for correctness yet.** A fluent answer here is not evidence that it is right. The run calls a paid API, and the SDK reads credentials from the environment; ragtriage never reads or prints them.

**Cost is an estimate.** It uses the token counts the provider reports, including for failed answers, which are still billed. It leaves out calls that failed without reporting usage (counted separately as unknown) and any retries the SDK made before a final response. Haiku's higher price for prompts over 100,000 tokens and prompt-cache pricing are not modeled.

### Faithfulness (experimental)

Add a `judge` to the `generation` block to check whether each answer's claims are supported by the chunks the generator saw:

```yaml
generation:
  generator: {provider: anthropic, model: claude-haiku-5-5}
  judge: {provider: anthropic, model: claude-sonnet-5-5}   # optional; a different model is safer
```

The judge makes two calls per answer, with versioned Turkish prompts:

1. **Claims.** It splits the answer into atomic claims. Code gives each claim a stable id (`c1`, `c2`, ...). The prompt asks the judge not to add information that is not in the answer, but code cannot fully guarantee that.
2. **Verdicts.** It labels each claim `supported`, `unsupported` or `uncertain` against the passages. A `supported` verdict must name a passage that was shown and quote it. Code checks that the passage exists and that the quote appears in it verbatim (whitespace aside). **A quote being present does not prove the passage supports the claim's meaning.**

Scores:

| Situation | Result |
|---|---|
| Every claim has a definite verdict | supported / total claims |
| Any claim is `uncertain` | not applicable (`uncertain_claims`); counts are kept |
| Abstained, empty, or no claims | not applicable, never 1.0 |
| API error, invalid output, a missing or duplicate claim id, or a quote that fails the check | failed, never averaged in as 0 |
| One stage failed | no partial score; the usage of finished calls is kept |

The judge sees the question, the answer and the top-k chunks, never the evidence labels or reference answers. So it also judges answers whose retrieval missed the evidence. Judge usage and cost are reported separately from the generator's.

**Faithfulness measures support by the given context only.** It does not measure correctness, completeness or relevance, and the judge has not been validated against human labels.

### Answer relevance (experimental)

Add `relevance` to the `generation` block. It is independent of the faithfulness judge, and the embedder must be named explicitly:

```yaml
generation:
  generator: {provider: anthropic, model: claude-haiku-5-5}
  relevance:
    judge: {provider: anthropic, model: claude-sonnet-5-5}
    embedder:
      type: sentence_transformers   # downloads on first use; hashing is a non-semantic baseline
      model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

How it works:

1. A judge writes three questions that the answer could answer. It sees **only the answer**: not the original question (showing it would inflate the score), not the passages, the labels or the reference answer.
2. Code checks for exactly three non-empty questions that differ after normalization (Turkish casing, whitespace, trailing punctuation). Invalid output fails the answer, with no automatic retry.
3. The original question and the three questions all go through the embedder's `embed_query`. The score is the mean of the three cosine similarities.

The score lies in **[-1, 1]**: negative values are kept, not clipped or rescaled. Embeddings that are zero, contain NaN or infinity, or differ in size fail the answer. Abstained, empty and failed answers are not applicable. A faithfulness failure does not skip relevance, and the other way round. Judge usage and cost are reported separately from the generator's and from faithfulness.

**This is a semantic-similarity proxy.** It does not measure correctness or completeness, and a high score does not mean the question was fully answered. Embeddings are not cached. If a judge uses the generator's model, the CLI warns that it may favor its own model's answers; a different model may reduce this bias but does not remove it.

### Response cache

Add `cache` to the `generation` block to keep validated answers on disk and reuse them on the next run:

```yaml
generation:
  generator: {provider: anthropic, model: claude-haiku-5-5}
  cache:
    enabled: true                          # false, or no cache block, means every call is made
    path: .ragtriage_cache/responses.sqlite   # relative to the config file
```

- The key covers the provider, model, settings that change the response (such as `effort`), system and user messages, `max_tokens`, output schema, and prompt version. Credentials are never part of it.
- Only answers that pass validation are stored. Refusals, truncated or invalid output, and API errors are not, so they are retried on the next run.
- A cache hit adds no new tokens or cost. The report keeps the original usage separately (`cached_original_usage`), and a response that came from the scripted test model stays marked as fake.
- An unreadable, corrupt or incompatible cache file stops the run with an error. Nothing is deleted, and the run never falls back to paid calls on its own.
- **The cache stores questions, passages and answers as plain text.** `.ragtriage_cache/` is git-ignored; keep it out of shared folders if the documents are sensitive.

**Documents are untrusted input.** The prompt marks passages as data and escapes them, so a passage cannot close its own element or pose as the question. This makes instructions hidden in a document easier for the model to ignore. It does not guarantee that the model ignores them.

## Limitations

- **Larger chunks match more easily.** They are more likely to cover the evidence, which can flatter recall. Compare the `Context chars` column alongside it.
- **The 0.5 overlap threshold is a choice.** Change it with `min_overlap` and check whether your conclusions hold.
- **No significance testing yet.** Small differences between configurations may be noise.
- **Chunk sizes are in characters, not tokens.** The `fixed` chunker can also cut through a word.
- **CI does not run a real embedding model.** It tests the adapter against a stub to avoid the download. The results above come from a manual run.
- **The `hashing` embedder has no notion of meaning.** It matches similar spelling and cannot connect synonyms, so it is only a baseline.
- **Answer quality is only partly measured.** Abstention, faithfulness and an answer-relevance proxy are; correctness is not. Neither judge is validated against human labels yet.
- **Abstention depends on the labels.** "Unanswerable" means no evidence was labeled in this corpus, not that no answer exists anywhere.
- **Generation has not been run against a real API yet.** Every generation, judge and cache test uses a scripted fake model and a stub embedder, so there are no real generation numbers in this README.
- **Generation runs one question at a time.** With the cache off, re-running repeats every paid call.

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
├── outcomes.py           # Scored / NotApplicable / Failed
├── generation/           # LLM interface, FakeLLM, generator, Anthropic adapter, cache, judges
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
- [x] Answer generation behind a provider-independent interface, with abstention rates
- [x] Response cache and a faithfulness judge with verified support quotes
- [x] Answer relevance from answer-only questions and embedding similarity
- [ ] Concurrent calls and rate limiting
- [ ] HTML report with the worst examples and significance testing
- [ ] Synthetic test sets, including unanswerable questions
- [ ] Judge validation against human labels, and a real Turkish case study

## License

[MIT](LICENSE)
