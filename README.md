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

## Uncertainty and paired comparison

Both are optional. Without these flags the output is unchanged.

```bash
ragtriage run examples/turkish_quickstart/config.yaml --stats \
  --baseline "fixed 300, naive" --candidate "fixed 300, prefix"
```

- `--stats` adds a question-level **percentile bootstrap** interval for the mean of recall@k, reciprocal rank (whose mean is MRR) and nDCG@k in every experiment. Questions are resampled with replacement.
- `--baseline A --candidate B` compares **two experiments you choose**, question by question. The difference is always **B − A**.
  - The interval comes from a **paired bootstrap**: one resample picks the same questions on both sides.
  - The p-value comes from a **two-sided paired permutation test** that flips the sign of each question's difference. Patterns exactly as extreme as the observed one count. Zero differences are left out of the flipping because flipping does not change them.
  - With up to 20 nonzero differences, every sign pattern is enumerated (**exact**, p = extreme / 2^m). Above that, random patterns are drawn (**Monte Carlo**, p = (extreme + 1) / (draws + 1)). The output says which one was used.
- Settings: `--confidence` (0.95), `--resamples` (10000), `--permutations` (100000 Monte Carlo draws) and `--seed` (0). The seed, the method, the confidence level and the number of resamples are saved in the JSON, so the same command gives the same numbers.
- In the JSON, each experiment gets `statistics`. The comparison is stored as `comparison` on the candidate's entry.
- `--group-stats` adds a **group-level (cluster) bootstrap** when every question has a `group` field, such as its source article. Whole groups are resampled with replacement, together with all their questions; a group drawn twice counts twice. The mean stays question-weighted, like the tables. With `--baseline` and `--candidate` it also bootstraps B − A over the same group draws. It gives intervals only, no p-value, and it is reported apart from the question-level method. The JSON fields are `group_statistics` and `group_comparison`. It allows for questions within a group being related. It does not help when there are few groups, or when the groups do not represent real use.
- The two methods are printed in separate tables, and the HTML report gives them separate sections. A question-level p-value is not part of a group-level interval, and the two can disagree.

How the edge cases are handled:

- Questions labeled unanswerable are excluded, and the output says how many were used and how many were excluded.
- Questions are matched by id. If the two experiments scored different, missing or duplicate question ids, that is an error. The run never quietly falls back to the questions both have in common.
- There is no result for zero questions. For one question there is a mean but no interval or p-value.
- If every value or every difference is the same, the interval has zero width and is labeled (`constant`, `constant_difference`, `all_zero`). That describes this sample, not certainty about new questions. All-zero differences give p = 1.

### Reading the demo

From the demo run above (21 scored questions, 2 unanswerable excluded, seed 0):

| Metric | naive | prefix | prefix − naive | 95% CI | Higher / lower / tied | Unadjusted p |
|---|---|---|---|---|---|---|
| recall@3 | 0.833 | 1.000 | +0.167 | [0.024, 0.333] | 4 / 0 / 17 | 0.125 (exact, 2/16) |
| MRR | 0.786 | 0.952 | +0.167 | [0.048, 0.286] | 6 / 0 / 15 | 0.031 (exact, 2/64) |
| nDCG@3 | 0.786 | 0.965 | +0.179 | [0.066, 0.305] | 7 / 0 / 14 | 0.016 (exact, 2/128) |

- **The two methods can disagree on few questions.** For recall the bootstrap interval excludes zero, but p = 0.125. Only 4 questions differ, so there are only 16 sign patterns. The smallest p-value the test can reach is 2/16, even when all 4 changes go the same way. With so few differences, a percentile bootstrap interval is also likely to be too narrow. Read both numbers, and look at the question counts behind them.
- **These p-values are unadjusted.** This table already checks three metrics, and picking one pair out of six experiments is itself a choice. The more things you check, the more likely it is that one looks significant by luck. This first version does not correct for that, and it never tests every pair on its own.
- **The tool does not name a winner**, and it does not label results as equivalent or as a proven improvement. These are demo numbers from a small, self-written test set, not a benchmark.

**What the method assumes and cannot fix.** Intervals and p-values treat the questions as independent draws from the questions the system will really face. Questions written about the same document are not independent. The same author wrote these documents and questions, so the set is not a sample of real use. Resampling does not correct either problem. The result is about these questions, not about retrieval in general.

## HTML report

Add `--html` to write one self-contained page from the same run. Nothing is evaluated again, and the page loads no scripts, fonts or other files.

```bash
ragtriage run examples/turkish_quickstart/config.yaml --stats   --baseline "fixed 300, naive" --candidate "fixed 300, prefix" --html results/report.html
```

The page has:

- **Summary tables.**
  - The retrieval table for every experiment, with the number of questions scored and excluded.
  - Bootstrap intervals and the paired comparison, when they were computed. The page names the method (exact or Monte Carlo) and labels every p-value unadjusted.
  - Generation, faithfulness and relevance, when they ran. The page shows not applicable, failed and stopped results apart. It also shows generator and judge usage on separate rows, with cache and fake responses marked.
- **One collapsible block per question**, sorted by recall@k (lowest first), then reciprocal rank, then question id. Each block holds:
  - The gold evidence spans.
  - The top-k chunks, separate from the deeper results that are used only for triage. Each chunk shows its rank, `doc_id` and character range.
  - The context actually sent to the model.
  - Any answer and judge output.
- **Highlighted overlap.** Characters that overlap a gold span are highlighted in the chunk text, and a chunk is marked as *covering* the evidence only when it passes `min_overlap`.
- **Questions labeled unanswerable** are listed apart and never shown as retrieval failures. In a stopped run, questions without a result say so, and experiments that never started are marked.

All text from documents, questions, answers, model output and errors is HTML-escaped, including inside highlights. **The file stores questions, passages and answers as plain text**, so treat it like the documents themselves. Responses from the scripted test model are marked as fake. The page never names a winner.

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
- **Uncertainty estimates assume representative questions.** The question-level methods also assume independent questions. `--group-stats` resamples whole groups instead, but it needs enough groups, and it gives no p-value. p-values are not adjusted for multiple comparisons. Generation and judge results have no intervals yet.
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
├── metrics/statistics.py # bootstrap intervals and paired permutation tests
├── evaluate.py           # score runs against a test set
├── datasets.py           # load documents and JSONL test sets
├── config.py             # validated YAML experiment file
├── runner.py             # build pipelines and run every experiment
├── cli.py                # ragtriage run config.yaml
├── html_report.py        # single-file HTML report of one run
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
- [x] Bootstrap intervals and paired permutation tests for retrieval metrics
- [x] Single-file HTML report with per-question details
- [ ] Synthetic test sets, including unanswerable questions
- [ ] Judge validation against human labels, and a real Turkish case study

## License

[MIT](LICENSE)
