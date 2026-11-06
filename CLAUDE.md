# ragtriage

An open-source tool that measures RAG pipelines, compares configurations from a YAML file, and shows where failures come from (retrieval, ranking, or generation). Turkish text is a first-class concern. It is a portfolio project, so clarity, honest limitations, and tests matter as much as features.

## Design decisions to keep

- **Gold evidence is a character span** (`Evidence(doc_id, start, end)`), never a chunk id, so one test set stays valid across chunk sizes. Test sets cite a `quote`; a missing or ambiguous quote is an error.
- **No evidence means "labeled unanswerable from this corpus."** Retrieval metrics skip these questions. Abstention metrics rely on this label.
- **Retrieval failure types** (`ok` / `ranking` / `retrieval`) need `depth > k`. Only the top `k` chunks go to the generator; deeper results are for retrieval triage.
- **Generation results are recorded separately** from retrieval failure types, so problems with an answer can be recorded even when retrieval failed.
- **Every measurement is `Scored`, `NotApplicable`, or `Failed`** (`ragtriage/outcomes.py`). API errors, refusals, truncation and invalid output are failures. They are counted and never averaged in as a score.
- **Structured output is validated, not trusted.** `abstained=false` with a blank answer is invalid output.
- **Retries belong to the provider SDK only** (`max_retries`). No second retry loop.
- **Provider-specific knobs stay in the provider.** `LLMRequest` carries only what every provider needs. The Anthropic adapter sends no `temperature`/`top_p`/`top_k` because current Claude models reject them; another provider may support them in its own config. Do not infer capabilities from model names.
- **Usage accounting:** a `Completion` has `source: api | cache | fake`. Only `api` is billed usage. Failed answers keep their reported usage and cost. Usage missing after an API error is unknown (`None`), never zero.
- **Prices are never built in.** Cost is `null` unless the config gives `price_per_mtok`, and it is an estimate (no visibility into SDK retries).
- **A fatal error keeps partial results.** The report gets `stopped`; the CLI prints and saves what finished, then exits with code 2. Unexpected exceptions are bugs and propagate. Clients are closed in `run_generation`.
- **Response cache** (`generation/cache.py`): wraps any `LLMClient`; written only after validation (`remember`); the key includes provider, model, `response_settings()`, and the whole request including `prompt_version`. A bad cache file raises `CacheError`: never delete it automatically or fall back to paid calls.
- **Faithfulness judge** (`generation/judge.py`): two calls (claims, verdicts), each validated before use or caching. Claim ids are assigned by code. `supported` needs a shown passage and a verbatim quote; a found quote is not proof of meaning. Any `uncertain` → NotApplicable; any stage failure → Failed with no partial score. The judge never sees labels or reference answers.
- **Answer relevance** (`generation/relevance.py`): the judge sees only the answer and writes exactly three distinct questions; all embeddings go through `embed_query`; the score is the mean cosine in [-1, 1], never clipped. Bad embeddings → Failed. The embedder must be configured explicitly. Every judge stage goes through `run_judge_stage` in `judge.py`.
- **Documents are untrusted.** Passages go inside escaped `<passage>` elements under a system prompt that treats them as data. This is mitigation, not protection.
- **Answers are not checked for correctness**, and the LLM judges are not yet validated against human labels. Never present either as proof of correctness.
- Optional dependencies stay optional (`[embeddings]`, `[anthropic]`). CI never calls an API or downloads a model.

## Checks

```bash
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
.venv\Scripts\mypy
```

Setup: `python -m venv .venv`, then `.venv\Scripts\python -m pip install -e ".[dev]"`.

## Working rules

- Chat with the user in Turkish. Code, comments, docs and commit messages in English.
- Do not use code, data or customer information from employers or other private projects, or their research-specific methods.
- Never read, print or commit API keys. Code gets credentials from the environment through the SDK.
- Ask before any paid API call or large download, with the expected cost or size.
- Commit or push only when the user asks. Work on `main` unless told otherwise.
- Numbers in the README must come from a real run. Small, self-written test sets are demos, not benchmarks; say so.
