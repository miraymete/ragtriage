# XQuAD-tr retrieval pilot: plan

Written before any experiment was run. Changes after a run are added below with the reason, and
the original text is kept.

## Data

- XQuAD Turkish file at commit `7d30520c`, checked by SHA-256 (see `manifest.json`).
- One document per paragraph, text unchanged. Evidence is the first (and in this file the only)
  answer span. 48 articles, 240 paragraphs, 1,190 questions.
- Split by article with seed `20261107`: 24 dev and 24 test articles (`split.json`). Each side
  searches its own 120-paragraph corpus, so dev and test have corpora of the same size.
- The pilot is 6 dev articles chosen by the same seeded hash. Its questions search the **whole
  dev corpus** (120 paragraphs), never only their own paragraph.
- Test questions and test labels are not used for any choice below. Test metrics are not computed
  in this step.

## Fixed settings

k = 3, depth = 20, min_overlap = 0.5, BM25 k1 = 1.5, b = 0.75.

## Stage 1: tokenizer

Chunker fixed to whole paragraphs: `fixed`, size 4000, no overlap. The longest paragraph has
3,266 characters, so every paragraph is one chunk and chunking plays no part.

| Experiment | Tokenizer |
|---|---|
| paragraph, naive | `naive` |
| paragraph, turkish | `turkish` |
| paragraph, prefix | `turkish_prefix` |

## Stage 2: chunking

Tokenizer fixed to the stage 1 choice.

| Experiment | Chunker |
|---|---|
| paragraph | `fixed`, size 4000 |
| fixed 300 | `fixed`, size 300, overlap 50 |
| fixed 500 | `fixed`, size 500, overlap 50 |
| sentence 300 | `sentence`, max_chars 300 |
| recursive 300 | `recursive`, size 300, overlap 50 |

## Selection rule

1. The highest MRR on the pilot wins.
2. A gap under 0.01 MRR counts as a tie. Ties go to the higher recall@3.
3. Remaining ties go, in stage 1, to `turkish` (the project default), then `turkish_prefix`;
   in stage 2, to the smaller mean context chars@3.

This is a rule for picking a setting on a small pilot, not a significance test. The questions are
clustered in 6 articles, and the current intervals treat questions as independent, so no
interval or p-value is reported for the pilot.

## Before the test set is used

1. Confirm the choice on the full dev split (615 questions), with the same rule.
2. Add an article-level cluster bootstrap, or report the test result without intervals.
3. Run the test split once, with the chosen setting and the stage 1 baseline, and record the
   result whatever it is.

## Pilot results (2026-11-08)

Added after the runs. Nothing above was changed. 157 pilot questions from 6 dev articles,
searching the 120-paragraph dev corpus. Results are in `results/xquad_tr/` (not in git).

Stage 1 (whole paragraphs):

| Tokenizer | Recall@3 | MRR | nDCG@3 |
|---|---|---|---|
| naive | 0.962 | 0.886 | 0.903 |
| turkish | 0.962 | 0.880 | 0.898 |
| turkish_prefix | 0.981 | 0.933 | 0.943 |

`turkish_prefix` has the highest MRR, 0.047 above the next, so it is chosen.

Stage 2 (`turkish_prefix`):

| Chunker | Chunks | Recall@3 | MRR | nDCG@3 | Context chars@3 |
|---|---|---|---|---|---|
| paragraph | 120 | 0.981 | 0.933 | 0.943 | 2316 |
| fixed 300 | 433 | 0.866 | 0.762 | 0.775 | 839 |
| fixed 500 | 272 | 0.936 | 0.848 | 0.864 | 1307 |
| sentence 300 | 462 | 0.847 | 0.798 | 0.793 | 693 |
| recursive 300 | 539 | 0.854 | 0.750 | 0.760 | 628 |

`paragraph` has the highest MRR, 0.085 above the next, so it is chosen. It also sends about
three times as much text to the generator as the 300-character chunkers, and larger chunks cover
a short answer more easily. The rule only looks at MRR, so this cost is noted, not weighed.

Provisional dev setting: whole paragraphs with `turkish_prefix`. It still has to be confirmed on
the full dev split before the test set is used.

## Full dev validation: plan (added before any full dev run)

Written on 2026-11-08, after the pilot and before any experiment was run on the full dev split.
The sections above are unchanged. The test split is still closed.

- **Questions and corpus:** all 615 dev questions (24 articles), searching the same 120-paragraph
  dev corpus as the pilot.
- **Candidates:** the pilot lists, not extended. Stage 1 compares the three tokenizers on whole
  paragraphs (`dev_stage1.yaml`). Stage 2 compares the same five chunkers with the stage 1
  winner on full dev (`dev_stage2.yaml`). If full dev picks a tokenizer other than
  `turkish_prefix`, the stage 2 file gets that tokenizer and nothing else changes.
- **Selection rule:** the pilot rule above, unchanged. Highest MRR wins, and a gap under 0.01 MRR
  is a tie, broken by the higher recall@3, then by the stage-specific rule. **Added now, before the
  full dev run:** if anything is still tied, the lower mean context chars@3 wins, then the
  experiment that comes first in the config file.
- **Fixed baseline:** whole paragraphs with `naive` (`paragraph, naive`), whatever full dev shows.
- **Intervals:** each candidate gets a percentile bootstrap over **articles**. Whole articles are
  resampled with replacement, together with all their questions, and the mean is
  question-weighted, like the metric tables. For the chosen setting against the baseline, the
  difference (chosen − baseline) is bootstrapped over the same article draws. There is no
  article-level p-value. 95%, 10,000 resamples, seed 0.
- **Not evidence:** these dev intervals are not evidence about new data. The same dev questions
  pick the setting, and 24 articles are few.
- **Lock:** after full dev, the chosen setting and the baseline are written to `test_locked.yaml`
  and recorded below with the file's SHA-256, before the test split is opened.

## Full dev results and lock (2026-11-08)

Added after the full dev runs, before the test split was opened. 615 questions, 24 articles,
120-paragraph dev corpus. Intervals are 95% percentile bootstraps over articles (10,000
resamples, seed 0), question-weighted. These intervals come from the same questions that chose
the setting, so they are not evidence about new data.

Stage 1 (whole paragraphs):

| Tokenizer | Recall@3 | MRR [article CI] | nDCG@3 | Context chars@3 |
|---|---|---|---|---|
| naive | 0.924 | 0.867 [0.828, 0.904] | 0.876 | 2493 |
| turkish | 0.924 | 0.865 [0.827, 0.902] | 0.874 | 2497 |
| turkish_prefix | 0.961 | 0.935 [0.910, 0.957] | 0.937 | 2547 |

`turkish_prefix` has the highest MRR, 0.068 above the next. No tie, so no tie-break was needed.

Stage 2 (`turkish_prefix`):

| Chunker | Chunks | Recall@3 | MRR [article CI] | nDCG@3 | Context chars@3 |
|---|---|---|---|---|---|
| paragraph | 120 | 0.961 | 0.935 [0.910, 0.957] | 0.937 | 2547 |
| fixed 300 | 433 | 0.880 | 0.797 [0.763, 0.831] | 0.807 | 833 |
| fixed 500 | 272 | 0.930 | 0.861 [0.833, 0.890] | 0.873 | 1278 |
| sentence 300 | 462 | 0.881 | 0.814 [0.777, 0.849] | 0.822 | 664 |
| recursive 300 | 539 | 0.857 | 0.774 [0.736, 0.814] | 0.783 | 646 |

`paragraph` has the highest MRR, 0.074 above the next. No tie. Full dev picks the same setting as
the pilot.

Chosen against the fixed baseline, same article draws (chosen − baseline):

| Metric | Baseline | Chosen | Difference [article CI] |
|---|---|---|---|
| recall@3 | 0.924 | 0.961 | +0.037 [0.015, 0.063] |
| MRR | 0.867 | 0.935 | +0.068 [0.042, 0.097] |
| nDCG@3 | 0.876 | 0.937 | +0.061 [0.035, 0.090] |

The run also prints question-level intervals and p-values. They treat questions as independent,
and they are not part of the article-level result above.

### Lock

- **Chosen:** `paragraph, prefix`, which is whole paragraphs (`fixed`, size 4000) with BM25
  `turkish_prefix`.
- **Baseline:** `paragraph, naive`, which is whole paragraphs with BM25 `naive`.
- **Settings:** k = 3, depth = 20, min_overlap = 0.5.
- **Locked file:** `test_locked.yaml`, SHA-256
  `cb97dcb323ff2f099bf339d630fa128b928be8c4b09582eca08d9cbc11f086a3` as written on
  2026-11-08T13:36Z, with LF line endings. Its experiments are identical to `dev_locked.yaml`;
  only the data paths differ.
- **Status:** not yet run. Test metrics have not been computed or looked at.

Test command, to be run once:

    ragtriage run examples/xquad_tr/test_locked.yaml --group-stats \
      --baseline "paragraph, naive" --candidate "paragraph, prefix" \
      --out results/xquad_tr/test_locked.json --html results/xquad_tr/test_locked.html

## Test result (2026-11-08, run once)

Run at 2026-11-08T14:25Z with the command above, plus `--seed 0 --confidence 0.95
--resamples 10000 --failures`. Exit code 0. Output is in `results/xquad_tr/test_locked.json`
and `.html` (not in git). Recorded whatever the outcome; the configuration was not changed
afterwards and the test split was not run again.

Checks run before it:

- the SHA-256 of `test_locked.yaml` matched the hash above;
- its experiments matched `dev_locked.yaml`;
- `fixed` 4000 made each of the 120 dev and 120 test paragraphs exactly one whole chunk (longest
  test paragraph: 1,631 characters);
- no test output existed yet.

Test split: 24 articles, 120 paragraphs, 575 questions, 120 chunks per experiment.

| Experiment | Recall@3 | MRR | nDCG@3 | Context chars@3 | OK / ranking / retrieval |
|---|---|---|---|---|---|
| paragraph, naive (baseline) | 0.934 | 0.904 | 0.906 | 2217 | 537 / 23 / 15 |
| paragraph, prefix (chosen) | 0.979 | 0.957 | 0.961 | 2285 | 563 / 12 / 0 |

Chosen − baseline. The interval is a paired percentile bootstrap over articles (24 groups,
question-weighted, 95%, 10,000 resamples, seed 0):

| Metric | Difference | Article-level 95% CI |
|---|---|---|
| recall@3 | +0.045 | [0.030, 0.062] |
| MRR | +0.054 | [0.038, 0.069] |
| nDCG@3 | +0.054 | [0.039, 0.070] |

The question-level output from the same run (MRR CI [0.036, 0.072], unadjusted p printed as
1.0e-05) treats questions as independent. It is not part of this result. The article-level
interval here is not wider than the question-level one. That alone does not show that the
questions are independent: sampling variation and 24 groups also shape it.

Failure review (read after the run; nothing was changed because of it):

- 28 questions fail only with `naive`, 2 only with `turkish_prefix`, and 10 with both.
- Every `naive` retrieval failure (15) is gone with the prefix. Some `naive` queries match no
  paragraph at all: "Kolonilerde nüfuslar nerede merkezlenmiştir?" shares no exact word form
  with its paragraph, while the prefix tokenizer puts it at rank 2.
- The prefix's 12 failures are all ranking failures, spread over 9 articles. Two examples:
  - A question about how French Protestants "changed their beliefs" ranks the gold Huguenot
    paragraph 4th, behind other Huguenot paragraphs. Prefixes match more paragraphs of the same
    article, which can push the right one down.
  - "Yapılarda zorlanma nedenleri nelerdir?" (answer: "Gerilme tensörü") ranks the gold paragraph
    17th. The question and the answer share few words.
- To look at these, the retrieval of 4 questions was recomputed with the locked settings. This was
  for inspection only and produced no new metric.

Limitations: the text is translated, not native Turkish. The evidence is a short answer span
that a whole paragraph covers easily. Each split searches only 120 paragraphs. The splits have
24 articles each. The questions were written while reading the paragraph, which favors BM25.
These numbers describe this setup, not Turkish RAG in general.

## Notes added after the test (2026-11-08)

Wording notes only. No setting, metric or earlier text was changed.

- **"About three times as much text"** (pilot and full dev, stage 2) compares whole paragraphs
  with the 300-character chunkers. The two locked test settings both use whole paragraphs, and
  their context chars@3 differ by only about 3% (2285 against 2217). Only the tokenizer differs
  between them.
- **The four recomputed questions** were for reading failures only. No setting, no aggregate
  metric and no saved output was changed.
- **Known limitation:** `results/xquad_tr/test_locked.html` is 21.5 MB, because the report writes
  every retrieved paragraph for every question. The report was not optimized in this step.
