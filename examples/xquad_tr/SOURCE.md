# XQuAD-tr case study: source, license and how to rebuild it

## Source and license

- **Data:** the Turkish file of [XQuAD](https://github.com/google-deepmind/xquad), `xquad.tr.json`,
  at commit `7d30520c717524000f0d9d2f9c10a069acd9d285`. SHA-256 and size are in `manifest.json`.
- **License:** CC BY-SA 4.0. The repository README at that commit says: "This dataset is
  distributed under the CC BY-SA 4.0 license."
- **Attribution:** Artetxe, M., Ruder, S., & Yogatama, D. (2019). *On the cross-lingual
  transferability of monolingual representations.* arXiv:1910.11856. XQuAD is a professional
  translation of 240 paragraphs and 1,190 questions from the SQuAD v1.1 development set
  (Rajpurkar et al., 2016).
- **Not in this repository:** the raw file and the converted documents and questions. They are
  rebuilt locally in `data/xquad_tr/`, which git ignores. This repository holds only the
  converter, this note, the manifest, the split, the plan and the configs. Converted data that you
  share yourself stays under CC BY-SA 4.0 and needs the attribution above.

## Rebuild

```bash
python examples/xquad_tr/prepare.py      # downloads about 712 KB once, checks SHA-256
ragtriage run examples/xquad_tr/pilot_tokenizers.yaml --out results/xquad_tr/pilot_tokenizers.json --html results/xquad_tr/pilot_tokenizers.html
ragtriage run examples/xquad_tr/pilot_chunking.yaml --out results/xquad_tr/pilot_chunking.json --html results/xquad_tr/pilot_chunking.html

# Full dev, with article-level intervals
ragtriage run examples/xquad_tr/dev_stage1.yaml --group-stats --out results/xquad_tr/dev_stage1.json --html results/xquad_tr/dev_stage1.html
ragtriage run examples/xquad_tr/dev_stage2.yaml --group-stats --out results/xquad_tr/dev_stage2.json --html results/xquad_tr/dev_stage2.html
ragtriage run examples/xquad_tr/dev_locked.yaml --group-stats --baseline "paragraph, naive" --candidate "paragraph, prefix" --out results/xquad_tr/dev_locked.json --html results/xquad_tr/dev_locked.html

# The locked test run (done once, on 2026-11-08; see PLAN.md)
ragtriage run examples/xquad_tr/test_locked.yaml --group-stats --seed 0 --confidence 0.95 --resamples 10000 --baseline "paragraph, naive" --candidate "paragraph, prefix" --out results/xquad_tr/test_locked.json --html results/xquad_tr/test_locked.html
```

The locked file's SHA-256 is recorded in `PLAN.md`. Check it before rerunning, so the result can be
compared with the recorded one. Running the test again does not choose anything new.

`prepare.py` stops if the hash does not match, or if the split it computes differs from
`split.json`. It writes `data/xquad_tr/report.json` with the conversion counts, every dropped
record and its reason, and the leakage check.

## How the data is converted

- **Documents:** one per paragraph, with the text exactly as in the file. The id is the article
  title made file-safe, plus the paragraph index (`Amazon_rainforest__p03`). The original title
  is kept in `meta.jsonl`.
- **Evidence:** `answer_start` is used as given and must point at the answer text exactly.
  Nothing is normalized or repaired. An answer that does not match is reported and left out.
- **Alternative answers:** only the first answer becomes evidence. SQuAD's extra answers are
  alternatives from other annotators, not extra evidence that is all required, so they go to
  `meta.jsonl` and no metric uses them. This Turkish file has one answer per question, so there
  are none.
- **Question ids:** kept from the source.
- **Split:** whole articles go to dev or test, ordered by a seeded SHA-256 of the title: 24 and 24
  articles, 120 paragraphs each. Each side searches its own corpus. No article, paragraph or
  question id is on both sides, and no paragraph or question text repeats across them (after
  case and whitespace folding).
- **Pilot:** 6 dev articles, by the same seeded hash. Its questions search the whole dev corpus.

## Limitations

- **Translated, not native Turkish.** The text is a professional translation of English
  Wikipedia paragraphs. It does not show how people write Turkish, or the topics a Turkish
  corpus would cover.
- **A short answer span is not enough context.** Evidence is the answer phrase, about three
  tokens on average. A chunk that contains it may still not contain what is needed to answer the
  question, and with `min_overlap` a short span is easy to cover. Larger chunks benefit most.
- **No unanswerable or multi-evidence questions.** Abstention cannot be measured here, and recall
  is effectively a hit rate.
- **Questions written while reading the paragraph.** They share many words with it, which favors
  lexical retrieval such as BM25. Some questions are vague outside their paragraph.
- **Small corpus.** Each split searches 120 paragraphs, five per article. That is far smaller and
  more on-topic than a real collection.
- **Questions are clustered.** Questions from the same paragraph or article are not independent.
  ragtriage's current intervals treat them as independent, so they would be too narrow here; no
  interval or p-value is reported for the pilot.
- **Not a benchmark.** These results describe this translated set and this corpus size only.
