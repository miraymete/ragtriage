"""Download XQuAD-tr at a pinned commit, check it, convert it, and split it by article.

    python examples/xquad_tr/prepare.py

Raw and converted data go to data/xquad_tr/, which git ignores. The source, hash, license and
split settings come from manifest.json next to this file. The split it produces is written to
split.json; running the script again must reproduce that file exactly, or it stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from ragtriage.converters.squad import (
    check_leakage,
    convert_squad,
    issue_counts,
    pilot_articles,
    split_articles,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, target: Path, expected_sha256: str) -> None:
    if target.exists():
        print(f"Using existing {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
    digest = _sha256(target)
    if digest != expected_sha256:
        raise SystemExit(f"SHA-256 mismatch for {target}: {digest} != {expected_sha256}")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "xquad_tr")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    source, plan = manifest["source"], manifest["split"]

    raw_path = args.data / "raw" / source["file"]
    _download(source["url"], raw_path, source["sha256"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    conversion = convert_squad(raw)

    titles = sorted(set(conversion.article_of_document.values()))
    split = split_articles(titles, plan["seed"], plan["n_dev_articles"])
    dev_titles = [t for t in titles if split[t] == "dev"]
    pilot = pilot_articles(dev_titles, plan["seed"], plan["n_pilot_articles"])
    leakage = check_leakage(conversion, split)
    if not leakage.clean:
        raise SystemExit(f"split leaks between dev and test: {leakage}")

    split_record = {
        "seed": plan["seed"],
        "dev": dev_titles,
        "test": [t for t in titles if split[t] == "test"],
        "pilot": pilot,
    }
    split_file = HERE / "split.json"
    if split_file.exists():
        if json.loads(split_file.read_text(encoding="utf-8")) != split_record:
            raise SystemExit(f"{split_file} does not match this run; the split is not reproducible")
    else:
        _dump(split_file, split_record)

    meta = {m.id: m for m in conversion.meta}
    counts: dict[str, Any] = {}
    for side in ("dev", "test"):
        docs_dir = args.data / side / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        for old in docs_dir.glob("*.txt"):
            old.unlink()
        side_docs = [
            d for d, a in sorted(conversion.article_of_document.items()) if split[a] == side
        ]
        for doc_id in side_docs:
            # newline="" writes the text exactly, so offsets stay valid.
            with (docs_dir / f"{doc_id}.txt").open("w", encoding="utf-8", newline="") as handle:
                handle.write(conversion.documents[doc_id].text)
        side_examples = [e for e in conversion.examples if split[meta[e.id].article] == side]
        rows = [
            {
                "id": e.id,
                "question": e.question,
                "reference_answer": e.reference_answer,
                "group": e.group,
                "evidence": [
                    {"doc_id": v.doc_id, "start": v.start, "end": v.end} for v in e.evidence
                ],
            }
            for e in side_examples
        ]
        _jsonl(args.data / side / "qa.jsonl", rows)
        _jsonl(
            args.data / side / "meta.jsonl",
            [meta[e.id].model_dump(mode="json") for e in side_examples],
        )
        counts[side] = {
            "articles": len({meta[e.id].article for e in side_examples}),
            "paragraphs": len(side_docs),
            "questions": len(side_examples),
        }
        if side == "dev":
            pilot_rows = [
                row
                for row, e in zip(rows, side_examples, strict=True)
                if meta[e.id].article in pilot
            ]
            _jsonl(args.data / "dev" / "pilot_qa.jsonl", pilot_rows)
            counts["pilot"] = {
                "articles": len(pilot),
                "questions": len(pilot_rows),
                "corpus_paragraphs": len(side_docs),
            }

    report = {
        "source_sha256": source["sha256"],
        "input": {
            "articles": len(titles),
            "paragraphs": len(conversion.documents),
            "questions": sum(len(p["qas"]) for a in raw["data"] for p in a["paragraphs"]),
            "answers_per_question": {
                str(n): c
                for n, c in sorted(
                    _count(
                        len(q["answers"])
                        for a in raw["data"]
                        for p in a["paragraphs"]
                        for q in p["qas"]
                    ).items()
                )
            },
        },
        "converted_questions": len(conversion.examples),
        "questions_with_alternatives": sum(bool(m.alternatives) for m in conversion.meta),
        "issues": issue_counts(conversion.issues),
        "issue_records": [i.model_dump(mode="json") for i in conversion.issues],
        "answer_text_repeated_in_paragraph": conversion.n_repeated_answer_text,
        "split": counts,
        "leakage": {
            "articles_in_both": list(leakage.articles_in_both),
            "questions_in_both": list(leakage.questions_in_both),
            "duplicate_paragraph_texts_across_split": [
                list(p) for p in leakage.duplicate_paragraph_texts
            ],
            "duplicate_question_texts_across_split": [
                list(p) for p in leakage.duplicate_question_texts
            ],
        },
    }
    _dump(args.data / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "issue_records"}, ensure_ascii=False, indent=2
        )
    )
    return 0


def _count(values: Any) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


if __name__ == "__main__":
    sys.exit(main())
