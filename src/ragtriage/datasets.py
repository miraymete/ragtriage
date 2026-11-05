"""Load documents and test sets from disk."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragtriage.models import Document, Evidence, QAExample


def load_documents(directory: str | Path) -> dict[str, Document]:
    """Read every .txt file in a directory. The file stem becomes the document id."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise ValueError(f"no .txt files in {directory}")
    documents = {}
    for path in paths:
        # newline="" keeps the bytes as they are: translating line endings would shift every span.
        with path.open(encoding="utf-8", newline="") as handle:
            documents[path.stem] = Document(id=path.stem, text=handle.read())
    return documents


def resolve_quote(document: Document, quote: str, grade: int = 1) -> Evidence:
    """Turn a quote into a span. Offsets are error-prone by hand, so test sets cite text instead.

    An ambiguous quote is an error, because guessing could silently label the wrong passage.
    """
    start = document.text.find(quote)
    if start == -1:
        raise ValueError(f"quote not found in {document.id!r}: {quote!r}")
    if document.text.find(quote, start + 1) != -1:
        raise ValueError(f"quote appears more than once in {document.id!r}: {quote!r}")
    return Evidence(doc_id=document.id, start=start, end=start + len(quote), grade=grade)


def _parse_evidence(raw: dict[str, Any], documents: Mapping[str, Document]) -> Evidence:
    doc_id = raw["doc_id"]
    if doc_id not in documents:
        raise ValueError(f"unknown document {doc_id!r}")
    if "quote" in raw:
        return resolve_quote(documents[doc_id], raw["quote"], raw.get("grade", 1))
    evidence = Evidence.model_validate(raw)
    if evidence.end > len(documents[doc_id].text):
        raise ValueError(f"span [{evidence.start}, {evidence.end}) runs past end of {doc_id!r}")
    return evidence


def load_qa_examples(path: str | Path, documents: Mapping[str, Document]) -> list[QAExample]:
    """Read a JSONL test set. Each evidence item has `doc_id` plus `quote` or `start`/`end`."""
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                raw["evidence"] = [_parse_evidence(e, documents) for e in raw.get("evidence", [])]
                examples.append(QAExample.model_validate(raw))
            except (ValueError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return examples
