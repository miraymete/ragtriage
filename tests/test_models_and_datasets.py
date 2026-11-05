import json

import pytest
from pydantic import ValidationError

from ragtriage import Chunk, Document, Evidence, load_documents, load_qa_examples
from ragtriage.datasets import resolve_quote

DOCUMENT = Document(id="d", text="Ankara başkenttir. İzmir bir liman kentidir. Ankara büyüktür.")


@pytest.mark.parametrize(
    "spans", [{"start": 5, "end": 5}, {"start": 9, "end": 5}, {"start": -1, "end": 5}]
)
def test_evidence_rejects_bad_spans(spans):
    with pytest.raises(ValidationError):
        Evidence(doc_id="d", **spans)


def test_chunk_text_must_match_its_span():
    with pytest.raises(ValidationError):
        Chunk(doc_id="d", start=0, end=5, text="abc")


def test_resolve_quote_finds_offsets():
    evidence = resolve_quote(DOCUMENT, "İzmir bir liman kentidir")
    assert DOCUMENT.text[evidence.start : evidence.end] == "İzmir bir liman kentidir"


def test_resolve_quote_rejects_missing_and_ambiguous_text():
    with pytest.raises(ValueError, match="not found"):
        resolve_quote(DOCUMENT, "Bursa")
    with pytest.raises(ValueError, match="more than once"):
        resolve_quote(DOCUMENT, "Ankara")


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_qa_examples_handles_quotes_spans_and_unanswerable(tmp_path):
    path = tmp_path / "qa.jsonl"
    write_jsonl(
        path,
        [
            {"id": "a", "question": "?", "evidence": [{"doc_id": "d", "quote": "İzmir"}]},
            {"id": "b", "question": "?", "evidence": [{"doc_id": "d", "start": 0, "end": 6}]},
            {"id": "c", "question": "?"},
        ],
    )
    a, b, c = load_qa_examples(path, {"d": DOCUMENT})
    assert a.evidence[0].start == DOCUMENT.text.index("İzmir")
    assert b.evidence[0].end == 6
    assert not c.answerable


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ({"doc_id": "nope", "quote": "x"}, "unknown document"),
        ({"doc_id": "d", "quote": "Bursa"}, "not found"),
        ({"doc_id": "d", "start": 0, "end": 10_000}, "past end"),
    ],
)
def test_load_qa_examples_reports_the_failing_line(tmp_path, evidence, message):
    path = tmp_path / "qa.jsonl"
    write_jsonl(
        path,
        [{"id": "ok", "question": "?"}, {"id": "x", "question": "?", "evidence": [evidence]}],
    )
    with pytest.raises(ValueError, match=rf"qa\.jsonl:2:.*{message}"):
        load_qa_examples(path, {"d": DOCUMENT})


def test_load_documents_keeps_line_endings_so_offsets_stay_true(tmp_path):
    (tmp_path / "a.txt").write_bytes("satır\r\nikinci".encode())
    assert load_documents(tmp_path)["a"].text == "satır\r\nikinci"


def test_load_documents_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match=r"no \.txt files"):
        load_documents(tmp_path)
