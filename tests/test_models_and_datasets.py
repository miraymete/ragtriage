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


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("[1]", "expected a JSON object"),
        ('{"id": "q", "question": "?", "evidence": "d"}', "'evidence' must be a list"),
        ('{"id": "q", "question": "?", "evidence": ["d"]}', "evidence must be a mapping"),
        ('{"id": "q", "question": "?", "evidence": [{"quote": "Ankara"}]}', "missing 'doc_id'"),
        (
            '{"id": "q", "question": "?", "evidence": [{"doc_id": "d", "quote": ""}]}',
            "quote must be a non-empty string",
        ),
        (
            '{"id": "q", "question": "?", "evidence": [{"doc_id": "d", "quote": 5}]}',
            "quote must be a non-empty string",
        ),
    ],
)
def test_malformed_test_set_lines_give_a_clear_error(tmp_path, line, message):
    path = tmp_path / "qa.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"qa\.jsonl:1:.*{message}"):
        load_qa_examples(path, {"d": DOCUMENT})


def test_duplicate_question_ids_are_rejected_with_both_line_numbers(tmp_path):
    path = tmp_path / "qa.jsonl"
    rows = [{"id": "q1", "question": "a?"}, {"id": "q2", "question": "b?"}]
    rows.append({"id": "q1", "question": "c?"})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"qa\.jsonl:3:.*duplicate question id 'q1'.*line 1"):
        load_qa_examples(path, {"d": DOCUMENT})


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"id": "", "question": "a?"}, "question id must not be blank"),
        ({"id": "  ", "question": "a?"}, "question id must not be blank"),
        ({"id": "q1", "question": ""}, "question text must not be blank"),
        ({"id": "q1", "question": " \t"}, "question text must not be blank"),
    ],
)
def test_blank_question_id_or_text_is_rejected_with_line_number(tmp_path, row, message):
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"(?s)qa\.jsonl:1:.*{message}"):
        load_qa_examples(path, {"d": DOCUMENT})


@pytest.mark.parametrize("content", ["", "\n  \n"])
def test_test_set_without_questions_is_rejected(tmp_path, content):
    path = tmp_path / "qa.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"qa\.jsonl: .*no questions"):
        load_qa_examples(path, {"d": DOCUMENT})


def test_load_documents_names_the_file_that_is_not_utf8(tmp_path):
    (tmp_path / "good.txt").write_text("İzmir", encoding="utf-8")
    (tmp_path / "bad.txt").write_bytes("İzmir".encode("cp1254") + b"\xff")
    with pytest.raises(ValueError, match=r"bad\.txt.*not valid UTF-8"):
        load_documents(tmp_path)
