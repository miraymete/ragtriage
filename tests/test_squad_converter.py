"""Small synthetic SQuAD files only: CI never downloads XQuAD."""

import pytest

from ragtriage.converters.squad import (
    check_leakage,
    convert_squad,
    document_id,
    issue_counts,
    pilot_articles,
    split_articles,
)
from ragtriage.datasets import load_documents, load_qa_examples
from ragtriage.models import Evidence

CONTEXT = ' İzmir & "Ege" kıyısında.\nİzmir bir liman kentidir. '


def qa(qid, answers, question="İzmir nedir?"):
    return {"id": qid, "question": question, "answers": answers}


def squad(*articles):
    return {"version": "1.1", "data": list(articles)}


def article(title, *paragraphs):
    return {
        "title": title,
        "paragraphs": [{"context": c, "qas": list(q)} for c, q in paragraphs],
    }


def at(text, context=CONTEXT, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = context.index(text, start + 1)
    return {"text": text, "answer_start": start}


def test_paragraph_text_is_unchanged_and_offsets_are_kept():
    answer = at("liman kenti")
    result = convert_squad(squad(article("Izmir_(city)", (CONTEXT, [qa("q-1", [answer])]))))
    doc_id = document_id("Izmir_(city)", 0)
    assert doc_id == "Izmir__city___p00"
    assert result.documents[doc_id].text == CONTEXT  # leading space, quotes, newline kept
    [example] = result.examples
    assert example.id == "q-1"
    assert example.evidence == (
        Evidence(doc_id=doc_id, start=answer["answer_start"], end=answer["answer_start"] + 11),
    )
    assert example.reference_answer == "liman kenti"
    assert example.group == "Izmir_(city)"
    assert result.meta[0].article == "Izmir_(city)"
    assert result.article_of_document == {doc_id: "Izmir_(city)"}
    assert result.issues == ()


def test_offsets_are_never_repaired():
    wrong = {"text": "liman", "answer_start": at("liman")["answer_start"] + 1}
    result = convert_squad(
        squad(article("A", (CONTEXT, [qa("q1", [wrong]), qa("q2", [at("Ege")])])))
    )
    assert [e.id for e in result.examples] == ["q2"]
    [issue] = result.issues
    assert (issue.question_id, issue.reason) == ("q1", "offset_mismatch")
    assert "found 'iman '" in issue.detail


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        ([], "no_answer"),
        ([{"text": "", "answer_start": 0}], "empty_answer"),
        ([{"text": "İzmir", "answer_start": "1"}], "bad_answer_start"),
        ([{"text": "İzmir", "answer_start": True}], "bad_answer_start"),
        ([{"text": "İzmir", "answer_start": -1}], "bad_answer_start"),
        (["İzmir"], "malformed_answer"),
    ],
)
def test_broken_records_are_reported_and_left_out(answers, reason):
    result = convert_squad(squad(article("A", (CONTEXT, [qa("q1", answers)]))))
    assert result.examples == ()
    assert issue_counts(result.issues) == {reason: 1}


def test_alternative_answers_are_metadata_not_extra_evidence():
    first, second = at("İzmir"), at("İzmir bir liman kentidir")
    broken = {"text": "Ankara", "answer_start": 0}
    result = convert_squad(
        squad(article("A", (CONTEXT, [qa("q1", [first, second, dict(first), broken])])))
    )
    [example] = result.examples
    # Only the first answer is evidence, so recall does not require every alternative.
    assert len(example.evidence) == 1
    assert example.evidence[0].start == first["answer_start"]
    [meta] = result.meta
    # The duplicate of the first answer is dropped; the broken one is reported.
    assert [(a.start, a.end) for a in meta.alternatives] == [
        (second["answer_start"], second["answer_start"] + len(second["text"]))
    ]
    assert issue_counts(result.issues) == {"alternative_offset_mismatch": 1}


def test_duplicate_ids_and_repeated_answer_text():
    result = convert_squad(
        squad(
            article("A", (CONTEXT, [qa("q1", [at("İzmir", occurrence=1)]), qa("q1", [at("Ege")])]))
        )
    )
    assert [e.id for e in result.examples] == ["q1"]
    assert issue_counts(result.issues) == {"duplicate_id": 1}
    # "İzmir" occurs twice; the offset picks the second one.
    assert result.n_repeated_answer_text == 1
    assert result.examples[0].evidence[0].start == CONTEXT.rindex("İzmir")


def test_structural_errors_raise():
    with pytest.raises(ValueError, match="'data' list"):
        convert_squad({"version": "1.1"})
    with pytest.raises(ValueError, match="collision"):
        convert_squad(squad(article("a/b", ("x", [])), article("a?b", ("y", []))))
    with pytest.raises(ValueError, match="duplicate article"):
        convert_squad(squad(article("A", ("x", [])), article("A", ("y", []))))


def test_documents_survive_a_round_trip_through_files(tmp_path):
    context = "Satır bir.\r\nSatır iki.\nİzmir."
    answer = {"text": "İzmir", "answer_start": context.index("İzmir")}
    result = convert_squad(squad(article("A", (context, [qa("q1", [answer])]))))
    for doc_id, document in result.documents.items():
        with (tmp_path / f"{doc_id}.txt").open("w", encoding="utf-8", newline="") as handle:
            handle.write(document.text)
    import json

    evidence = result.examples[0].evidence[0]
    row = {"id": "q1", "question": "?", "evidence": [evidence.model_dump(exclude={"grade"})]}
    (tmp_path / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    docs = load_documents(tmp_path)
    [loaded] = load_qa_examples(tmp_path / "qa.jsonl", docs)
    span = loaded.evidence[0]
    assert docs[span.doc_id].text == context
    assert docs[span.doc_id].text[span.start : span.end] == "İzmir"


# --- splits -----------------------------------------------------------------------------------

TITLES = [f"article_{i}" for i in range(10)]


def test_split_is_by_whole_article_deterministic_and_order_independent():
    first = split_articles(TITLES, "seed", 4)
    assert first == split_articles(list(reversed(TITLES)), "seed", 4)
    assert sorted(first) == sorted(TITLES)
    assert list(first.values()).count("dev") == 4
    assert split_articles(TITLES, "other", 4) != first
    with pytest.raises(ValueError):
        split_articles(TITLES, "seed", 0)
    with pytest.raises(ValueError):
        split_articles(TITLES, "seed", 10)


def test_pilot_comes_only_from_dev_and_is_reproducible():
    split = split_articles(TITLES, "seed", 6)
    dev = [t for t, s in split.items() if s == "dev"]
    pilot = pilot_articles(dev, "seed", 2)
    assert pilot == pilot_articles(list(reversed(dev)), "seed", 2)
    assert set(pilot) <= set(dev)
    assert len(pilot) == 2


def test_leakage_check_reports_duplicates_across_the_split():
    data = squad(
        article("A", ("Aynı paragraf.", [qa("a1", [{"text": "Aynı", "answer_start": 0}], "Ne?")])),
        article(
            "B", ("aynı  PARAGRAF.", [qa("b1", [{"text": "aynı", "answer_start": 0}], " ne? ")])
        ),
        article("C", ("Başka.", [qa("c1", [{"text": "Başka", "answer_start": 0}], "Başka?")])),
    )
    result = convert_squad(data)
    report = check_leakage(result, {"A": "dev", "B": "test", "C": "test"})
    assert report.clean  # no article or question id on both sides
    assert report.duplicate_paragraph_texts == (("A__p00", "B__p00"),)
    assert report.duplicate_question_texts == (("a1", "b1"),)
    same_side = check_leakage(result, {"A": "dev", "B": "dev", "C": "test"})
    assert same_side.duplicate_paragraph_texts == ()


def test_conversion_is_reproducible():
    data = squad(article("A", (CONTEXT, [qa("q1", [at("Ege")]), qa("q2", [at("liman")])])))
    assert convert_squad(data) == convert_squad(data)
