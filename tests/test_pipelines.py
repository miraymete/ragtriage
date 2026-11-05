import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtriage import Chunk, Document
from ragtriage.pipelines.bm25 import BM25Retriever
from ragtriage.pipelines.chunking import chunk_fixed
from ragtriage.pipelines.text import (
    naive_tokens,
    turkish_lower,
    turkish_prefix_tokens,
    turkish_tokens,
)


def make_chunks(*texts):
    return [Chunk(doc_id=f"d{i}", start=0, end=len(t), text=t) for i, t in enumerate(texts)]


def test_chunk_fixed_windows():
    chunks = chunk_fixed(Document(id="d", text="abcdefghij"), size=4, overlap=1)
    assert [(c.start, c.end, c.text) for c in chunks] == [
        (0, 4, "abcd"),
        (3, 7, "defg"),
        (6, 10, "ghij"),
    ]


def test_chunk_fixed_has_no_trailing_fragment_when_text_ends_on_a_window():
    assert len(chunk_fixed(Document(id="d", text="abcd"), size=4, overlap=2)) == 1


def test_chunk_fixed_empty_document():
    assert chunk_fixed(Document(id="d", text=""), size=4) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_chunk_fixed_rejects_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_fixed(Document(id="d", text="abc"), size=size, overlap=overlap)


@given(
    st.text(max_size=200),
    st.integers(min_value=1, max_value=50),
    st.integers(min_value=0, max_value=49),
)
def test_chunks_cover_the_whole_text_and_match_their_offsets(text, size, overlap):
    overlap = min(overlap, size - 1)
    chunks = chunk_fixed(Document(id="d", text=text), size=size, overlap=overlap)
    covered = set()
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        covered.update(range(chunk.start, chunk.end))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("KIRMIZI", "kırmızı"),
        ("İZNİK", "iznik"),
        ("İstanbul", "istanbul"),
        ("Isparta", "ısparta"),
        ("ÇAĞŞÖÜ", "çağşöü"),
    ],
)
def test_turkish_lower(raw, expected):
    assert turkish_lower(raw) == expected


def test_turkish_lower_handles_decomposed_input():
    assert turkish_lower("İznik") == "iznik"


def test_default_lowercasing_breaks_turkish_which_is_why_the_tokenizer_exists():
    assert naive_tokens("İznik") == ["i", "znik"]
    assert naive_tokens("KIRMIZI") != naive_tokens("kırmızı")
    assert turkish_tokens("İznik") == turkish_tokens("iznik")
    assert turkish_tokens("KIRMIZI") == turkish_tokens("kırmızı")


def test_prefix_tokens_merge_inflected_forms():
    assert turkish_prefix_tokens("kuruldu kurulan") == ["kurul", "kurul"]


def test_bm25_score_matches_hand_calculation():
    # N=2, df=1 -> idf = ln(1 + 1.5/1.5) = 0.693147
    # avg length 2.5, length 3 -> norm = 0.25 + 0.75 * 3/2.5 = 1.15
    # tf=2 -> 2 * 2.5 / (2 + 1.5 * 1.15) = 1.342282, score = 0.930399
    retriever = BM25Retriever(make_chunks("kedi köpek", "kedi kuş kuş"))
    [result] = retriever.retrieve("kuş", top_k=5)
    assert result.doc_id == "d1"
    assert result.score == pytest.approx(0.930399, abs=1e-5)


def test_bm25_ranks_rare_terms_higher_and_omits_non_matching_chunks():
    retriever = BM25Retriever(make_chunks("ortak kelime", "ortak nadir", "ortak yine", "ilgisiz"))
    results = retriever.retrieve("ortak nadir", top_k=10)
    assert results[0].doc_id == "d1"
    assert {r.doc_id for r in results} == {"d0", "d1", "d2"}
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_bm25_respects_top_k_and_breaks_ties_stably():
    retriever = BM25Retriever(make_chunks("kedi", "kedi", "kedi"))
    assert [r.doc_id for r in retriever.retrieve("kedi", top_k=2)] == ["d0", "d1"]


def test_bm25_no_matching_term_returns_nothing():
    assert BM25Retriever(make_chunks("kedi")).retrieve("zürafa", top_k=3) == []


def test_bm25_rejects_bad_input():
    with pytest.raises(ValueError):
        BM25Retriever([])
    with pytest.raises(ValueError):
        BM25Retriever(make_chunks("kedi")).retrieve("kedi", top_k=0)
