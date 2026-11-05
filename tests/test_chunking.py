from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ragtriage import Document
from ragtriage.pipelines.chunking import chunk_recursive, chunk_sentences, sentence_spans


def sentences(text):
    return [text[start:end] for start, end in sentence_spans(text)]


def texts(document_text, chunks):
    assert all(document_text[c.start : c.end] == c.text for c in chunks)
    return [c.text for c in chunks]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bir cümle. İkinci cümle.", ["Bir cümle.", "İkinci cümle."]),
        ("Geldi mi? Evet! Çok iyi.", ["Geldi mi?", "Evet!", "Çok iyi."]),
        # Turkish ordinals: the period belongs to the number.
        ("Sinan 16. yüzyılda yaşadı. Sonra öldü.", ["Sinan 16. yüzyılda yaşadı.", "Sonra öldü."]),
        ("1. Dünya Savaşı bitti. Barış geldi.", ["1. Dünya Savaşı bitti.", "Barış geldi."]),
        ("Bitti. 17. yüzyılda geriledi.", ["Bitti.", "17. yüzyılda geriledi."]),
        # Abbreviations and initials.
        ("Prof. Dr. Ayşe geldi. Konuştu.", ["Prof. Dr. Ayşe geldi.", "Konuştu."]),
        ("M.Ö. 9600 yılında başladı. Sürdü.", ["M.Ö. 9600 yılında başladı.", "Sürdü."]),
        ("Lale, sümbül vb. Çiçekler güzeldir.", ["Lale, sümbül vb. Çiçekler güzeldir."]),
        # Decimals have no space, and a lowercase word continues the sentence.
        ("Oran 3.5 oldu. Arttı.", ["Oran 3.5 oldu.", "Arttı."]),
        ('"Neden?" dedi. Sustu.', ['"Neden?" dedi.', "Sustu."]),
        # Paragraph breaks always separate, with or without punctuation.
        ("BAŞLIK\n\nİlk cümle.", ["BAŞLIK", "İlk cümle."]),
        ("  Boşluklu cümle.  ", ["Boşluklu cümle."]),
        ("Noktasız cümle", ["Noktasız cümle"]),
        ("", []),
        ("   \n\n  ", []),
    ],
)
def test_sentence_spans(text, expected):
    assert sentences(text) == expected


def test_chunk_sentences_packs_whole_sentences():
    text = "Birinci cümle. İkinci cümle. Üçüncü cümle."
    chunks = chunk_sentences(Document(id="d", text=text), max_chars=30)
    assert texts(text, chunks) == ["Birinci cümle. İkinci cümle.", "Üçüncü cümle."]


def test_chunk_sentences_overlap_repeats_trailing_sentences():
    text = "Aaaa aa. Bbbb bb. Cccc cc."
    chunks = chunk_sentences(Document(id="d", text=text), max_chars=17, overlap_sentences=1)
    assert texts(text, chunks) == ["Aaaa aa. Bbbb bb.", "Bbbb bb. Cccc cc."]


def test_chunk_sentences_overlap_always_makes_progress():
    text = "Aaaa aa. Bbbb bb. Cccc cc."
    chunks = chunk_sentences(Document(id="d", text=text), max_chars=8, overlap_sentences=5)
    assert texts(text, chunks) == ["Aaaa aa.", "Bbbb bb.", "Cccc cc."]


def test_chunk_sentences_splits_a_long_sentence_between_words():
    text = "kelime kelime kelime kelime."
    chunks = chunk_sentences(Document(id="d", text=text), max_chars=14)
    assert texts(text, chunks) == ["kelime kelime", "kelime kelime."]


def test_recursive_prefers_paragraphs_then_sentences():
    text = "Kısa paragraf.\n\nUzun bir cümle burada. Başka bir cümle burada."
    chunks = chunk_recursive(Document(id="d", text=text), size=30)
    assert texts(text, chunks) == [
        "Kısa paragraf.",
        "Uzun bir cümle burada.",
        "Başka bir cümle burada.",
    ]


def test_recursive_keeps_a_heading_with_the_text_after_it():
    text = "BAŞLIK\n\nBirinci cümle burada. İkinci cümle burada."
    chunks = chunk_recursive(Document(id="d", text=text), size=30)
    assert texts(text, chunks) == ["BAŞLIK\n\nBirinci cümle burada.", "İkinci cümle burada."]


def test_recursive_drops_a_chunk_that_overlap_swallows_whole():
    text = "Kısa.\n\nuzunuzunuzunuzunuzun"
    chunks = chunk_recursive(Document(id="d", text=text), size=30, overlap=10)
    assert texts(text, chunks) == [text]


def test_recursive_cuts_a_word_only_when_it_exceeds_size():
    text = "abcdefghij"
    assert texts(text, chunk_recursive(Document(id="d", text=text), size=4)) == [
        "abcd",
        "efgh",
        "ij",
    ]


def test_recursive_overlap_does_not_cut_words():
    text = "bir iki üç dört beş altı yedi sekiz"
    chunks = chunk_recursive(Document(id="d", text=text), size=15, overlap=6)
    words = set(text.split())
    for chunk in chunks:
        assert set(chunk.text.split()) <= words
        assert len(chunk.text) <= 15
    assert any(a.end > b.start for a, b in pairwise(chunks))


@pytest.mark.parametrize(
    "call",
    [
        lambda d: chunk_sentences(d, max_chars=0),
        lambda d: chunk_sentences(d, max_chars=5, overlap_sentences=-1),
        lambda d: chunk_recursive(d, size=0),
        lambda d: chunk_recursive(d, size=4, overlap=4),
    ],
)
def test_bad_settings_raise(call):
    with pytest.raises(ValueError):
        call(Document(id="d", text="abc"))


prose = st.lists(
    st.sampled_from(
        [
            "kelime",
            "İznik",
            "16.",
            "Dr.",
            "Bitti.",
            "Evet!",
            "\n\n",
            "  ",
            "uzunbirkelimedahaburada",
        ]
    ),
    max_size=40,
).map(" ".join)
all_text = st.one_of(prose, st.text(max_size=200))
limits = st.integers(min_value=1, max_value=60)


def assert_valid(text, chunks, limit):
    covered = set()
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert 0 < len(chunk.text) <= limit
        covered.update(range(chunk.start, chunk.end))
    missing = set(range(len(text))) - covered
    assert all(text[i].isspace() for i in missing), "only whitespace may be dropped"


@given(all_text, limits, st.integers(min_value=0, max_value=3))
def test_chunk_sentences_invariants(text, limit, overlap):
    chunks = chunk_sentences(Document(id="d", text=text), limit, overlap)
    assert_valid(text, chunks, limit)


@given(all_text, limits, st.integers(min_value=0, max_value=59))
def test_chunk_recursive_invariants(text, limit, overlap):
    overlap = min(overlap, limit - 1)
    chunks = chunk_recursive(Document(id="d", text=text), limit, overlap)
    assert_valid(text, chunks, limit)
    assert all(a.start < b.start and a.end < b.end for a, b in pairwise(chunks))
