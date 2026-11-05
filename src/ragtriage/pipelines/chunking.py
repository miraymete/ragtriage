"""Split documents into chunks that remember their position."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from itertools import pairwise

from ragtriage.models import Chunk, Document
from ragtriage.pipelines.text import turkish_lower

Span = tuple[int, int]

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
_LAST_WORD = re.compile(r"(\w+)[.!?…\"'”’)\]]*$")
_WORD = re.compile(r"\S+")
# A period after these does not end a sentence. Single letters cover "M.Ö." and "T.C.".
_ABBREVIATIONS = frozenset(
    {"av", "bkz", "cad", "doç", "dr", "mah", "müh", "no", "prof", "sok", "vb", "vs", "yrd", "örn"}
)


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_at(text: str, start: int, end: int, cuts: Sequence[int]) -> list[Span]:
    edges = [start, *cuts, end]
    spans = (_trim(text, a, b) for a, b in pairwise(edges))
    return [span for span in spans if span]


def _paragraph_spans(text: str, start: int, end: int) -> list[Span]:
    return _split_at(
        text, start, end, [m.end() for m in _PARAGRAPH_BREAK.finditer(text, start, end)]
    )


def _ends_sentence(text: str, match: re.Match[str], end: int) -> bool:
    following = _trim(text, match.end(), end)
    if following is None:
        return True
    next_char = text[following[0]]
    if next_char.islower():
        return False
    if "." not in match.group():
        return True
    # "10. yüzyıl" and "1. Dünya Savaşı" are ordinals: digits before a period do not end a sentence.
    word = _LAST_WORD.search(text, max(0, match.start() - 40), match.end())
    if word is None:
        return True
    token = turkish_lower(word.group(1))
    return not (token.isdigit() or len(token) == 1 or token in _ABBREVIATIONS)


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Find sentences with rules that hold for Turkish ordinals and abbreviations."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    for paragraph_start, paragraph_end in _paragraph_spans(text, start, end):
        cuts = [
            match.end()
            for match in _SENTENCE_END.finditer(text, paragraph_start, paragraph_end)
            if _ends_sentence(text, match, paragraph_end)
        ]
        spans.extend(_split_at(text, paragraph_start, paragraph_end, cuts))
    return spans


def _word_spans(text: str, start: int, end: int) -> list[Span]:
    return [match.span() for match in _WORD.finditer(text, start, end)]


_SPLITTERS: tuple[Callable[[str, int, int], list[Span]], ...] = (
    _paragraph_spans,
    sentence_spans,
    _word_spans,
)
_WORD_LEVEL = 2


def _units(text: str, span: Span, size: int, level: int) -> list[Span]:
    """Break a span into the largest natural units that fit in `size`."""
    start, end = span
    if end - start <= size:
        return [span]
    if level == len(_SPLITTERS):
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    return [
        unit
        for piece in _SPLITTERS[level](text, start, end)
        for unit in _units(text, piece, size, level + 1)
    ]


def _fit(text: str, span: Span, size: int, level: int) -> list[Span]:
    """Merge units greedily up to `size`, so a heading stays with the text that follows it."""
    fitted: list[Span] = []
    current: Span | None = None
    for unit in _units(text, span, size, level):
        if current and unit[1] - current[0] > size:
            fitted.append(current)
            current = None
        current = (current[0] if current else unit[0], unit[1])
    if current:
        fitted.append(current)
    return fitted


def _to_chunk(document: Document, start: int, end: int) -> Chunk:
    return Chunk(doc_id=document.id, start=start, end=end, text=document.text[start:end])


def validate_fixed(size: int, overlap: int) -> None:
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap}")


def chunk_fixed(document: Document, size: int, overlap: int = 0) -> list[Chunk]:
    """Slide a window of `size` characters over the text, advancing by `size - overlap`.

    This can cut through a word. It is the baseline the other strategies are compared with.
    """
    validate_fixed(size, overlap)
    chunks = []
    start = 0
    while start < len(document.text):
        end = min(start + size, len(document.text))
        chunks.append(_to_chunk(document, start, end))
        if end == len(document.text):
            break
        start += size - overlap
    return chunks


def chunk_sentences(document: Document, max_chars: int, overlap_sentences: int = 0) -> list[Chunk]:
    """Pack whole sentences into chunks of at most `max_chars`.

    A sentence longer than `max_chars` is split at word boundaries. `overlap_sentences` repeats
    that many trailing sentences at the start of the next chunk.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    if overlap_sentences < 0:
        raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")
    text = document.text
    units = [
        unit
        for sentence in sentence_spans(text)
        for unit in _fit(text, sentence, max_chars, _WORD_LEVEL)
    ]
    chunks = []
    first = 0
    while first < len(units):
        last = first
        while last + 1 < len(units) and units[last + 1][1] - units[first][0] <= max_chars:
            last += 1
        chunks.append(_to_chunk(document, units[first][0], units[last][1]))
        if last == len(units) - 1:
            break
        first = max(first + 1, last + 1 - overlap_sentences)
    return chunks


def _extend_back(text: str, start: int, overlap: int) -> int:
    """Move a chunk's start back by up to `overlap` characters without cutting a word."""
    new_start = max(0, start - overlap)
    while new_start > 0 and new_start < start and not text[new_start - 1].isspace():
        new_start += 1
    while new_start < start and text[new_start].isspace():
        new_start += 1
    return new_start


def chunk_recursive(document: Document, size: int, overlap: int = 0) -> list[Chunk]:
    """Split at paragraphs, then sentences, then words, and merge pieces up to `size` characters.

    Every chunk, including its `overlap` characters, fits in `size`.
    """
    validate_fixed(size, overlap)
    text = document.text
    whole = _trim(text, 0, len(text))
    if whole is None:
        return []
    spans = _fit(text, whole, size - overlap, 0)
    starts = [spans[0][0], *(_extend_back(text, start, overlap) for start, _ in spans[1:])]
    # Overlap can swallow a short chunk whole. Keeping it would only add a duplicate result.
    return [
        _to_chunk(document, start, end)
        for i, (start, (_, end)) in enumerate(zip(starts, spans, strict=True))
        if i + 1 == len(spans) or starts[i + 1] > start
    ]
