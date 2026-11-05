"""Decide which gold evidence each retrieved chunk covers."""

from __future__ import annotations

from collections.abc import Sequence

from ragtriage.models import Chunk, Evidence


def covers(chunk: Chunk, evidence: Evidence, min_overlap: float = 0.5) -> bool:
    """True if the overlap is at least `min_overlap` of the evidence or of the chunk.

    Checking both directions handles a large chunk holding a short quote and a small chunk sitting
    inside a long passage, while a chunk that only clips the edge of a passage does not count.
    """
    if not 0 < min_overlap <= 1:
        raise ValueError(f"min_overlap must be in (0, 1], got {min_overlap}")
    if chunk.doc_id != evidence.doc_id:
        return False
    overlap = min(chunk.end, evidence.end) - max(chunk.start, evidence.start)
    if overlap <= 0:
        return False
    return (
        overlap / (evidence.end - evidence.start) >= min_overlap
        or overlap / (chunk.end - chunk.start) >= min_overlap
    )


def match_chunks(
    evidence: Sequence[Evidence], chunks: Sequence[Chunk], min_overlap: float = 0.5
) -> list[frozenset[int]]:
    """Return, per rank, the indices of the evidence that chunk covers.

    A chunk returned twice keeps only its best rank, so a retriever bug cannot inflate precision.
    """
    seen: set[tuple[str, int, int]] = set()
    matches: list[frozenset[int]] = []
    for chunk in chunks:
        key = (chunk.doc_id, chunk.start, chunk.end)
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            frozenset(i for i, gold in enumerate(evidence) if covers(chunk, gold, min_overlap))
        )
    return matches
