"""Split documents into chunks that remember their position."""

from __future__ import annotations

from ragtriage.models import Chunk, Document


def chunk_fixed(document: Document, size: int, overlap: int = 0) -> list[Chunk]:
    """Slide a window of `size` characters over the text, advancing by `size - overlap`."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got {overlap}")
    chunks = []
    start = 0
    while start < len(document.text):
        end = min(start + size, len(document.text))
        chunks.append(
            Chunk(doc_id=document.id, start=start, end=end, text=document.text[start:end])
        )
        if end == len(document.text):
            break
        start += size - overlap
    return chunks
