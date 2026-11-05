"""Interfaces that pipeline components implement."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragtriage.models import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedChunk]: ...
