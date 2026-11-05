"""Tokenizers, including the handling Turkish needs and `str.lower()` does not provide."""

from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules: I becomes ı and İ becomes i.

    Plain `str.lower()` turns "KIRMIZI" into "kirmizi", which never matches "kırmızı". It also turns
    "İ" into "i" plus a combining dot, which splits "İznik" into the tokens "i" and "znik".
    """
    return unicodedata.normalize("NFC", text).replace("I", "ı").replace("İ", "i").lower()


def naive_tokens(text: str) -> list[str]:
    """Baseline: Python's default lowercasing. Kept to measure what it costs on Turkish."""
    return _WORD.findall(text.lower())


def turkish_tokens(text: str) -> list[str]:
    return _WORD.findall(turkish_lower(text))


def turkish_prefix_tokens(text: str, length: int = 5) -> list[str]:
    """Keep the first `length` letters, a cheap stand-in for stemming an agglutinative language.

    "kuruldu" and "kurulan" both become "kurul". Can et al. (SIGIR 2006) found a prefix length of
    5 effective for Turkish, and Can et al. (JASIST 2008) found simple truncation similar in
    effectiveness to a lemmatizer-based stemmer.
    """
    return [token[:length] for token in turkish_tokens(text)]
