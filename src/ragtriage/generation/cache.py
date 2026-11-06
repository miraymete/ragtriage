"""A SQLite cache of validated LLM responses, wrapped around any `LLMClient`.

Only responses the caller has validated are written: `CachedLLM.complete` never stores anything,
and the generator calls `remember` after an answer passes its checks. Refusals, truncated or
invalid output, and API errors therefore never reach the cache.

Entries are plain text: questions, retrieved passages and answers are readable by anyone who can
read the file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from ragtriage.generation.llm import (
    Completion,
    LLMClient,
    LLMRequest,
    Origin,
    StopReason,
    TokenUsage,
)

# Bump when the stored layout or the key changes. Older files then fail loudly instead of being
# misread. There is no migration: move or delete the file to start again.
CACHE_SCHEMA_VERSION = 1
_KEY_FORMAT = "ragtriage-llm-cache-v1"


class CacheError(Exception):
    """The cache file cannot be trusted. Nothing is deleted and no paid call replaces it."""


def cache_key(provider: str, model: str, settings: dict[str, Any], request: LLMRequest) -> str:
    """A deterministic hash of everything that can change the response.

    Credentials are never part of a request or of client settings, so they cannot leak in here.
    """
    material = {
        "format": _KEY_FORMAT,
        "provider": provider,
        "model": model,
        "settings": settings,
        "request": request.model_dump(mode="json"),
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class _StoredResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    stop_reason: StopReason
    raw_stop_reason: str | None
    usage: TokenUsage
    model: str
    origin: Origin


class ResponseCache:
    """One SQLite file. Open it once per run and close it with `close`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        try:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in self._db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if version == 0 and not tables:
                self._db.execute(
                    "CREATE TABLE responses ("
                    " key TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL,"
                    " payload TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
                )
                self._db.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
                self._db.commit()
            elif version != CACHE_SCHEMA_VERSION or "responses" not in tables:
                raise CacheError(
                    f"{path} is not a ragtriage cache of version {CACHE_SCHEMA_VERSION} "
                    f"(found version {version}). Move or delete it to start a new cache."
                )
        except sqlite3.DatabaseError as error:
            self.close()
            raise CacheError(
                f"{path} is not a readable SQLite cache ({error}). "
                "Move or delete it to start a new cache."
            ) from error
        except CacheError:
            self.close()
            raise

    def get(self, key: str, provider: str, model: str) -> _StoredResponse | None:
        try:
            row = self._db.execute(
                "SELECT provider, model, payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        except sqlite3.DatabaseError as error:
            raise CacheError(f"cannot read {self.path}: {error}") from error
        if row is None:
            return None
        stored_provider, stored_model, payload = row
        if (stored_provider, stored_model) != (provider, model):
            raise CacheError(
                f"entry {key[:12]} in {self.path} belongs to {stored_provider}/{stored_model}, "
                f"not {provider}/{model}. The file may be corrupt."
            )
        try:
            return _StoredResponse.model_validate_json(payload)
        except ValidationError as error:
            raise CacheError(
                f"entry {key[:12]} in {self.path} is corrupt or from an incompatible version. "
                "Move or delete the file to start a new cache."
            ) from error

    def put(self, key: str, provider: str, model: str, stored: _StoredResponse) -> None:
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, provider, model, payload)"
                " VALUES (?, ?, ?, ?)",
                (key, provider, model, stored.model_dump_json()),
            )
            self._db.commit()
        except sqlite3.DatabaseError as error:
            raise CacheError(f"cannot write {self.path}: {error}") from error

    def close(self) -> None:
        if not self.closed:
            self._db.close()
            self.closed = True


@runtime_checkable
class RemembersValidated(Protocol):
    async def remember(self, request: LLMRequest, completion: Completion) -> None: ...


class CachedLLM:
    """Serve repeated requests from a `ResponseCache`; pass misses to the wrapped client.

    A hit returns the stored response with `source="cache"` and the original usage, so reports
    can show what was reused without counting it as new spend. Fake responses stay fake: their
    `original_source` is "fake", and the provider name keeps them apart from real ones.
    """

    def __init__(self, inner: LLMClient, cache: ResponseCache) -> None:
        self.provider = inner.provider
        self.model = inner.model
        self._inner = inner
        self._cache = cache

    def response_settings(self) -> dict[str, Any]:
        return self._inner.response_settings()

    def _key(self, request: LLMRequest) -> str:
        return cache_key(self.provider, self.model, self.response_settings(), request)

    async def complete(self, request: LLMRequest) -> Completion:
        stored = self._cache.get(self._key(request), self.provider, self.model)
        if stored is None:
            return await self._inner.complete(request)
        return Completion(
            text=stored.text,
            stop_reason=stored.stop_reason,
            raw_stop_reason=stored.raw_stop_reason,
            usage=stored.usage,
            model=stored.model,
            source="cache",
            original_source=stored.origin,
        )

    async def remember(self, request: LLMRequest, completion: Completion) -> None:
        """Store a response the caller has validated. Cache hits and unfinished responses are
        never stored, even if a caller passes one."""
        if completion.source == "cache" or completion.stop_reason is not StopReason.COMPLETE:
            return
        stored = _StoredResponse(
            text=completion.text,
            stop_reason=completion.stop_reason,
            raw_stop_reason=completion.raw_stop_reason,
            usage=completion.usage,
            model=completion.model,
            origin=completion.source,
        )
        self._cache.put(self._key(request), self.provider, self.model, stored)

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._cache.close()
