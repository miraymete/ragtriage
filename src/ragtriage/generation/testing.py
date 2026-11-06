"""A scripted model for tests: no network, no download, no API key."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ragtriage.generation.llm import (
    Completion,
    LLMError,
    LLMRequest,
    Origin,
    Source,
    StopReason,
    TokenUsage,
)

Script = Callable[[LLMRequest], Completion | LLMError]


def fake_completion(
    payload: dict[str, Any] | str,
    stop_reason: StopReason = StopReason.COMPLETE,
    input_tokens: int = 100,
    output_tokens: int = 20,
    model: str = "fake-model",
    source: Source = "fake",
    original_source: Origin | None = None,
) -> Completion:
    """Build a response. A dict is sent as JSON; a string is sent as is, to test bad output.

    The default source is "fake", so scripted usage is never reported as a billed call. Pass
    source="api" only to test how real calls are counted. A cache hit (source="cache") is a
    replayed real call unless `original_source` says otherwise.
    """
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return Completion(
        text=text,
        stop_reason=stop_reason,
        raw_stop_reason=str(stop_reason),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        source=source,
        original_source=original_source or ("api" if source == "cache" else None),
    )


class FakeLLM:
    """Replays a list of responses in order, or calls a function for each request.

    An `LLMError` in the script is raised instead of returned. Every request is kept in
    `requests`, so a test can check what was sent.
    """

    provider = "fake"

    def __init__(self, script: Sequence[Completion | LLMError] | Script, model: str = "fake-model"):
        self.model = model
        self.requests: list[LLMRequest] = []
        self._script: Script | None = script if callable(script) else None
        self._queue: Iterator[Completion | LLMError] = iter([] if callable(script) else script)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    def response_settings(self) -> dict[str, Any]:
        return {}

    async def complete(self, request: LLMRequest) -> Completion:
        self.requests.append(request)
        if self._script is not None:
            result = self._script(request)
        else:
            try:
                result = next(self._queue)
            except StopIteration:
                raise AssertionError(
                    "FakeLLM received more requests than it was scripted for"
                ) from None
        if isinstance(result, LLMError):
            raise result
        return result
