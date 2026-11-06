"""Claude through the official Anthropic SDK. Install with `pip install "ragtriage[anthropic]"`.

Credentials come from wherever the SDK finds them (ANTHROPIC_API_KEY, an auth token, or a
profile). This module never reads or logs them.

The SDK retries connection errors, 408, 409, 429 and 5xx responses itself (`max_retries`). This
adapter adds no retry loop of its own.
"""

from __future__ import annotations

from typing import Any

from ragtriage.generation.llm import (
    Completion,
    LLMError,
    LLMErrorKind,
    LLMRequest,
    StopReason,
    TokenUsage,
)

_STOP_REASONS = {
    "end_turn": StopReason.COMPLETE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}
_STATUS_KINDS = {
    401: LLMErrorKind.AUTH,
    402: LLMErrorKind.BILLING,
    403: LLMErrorKind.PERMISSION,
    404: LLMErrorKind.NOT_FOUND,
    413: LLMErrorKind.TOO_LARGE,
    429: LLMErrorKind.RATE_LIMIT,
    529: LLMErrorKind.OVERLOADED,
}
# Structured `error.details.error_code` values (anthropic 2.6, types/shared) that describe the
# account or workspace, so every later request would fail the same way. Unknown codes are
# treated as the bare error type, as the SDK recommends.
_RUN_WIDE_CODES = frozenset(
    {
        "enforced_spend_limit_reached",
        "organization_on_hold",
        "organization_type_not_supported",
        "cmek_key_disabled",
        "cmek_key_unreachable",
        "cmek_key_network_blocked",
        "cmek_context_unavailable",
    }
)
_MAX_MESSAGE = 300


def _import_sdk() -> Any:
    try:
        import anthropic
    except ImportError as error:
        raise ImportError(
            'the anthropic provider needs: pip install "ragtriage[anthropic]"'
        ) from error
    return anthropic


def _to_llm_error(error: Exception) -> LLMError | None:
    """Translate an SDK exception, or return None if it is not one the SDK documents."""
    sdk = _import_sdk()
    message = str(getattr(error, "message", error))[:_MAX_MESSAGE]
    if isinstance(error, sdk.APITimeoutError):
        return LLMError(LLMErrorKind.TIMEOUT, message)
    if isinstance(error, sdk.APIConnectionError):
        return LLMError(LLMErrorKind.CONNECTION, message)
    if isinstance(error, sdk.APIStatusError):
        status = int(error.status_code)
        if status in _STATUS_KINDS:
            kind = _STATUS_KINDS[status]
        elif status >= 500:
            kind = LLMErrorKind.SERVER
        else:
            kind = LLMErrorKind.BAD_REQUEST
        # Fields parsed from the response body; the request ID is not part of them.
        error_code = getattr(error, "error_code", None)
        error_type = getattr(error, "type", None)
        api_message = getattr(error, "api_message", None)
        if api_message:
            message = str(api_message)[:_MAX_MESSAGE]
        run_wide = error_code in _RUN_WIDE_CODES
        # An invalid request without a run-wide code may be about this question (a prompt that is
        # too long) or about the settings (an unsupported effort or schema). The API gives no code
        # to tell them apart, so the key lets the pipeline notice the same error repeating.
        repeat_key = (
            f"{status}|{error_type}|{error_code}|{api_message or message}"
            if kind is LLMErrorKind.BAD_REQUEST and not run_wide
            else None
        )
        return LLMError(
            kind,
            message,
            status_code=status,
            error_code=error_code,
            run_wide=run_wide,
            repeat_key=repeat_key,
        )
    # The SDK raises TypeError before sending anything when it finds no credentials.
    if isinstance(error, TypeError) and "authentication method" in str(error):
        return LLMError(LLMErrorKind.AUTH, "no Anthropic credentials found")
    return None


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        effort: str | None = None,
        max_retries: int = 2,
        timeout_s: float = 120.0,
        client: Any = None,
    ) -> None:
        """`client` replaces the SDK client, so tests can run without the network.

        A client passed in belongs to the caller and is not closed by `aclose`.
        """
        self.model = model
        self._effort = effort
        self._owns_client = client is None
        if client is None:
            client = _import_sdk().AsyncAnthropic(max_retries=max_retries, timeout=timeout_s)
        self._client = client

    def response_settings(self) -> dict[str, Any]:
        # Retries, timeouts and credentials do not change what the model writes, so they stay out.
        return {"effort": self._effort}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()
            self._owns_client = False

    async def complete(self, request: LLMRequest) -> Completion:
        # No temperature, top_p or top_k: the Claude models this was written for reject them.
        # A model that needs them would get its own config field, not a change to LLMRequest.
        output_config: dict[str, Any] = {}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if self._effort is not None:
            output_config["effort"] = self._effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if output_config:
            kwargs["output_config"] = output_config

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as error:
            translated = _to_llm_error(error)
            if translated is None:
                raise
            raise translated from error

        usage = response.usage
        return Completion(
            # Thinking blocks may come first; only text blocks hold the answer.
            text="".join(block.text for block in response.content if block.type == "text"),
            stop_reason=_STOP_REASONS.get(response.stop_reason, StopReason.OTHER),
            raw_stop_reason=response.stop_reason,
            usage=TokenUsage(
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None)
                or 0,
            ),
            model=response.model,
            source="api",
        )
