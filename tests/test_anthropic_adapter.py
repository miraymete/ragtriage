"""The Anthropic adapter, tested against a stand-in for the SDK. No network, no credentials."""

import asyncio
import inspect
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ragtriage.generation.llm import LLMError, LLMErrorKind, LLMRequest, StopReason
from ragtriage.generation.providers.anthropic import AnthropicClient


class APIError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class APIStatusError(APIError):
    def __init__(self, message, status_code, *, error_code=None, type=None, api_message=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.type = type
        self.api_message = api_message


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIConnectionError):
    pass


class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeAsyncAnthropic:
    instances: ClassVar[list["FakeAsyncAnthropic"]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = FakeMessages(None)
        FakeAsyncAnthropic.instances.append(self)


@pytest.fixture
def fake_sdk(monkeypatch):
    module = SimpleNamespace(
        APIError=APIError,
        APIStatusError=APIStatusError,
        APIConnectionError=APIConnectionError,
        APITimeoutError=APITimeoutError,
        AsyncAnthropic=FakeAsyncAnthropic,
    )
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return module


def response(text='{"abstained": false, "answer": "x"}', stop_reason="end_turn", **usage):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 12),
            output_tokens=usage.get("output_tokens", 3),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            cache_creation_input_tokens=None,
        ),
        model="claude-test",
    )


REQUEST = LLMRequest(system="sys", prompt="soru", max_tokens=50, json_schema={"type": "object"})


def complete(result, request=REQUEST, **kwargs):
    messages = FakeMessages(result)
    client = AnthropicClient("claude-test", client=SimpleNamespace(messages=messages), **kwargs)
    return asyncio.run(client.complete(request)), messages


def test_request_uses_structured_output_and_never_sends_sampling_parameters(fake_sdk):
    _, messages = complete(response(), effort="low")
    [call] = messages.calls
    assert call == {
        "model": "claude-test",
        "max_tokens": 50,
        "system": "sys",
        "messages": [{"role": "user", "content": "soru"}],
        "output_config": {
            "format": {"type": "json_schema", "schema": {"type": "object"}},
            "effort": "low",
        },
    }


def test_output_config_is_omitted_when_there_is_nothing_to_configure(fake_sdk):
    plain = LLMRequest(system="sys", prompt="soru", max_tokens=50)
    _, messages = complete(response(), request=plain)
    assert "output_config" not in messages.calls[0]


def test_response_text_skips_thinking_blocks_and_usage_fills_missing_counts(fake_sdk):
    completion, _ = complete(response(input_tokens=40, output_tokens=7))
    assert completion.text == '{"abstained": false, "answer": "x"}'
    assert completion.model == "claude-test"
    assert completion.source == "api"
    assert completion.usage.input_tokens == 40
    assert completion.usage.output_tokens == 7
    assert completion.usage.cache_read_input_tokens == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", StopReason.COMPLETE),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("stop_sequence", StopReason.OTHER),
        ("model_context_window_exceeded", StopReason.OTHER),
        (None, StopReason.OTHER),
    ],
)
def test_stop_reasons(fake_sdk, raw, expected):
    completion, _ = complete(response(stop_reason=raw))
    assert completion.stop_reason is expected
    assert completion.raw_stop_reason == raw


@pytest.mark.parametrize(
    ("error", "kind", "fatal"),
    [
        (APIStatusError("bad key", 401), LLMErrorKind.AUTH, True),
        (APIStatusError("forbidden", 403), LLMErrorKind.PERMISSION, True),
        (APIStatusError("no such model", 404), LLMErrorKind.NOT_FOUND, True),
        (APIStatusError("too fast", 429), LLMErrorKind.RATE_LIMIT, False),
        (APIStatusError("busy", 529), LLMErrorKind.OVERLOADED, False),
        (APIStatusError("oops", 500), LLMErrorKind.SERVER, False),
        (APIStatusError("bad", 400), LLMErrorKind.BAD_REQUEST, False),
        (APIStatusError("big", 413), LLMErrorKind.TOO_LARGE, False),
        (APIStatusError("pay", 402), LLMErrorKind.BILLING, True),
        (APITimeoutError("slow"), LLMErrorKind.TIMEOUT, False),
        (APIConnectionError("offline"), LLMErrorKind.CONNECTION, False),
        (TypeError('"Could not resolve authentication method..."'), LLMErrorKind.AUTH, True),
    ],
)
def test_sdk_errors_are_translated(fake_sdk, error, kind, fatal):
    with pytest.raises(LLMError) as raised:
        complete(error)
    assert raised.value.kind is kind
    assert raised.value.fatal is fatal
    assert raised.value.__cause__ is error


def test_unknown_exceptions_are_not_hidden(fake_sdk):
    with pytest.raises(ValueError, match="a bug"):
        complete(ValueError("a bug"))


def test_long_error_messages_are_shortened(fake_sdk):
    with pytest.raises(LLMError) as raised:
        complete(APIStatusError("x" * 5000, 400))
    assert len(raised.value.args[0]) <= 300


def test_retries_and_timeout_are_handed_to_the_sdk_client(fake_sdk):
    FakeAsyncAnthropic.instances.clear()
    AnthropicClient("claude-test", max_retries=5, timeout_s=30.0)
    [instance] = FakeAsyncAnthropic.instances
    assert instance.kwargs == {"max_retries": 5, "timeout": 30.0}


def test_missing_sdk_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ImportError, match=r"ragtriage\[anthropic\]"):
        AnthropicClient("claude-test")


def test_the_real_sdk_accepts_what_the_adapter_sends():
    """Runs only where the optional SDK is installed; CI skips it."""
    anthropic = pytest.importorskip("anthropic")
    from anthropic.resources.messages import AsyncMessages

    parameters = inspect.signature(AsyncMessages.create).parameters
    for name in ("model", "max_tokens", "system", "messages", "output_config"):
        assert name in parameters
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)
    assert issubclass(anthropic.RateLimitError, anthropic.APIStatusError)
    init = inspect.signature(anthropic.AsyncAnthropic.__init__).parameters
    assert "max_retries" in init and "timeout" in init


class ClosableSDK(FakeAsyncAnthropic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_aclose_closes_a_client_the_adapter_created_once(fake_sdk, monkeypatch):
    monkeypatch.setattr(fake_sdk, "AsyncAnthropic", ClosableSDK)
    adapter = AnthropicClient("claude-test")
    asyncio.run(adapter.aclose())
    asyncio.run(adapter.aclose())
    assert adapter._client.closed == 1


def test_aclose_leaves_an_injected_client_to_its_owner(fake_sdk):
    injected = ClosableSDK()
    adapter = AnthropicClient("claude-test", client=injected)
    asyncio.run(adapter.aclose())
    assert injected.closed == 0


def test_a_run_wide_error_code_is_fatal_even_on_a_400(fake_sdk):
    error = APIStatusError(
        "Error code: 400 - {...}",
        400,
        error_code="enforced_spend_limit_reached",
        type="invalid_request_error",
        api_message="Spend limit reached.",
    )
    with pytest.raises(LLMError) as raised:
        complete(error)
    assert raised.value.fatal
    assert raised.value.error_code == "enforced_spend_limit_reached"
    assert raised.value.repeat_key is None
    assert raised.value.args[0] == "Spend limit reached.", "the body message, not the SDK wrapper"


def test_an_unknown_error_code_is_treated_as_its_bare_type(fake_sdk):
    error = APIStatusError("x", 400, error_code="some_future_code", type="invalid_request_error")
    with pytest.raises(LLMError) as raised:
        complete(error)
    assert not raised.value.fatal
    assert raised.value.repeat_key is not None


def test_repeat_key_ignores_the_request_specific_wrapper_text(fake_sdk):
    keys = []
    for request_id in ("req_1", "req_2"):
        error = APIStatusError(
            f"Error code: 400 - request_id={request_id}",
            400,
            type="invalid_request_error",
            api_message="output_config.effort: unsupported value",
        )
        with pytest.raises(LLMError) as raised:
            complete(error)
        keys.append(raised.value.repeat_key)
    assert keys[0] == keys[1] is not None


@pytest.mark.parametrize("status", [413, 429, 500, 401])
def test_only_unclassified_bad_requests_get_a_repeat_key(fake_sdk, status):
    with pytest.raises(LLMError) as raised:
        complete(APIStatusError("same text", status, api_message="same text"))
    assert raised.value.repeat_key is None
