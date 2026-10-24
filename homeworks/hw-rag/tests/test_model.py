import pytest

from newsrag.fakes import ScriptedChat
from newsrag.model import ModelError, Reply, RetryingChat

MESSAGES = [{"role": "user", "content": "hi"}]


def test_retries_once_then_succeeds():
    inner = ScriptedChat([ModelError("busy", retryable=True), Reply("ok")])
    sleeps: list[float] = []
    assert RetryingChat(inner, attempts=3, sleep=sleeps.append).chat(MESSAGES).content == "ok"
    assert len(inner.calls) == 2
    assert len(sleeps) == 1 and sleeps[0] >= 0


def test_non_retryable_error_is_raised_at_once():
    inner = ScriptedChat([ModelError("bad request", retryable=False), Reply("never")])
    with pytest.raises(ModelError):
        RetryingChat(inner, sleep=lambda s: None).chat(MESSAGES)
    assert len(inner.calls) == 1


def test_gives_up_after_attempts():
    inner = ScriptedChat([ModelError("busy", retryable=True)] * 3 + [Reply("late")])
    with pytest.raises(ModelError):
        RetryingChat(inner, attempts=3, sleep=lambda s: None).chat(MESSAGES)
    assert len(inner.calls) == 3


def test_retry_after_is_honoured():
    inner = ScriptedChat([ModelError("slow down", retryable=True, retry_after=7.0), Reply("ok")])
    sleeps: list[float] = []
    RetryingChat(inner, sleep=sleeps.append).chat(MESSAGES)
    assert sleeps == [7.0]


def test_tools_and_response_format_are_passed_through():
    inner = ScriptedChat([Reply("ok")])
    RetryingChat(inner).chat(MESSAGES, tools=[{"type": "function"}], response_format={"type": "json_object"})
    assert inner.calls[0]["tools"] == [{"type": "function"}]
    assert inner.calls[0]["response_format"] == {"type": "json_object"}
