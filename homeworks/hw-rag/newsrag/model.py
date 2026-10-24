"""Model protocols, OpenAI-compatible adapters and the retry template (TODO A1)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import openai


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Reply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class ModelError(Exception):
    """A failed model call; `retryable` says whether repeating it may help."""

    def __init__(self, message: str, *, retryable: bool = False, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@runtime_checkable
class ChatModel(Protocol):
    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None
    ) -> Reply: ...


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _wrap(error: Exception) -> ModelError:
    if isinstance(error, openai.APITimeoutError | openai.APIConnectionError):
        return ModelError(str(error), retryable=True)
    if isinstance(error, openai.APIStatusError):
        retryable = error.status_code in RETRYABLE_STATUS
        return ModelError(
            str(error), retryable=retryable, retry_after=_seconds(error.response.headers.get("retry-after"))
        )
    return ModelError(str(error))


class OpenAIChat:
    """Chat completions on any OpenAI-compatible server; the SDK's own retries are off."""

    def __init__(
        self, base_url: str, api_key: str, model: str, *, extra_body: dict | None = None, timeout: float = 120.0
    ) -> None:
        self.model = model
        self.extra_body = extra_body or {}
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None
    ) -> Reply:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            choice = self.client.chat.completions.create(**kwargs).choices[0]
        except openai.OpenAIError as error:
            raise _wrap(error) from error
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in choice.message.tool_calls or []]
        return Reply(choice.message.content, calls, choice.finish_reason or "stop")


class OpenAIEmbedder:
    """Embeddings endpoint of an OpenAI-compatible server, requested in batches of 64."""

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 120.0) -> None:
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts[start : start + 64])
            except openai.OpenAIError as error:
                raise _wrap(error) from error
            vectors.extend(item.embedding for item in response.data)
        return vectors


class RetryingChat:
    """Repeat a failed `chat` call while the failure is retryable.

    Args:
        inner: the model to call.
        attempts: total number of calls, the first one included.
        sleep: pause function, injected so tests do not wait.

    Rules: retry only ModelError(retryable=True); pause `retry_after` seconds when the error
    carries it, otherwise 0.5 * 2**n seconds plus a little jitter for attempt n (n starts at 0);
    after the last failed attempt raise that ModelError unchanged.
    """

    def __init__(self, inner: ChatModel, attempts: int = 3, sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.attempts = attempts
        self.sleep = sleep

    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None
    ) -> Reply:
        raise NotImplementedError("TODO A1")
