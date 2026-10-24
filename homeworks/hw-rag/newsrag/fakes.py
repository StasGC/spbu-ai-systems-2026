"""Offline stand-ins for the model protocols; they test plumbing, not quality."""

import hashlib
import json
import math
import re

from newsrag.model import Reply, ToolCall


class ScriptedChat:
    """Return scripted replies in order; an Exception in the list is raised instead."""

    def __init__(self, replies: list[Reply | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None
    ) -> Reply:
        self.calls.append(
            {"messages": json.loads(json.dumps(messages)), "tools": tools, "response_format": response_format}
        )
        if not self.replies:
            raise AssertionError("ScriptedChat ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class HashEmbedder:
    """Deterministic character-trigram embeddings; every batch is recorded in `calls`."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        padded = f"  {text.lower()} "
        for i in range(len(padded) - 2):
            digest = hashlib.sha1(padded[i : i + 3].encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class RuleChat:
    """Rule-based chat for `NEWSRAG_BACKEND=mock`: one search, then an answer from the first hit."""

    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, response_format: dict | None = None
    ) -> Reply:
        last = messages[-1]
        if response_format is not None:
            return Reply(self._grounded(str(last.get("content") or "")))
        if tools and last["role"] != "tool":
            query = next(m["content"] for m in reversed(messages) if m["role"] == "user")
            return Reply(None, [ToolCall("call_1", "search", json.dumps({"query": query, "k": 5}))], "tool_calls")
        hits = json.loads(last["content"]).get("hits", []) if last["role"] == "tool" else []
        answer = " ".join(hits[0]["text"].split()[:3]) if hits else "Insufficient information."
        return Reply(f'The answer to the question is "{answer}"')

    @staticmethod
    def _grounded(prompt: str) -> str:
        ids = re.findall(r"\bd\d{3}#\d+\b", prompt)
        if not ids:
            return json.dumps({"answerable": False, "answer": "", "citations": []})
        words = re.findall(r"[A-Za-z0-9']+", prompt.split(ids[0], 1)[1])[:3]
        return json.dumps({"answerable": True, "answer": " ".join(words), "citations": [ids[0]]})
