"""Retriever protocol and the three index templates (TODO B2-B4)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import snowballstemmer

from newsrag.corpus import Chunk
from newsrag.model import Embedder

_stemmer = snowballstemmer.stemmer("english")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[Hit]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase, keep [a-z0-9] words, stem with Snowball (english)."""
    return _stemmer.stemWords(re.findall(r"[a-z0-9]+", text.lower()))


class BM25Index:
    """Okapi BM25 over chunk texts tokenized with `tokenize`.

    score(q, d) = sum over query terms t of idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + K(d)),
    K(d) = k1 * (1 - b + b * |d| / avgdl), idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)),
    where N is the number of chunks and n(t) the number of chunks containing t.
    `search` returns at most k hits with a positive score, best first.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        raise NotImplementedError("TODO B2")

    def search(self, query: str, k: int = 10) -> list[Hit]:
        raise NotImplementedError("TODO B2")


class DenseIndex:
    """Cosine similarity between the query embedding and chunk embeddings.

    With `cache` given, chunk vectors live in a JSON object {sha1(chunk.text): vector} with floats
    rounded to 6 digits; a rebuild embeds only the texts missing from the file.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache: Path | None = None) -> None:
        raise NotImplementedError("TODO B3")

    def search(self, query: str, k: int = 10) -> list[Hit]:
        raise NotImplementedError("TODO B3")


class HybridIndex:
    """Reciprocal rank fusion: each index is asked for k hits, score = sum of 1 / (k_rrf + rank), rank from 1."""

    def __init__(self, *indexes: Retriever, k_rrf: int = 60) -> None:
        raise NotImplementedError("TODO B4")

    def search(self, query: str, k: int = 10) -> list[Hit]:
        raise NotImplementedError("TODO B4")
