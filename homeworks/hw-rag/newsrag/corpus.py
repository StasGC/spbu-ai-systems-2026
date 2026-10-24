"""Documents, chunks and loaders for the MultiHop-RAG files; chunking is TODO B1."""

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    source: str
    published_at: str
    body: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


def load_corpus(path: Path = DATA_DIR / "corpus.json") -> list[Doc]:
    """Read corpus.json; ids follow file order: d000, d001, ..."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Doc(f"d{i:03d}", r["title"], r["source"], r["published_at"], r["body"]) for i, r in enumerate(rows)]


def load_questions(path: Path = DATA_DIR / "MultiHopRAG.json") -> list[dict]:
    """Read MultiHopRAG.json as a list of dicts: query, answer, question_type, evidence_list."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chunk(doc: Doc, max_chars: int = 1000) -> list[Chunk]:
    """Split a document into chunks of at most `max_chars` characters.

    Rules:
        - the text of every chunk starts with the line f"{doc.title} ({doc.source}, {doc.published_at[:10]})";
        - the body is split into sentences (blank lines and ".", "!", "?" followed by whitespace are
          boundaries) and sentences are packed in order, separated by one space; a sentence is never cut;
        - `max_chars` counts the header line; a single sentence longer than the limit becomes its own
          chunk even though it exceeds it;
        - ids are f"{doc.id}#{n}" with n counted from 0.
    """
    raise NotImplementedError("TODO B1")
