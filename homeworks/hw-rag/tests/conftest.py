import json

import pytest

from newsrag.corpus import Chunk, Doc
from newsrag.model import Reply, ToolCall
from newsrag.retrieval import Hit


class StubRetriever:
    """Returns the given chunks in order, whatever the query."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return [Hit(c, 1.0) for c in self.chunks[:k]]


def tool_reply(*calls: tuple[str, str, dict]) -> Reply:
    return Reply(None, [ToolCall(call_id, name, json.dumps(args)) for call_id, name, args in calls], "tool_calls")


def final_reply(answer: str) -> Reply:
    return Reply(f'The answer to the question is "{answer}"')


@pytest.fixture
def docs() -> list[Doc]:
    return [
        Doc(
            "d000",
            "Apple unveils new iPhone",
            "The Verge",
            "2023-09-12T10:00:00+00:00",
            "Apple unveiled the iPhone 15 on Tuesday. The phone uses a USB-C port.\n\n"
            "Prices start at 799 dollars. Analysts expect strong sales.",
        ),
        Doc(
            "d001",
            "Google launches Pixel 8",
            "TechCrunch",
            "2023-10-04T15:00:00+00:00",
            "Google launched the Pixel 8 on Wednesday. The camera got a larger sensor.\n\n"
            "The phone ships with seven years of updates.",
        ),
        Doc(
            "d002",
            "Lakers beat Suns",
            "Sporting News",
            "2023-10-26T03:00:00+00:00",
            "The Lakers beat the Suns 100 to 95. LeBron James scored 21 points.\n\nThe next game is on Friday.",
        ),
    ]


@pytest.fixture
def chunks(docs: list[Doc]) -> list[Chunk]:
    return [Chunk(f"{d.id}#0", d.id, f"{d.title} ({d.source}, {d.published_at[:10]})\n{d.body}") for d in docs]
