"""Tool-using agent over the news archive (TODO A2-A3) and the single-shot grounded answer (TODO B5)."""

from dataclasses import dataclass

from newsrag.corpus import Doc
from newsrag.model import ChatModel
from newsrag.retrieval import Retriever

SYSTEM = """You are an analyst answering questions over a news archive (September to December 2023).
Tools: `search` finds passages; `read_document` returns a full article by doc_id.
Every passage starts with a line "Title (Source, YYYY-MM-DD)": use it for dates and sources.
Many questions need facts from several articles: search more than once with different queries.
Answer only from what the tools returned. If the archive does not contain the answer, say so.
Answers are short: a name, an entity, Yes or No, before or after; use "Insufficient information." when the archive has no answer.
Finish with exactly one line: The answer to the question is "<answer>\""""

TOOLS: list[dict] = []  # TODO A2: OpenAI tool schemas for `search` and `read_document`


@dataclass(frozen=True)
class AgentRun:
    answer: str
    calls: list[dict]
    messages: list[dict]
    status: str  # ok | step_limit | model_error


@dataclass(frozen=True)
class Answer:
    answerable: bool
    text: str
    citations: list[str]
    raw: str = ""


def search(retriever: Retriever, query: str, k: int = 5) -> dict:
    """Tool body: {"hits": [{"chunk_id", "doc_id", "text"}]} for the top-k passages; k is clamped to 1..10."""
    raise NotImplementedError("TODO A2")


def read_document(corpus: list[Doc], doc_id: str, max_chars: int = 12000) -> dict:
    """Tool body: {"doc_id", "title", "source", "published_at", "text", "truncated"}; max_chars is clamped to 20000; unknown id -> {"error": "not_found"}."""
    raise NotImplementedError("TODO A2")


def run_agent(chat: ChatModel, retriever: Retriever, corpus: list[Doc], question: str, max_steps: int = 6) -> AgentRun:
    """Tool loop: call the model, execute every tool call it returns, repeat until it answers.

    Args:
        chat: the model; call it with `tools=TOOLS`.
        retriever: backs the `search` tool.
        corpus: backs the `read_document` tool.
        question: user text.
        max_steps: maximum number of `chat` calls.

    Contract:
        - messages: system, user, then each assistant reply with its `tool_calls` followed by one
          `role="tool"` message per call carrying the same `tool_call_id`;
        - all calls of one reply are executed before the model is called again;
        - an unknown tool or arguments that are not a valid JSON object of the right types give
          {"error": "unknown_tool"} or {"error": "invalid_arguments"} as the tool result, never an exception;
        - duplicate tool_call ids in one reply, or finish_reason == "length": nothing is executed,
          status "model_error";
        - tool calls still pending after the last allowed `chat`: status "step_limit";
        - `calls` records every executed call as {"name", "args", "result"}; `messages` is the full history.
    """
    raise NotImplementedError("TODO A3")


def answer_grounded(chat: ChatModel, retriever: Retriever, question: str, k: int = 6) -> Answer:
    """Single-shot RAG: top-k passages in one prompt, a JSON answer, citations checked in code.

    Contract:
        - no hits -> Answer(False, "", [], "") without calling the model;
        - every passage block in the prompt contains the literal chunk id; the model is asked through
          `response_format` for {"answerable": bool, "answer": str, "citations": [chunk ids]};
        - invalid JSON -> refusal; citations outside the context are dropped; answerable without a
          surviving citation -> refusal; `raw` keeps the model text.
    """
    raise NotImplementedError("TODO B5")
