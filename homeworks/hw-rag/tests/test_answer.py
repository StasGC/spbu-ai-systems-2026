import json

from conftest import StubRetriever

from newsrag.agent import answer_grounded
from newsrag.fakes import ScriptedChat
from newsrag.model import Reply


def grounded(payload: dict | str) -> Reply:
    return Reply(payload if isinstance(payload, str) else json.dumps(payload))


def test_context_carries_ids_and_foreign_citation_is_dropped(chunks):
    chat = ScriptedChat([grounded({"answerable": True, "answer": "Pixel 8", "citations": ["d001#0", "zzz#9"]})])
    answer = answer_grounded(chat, StubRetriever(chunks), "Which phone?", k=3)
    assert answer.answerable and answer.citations == ["d001#0"] and answer.text == "Pixel 8"
    assert chat.calls[0]["response_format"] is not None
    prompt = chat.calls[0]["messages"][-1]["content"]
    assert all(c.id in prompt for c in chunks)


def test_answerable_without_valid_citation_is_a_refusal(chunks):
    chat = ScriptedChat([grounded({"answerable": True, "answer": "Pixel 8", "citations": ["nope#1"]})])
    answer = answer_grounded(chat, StubRetriever(chunks), "Which phone?")
    assert answer.answerable is False and answer.citations == []


def test_invalid_json_is_a_refusal(chunks):
    chat = ScriptedChat([grounded("not json at all")])
    answer = answer_grounded(chat, StubRetriever(chunks), "Which phone?")
    assert answer.answerable is False and answer.raw == "not json at all"


def test_no_hits_means_no_model_call():
    chat = ScriptedChat([])
    answer = answer_grounded(chat, StubRetriever([]), "anything")
    assert answer.answerable is False and chat.calls == []
