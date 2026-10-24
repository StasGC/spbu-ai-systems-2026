import json

from conftest import StubRetriever, final_reply, tool_reply

from newsrag.agent import TOOLS, read_document, run_agent, search
from newsrag.fakes import ScriptedChat
from newsrag.model import Reply, ToolCall


def tool_messages(chat: ScriptedChat, call_index: int) -> list[dict]:
    return [m for m in chat.calls[call_index]["messages"] if m["role"] == "tool"]


def test_search_result_returns_to_the_model_with_the_same_id(chunks, docs):
    chat = ScriptedChat([tool_reply(("c1", "search", {"query": "pixel"})), final_reply("Pixel 8")])
    run = run_agent(chat, StubRetriever(chunks), docs, "Which phone did Google launch?")
    assert run.status == "ok" and run.answer.endswith('"Pixel 8"')
    assert chat.calls[0]["tools"] == TOOLS
    tool = tool_messages(chat, 1)[0]
    assert tool["tool_call_id"] == "c1"
    assert json.loads(tool["content"])["hits"][0]["chunk_id"] == "d000#0"
    assert run.calls[0]["name"] == "search"
    assert run.messages[-1]["content"] == run.answer


def test_whole_batch_is_executed_before_the_next_call(chunks, docs):
    chat = ScriptedChat(
        [tool_reply(("c1", "search", {"query": "a"}), ("c2", "read_document", {"doc_id": "d001"})), final_reply("x")]
    )
    run = run_agent(chat, StubRetriever(chunks), docs, "q")
    assert [m["tool_call_id"] for m in tool_messages(chat, 1)] == ["c1", "c2"]
    assert [c["name"] for c in run.calls] == ["search", "read_document"]
    assert run.calls[1]["result"]["title"] == "Google launches Pixel 8"


def test_unknown_tool_and_bad_json_become_tool_errors(chunks, docs):
    bad = Reply(None, [ToolCall("c1", "delete_everything", "{}"), ToolCall("c2", "search", "{not json")], "tool_calls")
    chat = ScriptedChat([bad, final_reply("x")])
    run = run_agent(chat, StubRetriever(chunks), docs, "q")
    results = [json.loads(m["content"]) for m in tool_messages(chat, 1)]
    assert results == [{"error": "unknown_tool"}, {"error": "invalid_arguments"}]
    assert run.status == "ok"


def test_length_finish_is_a_model_error(chunks, docs):
    chat = ScriptedChat([Reply(None, [ToolCall("c1", "search", '{"query": "a"}')], "length")])
    run = run_agent(chat, StubRetriever(chunks), docs, "q")
    assert run.status == "model_error" and run.calls == []


def test_step_limit(chunks, docs):
    chat = ScriptedChat([tool_reply(("c1", "search", {"query": "a"})), tool_reply(("c2", "search", {"query": "b"}))])
    run = run_agent(chat, StubRetriever(chunks), docs, "q", max_steps=1)
    assert run.status == "step_limit" and len(chat.calls) == 1


def test_read_document_tool(docs):
    assert read_document(docs, "nope") == {"error": "not_found"}
    short = read_document(docs, "d001", max_chars=20)
    assert short["truncated"] is True and len(short["text"]) == 20 and short["source"] == "TechCrunch"
    assert read_document(docs, "d001")["truncated"] is False


def test_search_tool_shape_and_clamping(chunks):
    out = search(StubRetriever(chunks), "pixel", k=50)
    assert set(out["hits"][0]) == {"chunk_id", "doc_id", "text"}
    assert 1 <= len(out["hits"]) <= 10


def test_tool_schemas():
    functions = {t["function"]["name"]: t["function"] for t in TOOLS}
    assert set(functions) == {"search", "read_document"}
    assert "query" in functions["search"]["parameters"]["properties"]
    assert "doc_id" in functions["read_document"]["parameters"]["properties"]
