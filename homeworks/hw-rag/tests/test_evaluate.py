import json

import pytest
from conftest import StubRetriever

from newsrag.evaluate import (
    answer_matches,
    answer_metrics,
    exact_match,
    extract_answer,
    fact_in_chunk,
    retrieval_metrics,
    run_eval,
    sample_questions,
)
from newsrag.model import ModelError


def test_official_fact_rule_ignores_whitespace():
    assert fact_in_chunk("Apple unveiled\nthe iPhone 15 on Tuesday.", ["Apple unveiled the iPhone 15 on Tuesday."])
    assert not fact_in_chunk("Google launched the Pixel.", ["Apple unveiled the iPhone 15 on Tuesday."])


def test_official_answer_rules():
    assert extract_answer('Blah.\nThe answer to the question is "Yes"') == "Yes"
    assert extract_answer("plain text") == "plain text"
    assert answer_matches("Sam Bankman-Fried", "sam bankman-fried")
    assert answer_matches("Yes it did", "Yes") and not answer_matches("Yes,", "Yes")  # official rule keeps punctuation
    assert exact_match("The Lakers.", "lakers") and not exact_match("Lakers and Suns", "Lakers")


def test_sample_is_deterministic():
    questions = [{"id": i} for i in range(50)]
    assert sample_questions(questions, 5, seed=1) == sample_questions(questions, 5, seed=1)
    assert len(sample_questions(questions, 500)) == 50


def test_retrieval_metrics_on_toy_data(chunks):
    questions = [
        {
            "query": "pixel",
            "question_type": "inference_query",
            "evidence_list": [{"fact": "The camera got a larger sensor."}],
        },
        {"query": "x", "question_type": "null_query", "evidence_list": []},
        {"query": "y", "question_type": "temporal_query", "evidence_list": [{"fact": "Nothing like this exists."}]},
    ]
    metrics = retrieval_metrics(StubRetriever(chunks), questions, k=10)
    assert metrics["n"] == 2
    assert metrics["hits@10"] == 0.5 and metrics["hits@4"] == 0.5
    assert metrics["mrr@10"] == pytest.approx(0.25)


def test_answer_metrics_by_type():
    rows = [
        {
            "question_type": "inference_query",
            "answer": "Sam Altman",
            "prediction": 'The answer to the question is "Sam Altman"',
        },
        {
            "question_type": "inference_query",
            "answer": "OpenAI",
            "prediction": 'The answer to the question is "Google"',
        },
        {
            "question_type": "null_query",
            "answer": "Insufficient information.",
            "prediction": "Insufficient information.",
        },
        {
            "question_type": "null_query",
            "answer": "Insufficient information.",
            "prediction": 'The answer to the question is "Yes"',
        },
    ]
    metrics = answer_metrics(rows)
    assert metrics["n"] == 4 and metrics["accuracy"] == 0.5 and metrics["exact_match"] == 0.5
    assert metrics["by_type"]["inference_query"] == 0.5 and metrics["null_accuracy"] == 0.5


def test_run_eval_writes_rows_and_survives_a_model_error(tmp_path):
    questions = [
        {"query": "a", "question_type": "inference_query", "answer": "x", "evidence_list": []},
        {"query": "b", "question_type": "inference_query", "answer": "y", "evidence_list": []},
    ]

    def run_fn(question: dict) -> dict:
        if question["query"] == "b":
            raise ModelError("down")
        return {"prediction": 'The answer to the question is "x"', "chunk_ids": ["d000#0"], "extra": 1}

    out = tmp_path / "rows.jsonl"
    metrics = run_eval(run_fn, questions, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["extra"] == 1 and rows[0]["chunk_ids"] == ["d000#0"] and "error" in rows[1]
    assert rows[1]["query"] == "b" and rows[1]["prediction"] == ""
    assert metrics["n"] == 2 and metrics["accuracy"] == 0.5
