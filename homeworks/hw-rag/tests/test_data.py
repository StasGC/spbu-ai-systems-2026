from collections import Counter

from newsrag.corpus import load_corpus, load_questions

EXPECTED_TYPES = {"inference_query": 816, "comparison_query": 856, "temporal_query": 583, "null_query": 301}


def test_corpus_file():
    docs = load_corpus()
    assert len(docs) == 609 and docs[0].id == "d000"
    assert len({d.title for d in docs}) == 609


def test_questions_file():
    questions = load_questions()
    assert len(questions) == 2556
    assert Counter(q["question_type"] for q in questions) == EXPECTED_TYPES
    nulls = [q for q in questions if q["question_type"] == "null_query"]
    assert all(q["answer"] == "Insufficient information." and not q["evidence_list"] for q in nulls)
    titles = {d.title for d in load_corpus()}
    assert all(e["title"] in titles for q in questions for e in q["evidence_list"])
