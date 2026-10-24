from newsrag.corpus import chunk
from newsrag.evaluate import fact_in_chunk

HEADER = "Apple unveils new iPhone (The Verge, 2023-09-12)"


def test_header_line_and_ids(docs):
    parts = chunk(docs[0], max_chars=120)
    assert all(p.text.startswith(HEADER + "\n") for p in parts)
    assert [p.id for p in parts] == [f"d000#{i}" for i in range(len(parts))]
    assert all(p.doc_id == "d000" for p in parts)


def test_respects_max_chars_without_cutting_sentences(docs):
    parts = chunk(docs[0], max_chars=120)
    assert len(parts) == 2
    assert all(len(p.text) <= 120 for p in parts)
    assert all(p.text.rstrip().endswith(".") for p in parts)


def test_fact_survives_in_exactly_one_chunk(docs):
    parts = chunk(docs[0], max_chars=120)
    assert sum(fact_in_chunk(p.text, ["The phone uses a USB-C port."]) for p in parts) == 1


def test_short_document_is_one_chunk(docs):
    parts = chunk(docs[2])
    assert len(parts) == 1
    assert "LeBron James scored 21 points." in parts[0].text
