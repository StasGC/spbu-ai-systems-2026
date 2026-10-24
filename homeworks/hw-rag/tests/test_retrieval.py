import json

import pytest
from conftest import StubRetriever

from newsrag.corpus import Chunk
from newsrag.fakes import HashEmbedder
from newsrag.retrieval import BM25Index, DenseIndex, HybridIndex, Retriever, tokenize

MICRO = [Chunk("A", "A", "apple banana"), Chunk("B", "B", "banana banana cherry"), Chunk("C", "C", "cherry")]


def test_tokenize_lowercases_and_stems():
    assert tokenize("Launches, Cameras!") == ["launch", "camera"]


def test_bm25_matches_hand_computed_scores():
    hits = BM25Index(MICRO).search("banana", k=3)
    assert [h.chunk.id for h in hits] == ["B", "A"]
    assert hits[0].score == pytest.approx(0.578466, abs=1e-4)
    assert hits[1].score == pytest.approx(0.470004, abs=1e-4)


def test_bm25_ranks_the_matching_chunk_first(chunks):
    hits = BM25Index(chunks).search("Pixel camera sensor", k=2)
    assert hits[0].chunk.id == "d001#0"


def test_dense_index_finds_identical_text_and_uses_cache(chunks, tmp_path):
    cache = tmp_path / "vectors.json"
    index = DenseIndex(chunks, HashEmbedder(), cache=cache)
    assert index.search(chunks[1].text, k=1)[0].chunk.id == "d001#0"
    assert len(json.loads(cache.read_text())) == 3
    embedder = HashEmbedder()
    DenseIndex(chunks, embedder, cache=cache)
    assert embedder.calls == []


def test_rrf_fuses_known_rankings(chunks):
    a, b, c = chunks
    fused = HybridIndex(StubRetriever([a, b, c]), StubRetriever([c, a, b])).search("q", k=3)
    assert [h.chunk.id for h in fused] == [a.id, c.id, b.id]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_indexes_satisfy_the_protocol(chunks):
    bm25 = BM25Index(chunks)
    dense = DenseIndex(chunks, HashEmbedder())
    for index in (bm25, dense, HybridIndex(bm25, dense)):
        assert isinstance(index, Retriever)
        assert len(index.search("phone", k=2)) <= 2
