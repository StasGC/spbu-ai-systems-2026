"""Compare your BM25Index with rank_bm25 on the same chunks: pip install -e ".[baselines]" && python baselines.py [N]."""

import re
import sys

from rank_bm25 import BM25Okapi

from newsrag import corpus, evaluate, retrieval
from newsrag.corpus import Chunk

PAPER = {"bge-large-en-v1.5 (paper, 256-token chunks)": {"hits@4": 0.5221, "hits@10": 0.6718, "mrr@10": 0.4298}}


class RankBM25:
    """rank_bm25.BM25Okapi with the official repository's tokenizer: [a-z0-9]+, no stemming."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([self._tokens(c.text) for c in chunks])

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, k: int = 10) -> list[retrieval.Hit]:
        scores = self.index.get_scores(self._tokens(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [retrieval.Hit(self.chunks[i], float(scores[i])) for i in top]


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    chunks = [c for doc in corpus.load_corpus() for c in corpus.chunk(doc)]
    questions = corpus.load_questions()[:limit]
    rows = {
        "your BM25Index": evaluate.retrieval_metrics(retrieval.BM25Index(chunks), questions),
        "rank_bm25": evaluate.retrieval_metrics(RankBM25(chunks), questions),
        **PAPER,
    }
    for name, m in rows.items():
        print(f"{name:46} hits@4 {m['hits@4']:.3f}  hits@10 {m['hits@10']:.3f}  mrr@10 {m['mrr@10']:.3f}")


if __name__ == "__main__":
    main()
