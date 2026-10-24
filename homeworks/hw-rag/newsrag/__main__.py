"""Command line: ask / agent / eval."""

import argparse
import json
import os
import sys
from pathlib import Path

from newsrag import agent, corpus, evaluate, fakes, model, retrieval

INDEXES = ("bm25", "dense", "hybrid")
DEFAULTS = {
    "NEWSRAG_BASE_URL": "http://localhost:11434/v1",
    "NEWSRAG_API_KEY": "none",
    "NEWSRAG_MODEL": "qwen3:8b",
    "NEWSRAG_EMBED_MODEL": "bge-m3",
    "NEWSRAG_EXTRA_BODY": '{"reasoning_effort": "none"}',
}


def setting(name: str) -> str:
    return os.environ.get(name, DEFAULTS[name])


def build_models() -> tuple[model.ChatModel, model.Embedder, str]:
    if os.environ.get("NEWSRAG_BACKEND") == "mock":
        return fakes.RuleChat(), fakes.HashEmbedder(), "mock"
    base_url, api_key = setting("NEWSRAG_BASE_URL"), setting("NEWSRAG_API_KEY")
    extra_body = json.loads(setting("NEWSRAG_EXTRA_BODY"))
    chat = model.RetryingChat(model.OpenAIChat(base_url, api_key, setting("NEWSRAG_MODEL"), extra_body=extra_body))
    return chat, model.OpenAIEmbedder(base_url, api_key, setting("NEWSRAG_EMBED_MODEL")), setting("NEWSRAG_EMBED_MODEL")


def build_index(
    kind: str, chunks: list[corpus.Chunk], embedder: model.Embedder, embed_model: str
) -> retrieval.Retriever:
    bm25 = retrieval.BM25Index(chunks)
    if kind == "bm25":
        return bm25
    dense = retrieval.DenseIndex(chunks, embedder, cache=Path("index") / f"{embed_model}.json")
    return dense if kind == "dense" else retrieval.HybridIndex(bm25, dense)


def agent_row(chat: model.ChatModel, index: retrieval.Retriever, docs: list[corpus.Doc], question: dict) -> dict:
    run = agent.run_agent(chat, index, docs, question["query"])
    hits = [h for c in run.calls if c["name"] == "search" for h in c["result"].get("hits", [])]
    return {
        "prediction": run.answer,
        "chunk_ids": [h["chunk_id"] for h in hits],
        "status": run.status,
        "calls": run.calls,
        "messages": run.messages,
    }


def rag_row(chat: model.ChatModel, index: retrieval.Retriever, question: dict) -> dict:
    answer = agent.answer_grounded(chat, index, question["query"])
    prediction = answer.text if answer.answerable else "Insufficient information."
    return {"prediction": prediction, "chunk_ids": answer.citations, "answerable": answer.answerable, "raw": answer.raw}


def run(args: argparse.Namespace) -> int:
    chat, embedder, embed_model = build_models()
    docs = corpus.load_corpus()
    chunks = [c for doc in docs for c in corpus.chunk(doc)]
    index = build_index(args.index, chunks, embedder, embed_model)
    if args.command == "ask":
        answer = agent.answer_grounded(chat, index, args.question, k=args.k)
        print(json.dumps(answer.__dict__, ensure_ascii=False, indent=1))
    elif args.command == "agent":
        result = agent.run_agent(chat, index, docs, args.question, max_steps=args.max_steps)
        with Path("traces.jsonl").open("a", encoding="utf-8") as trace:
            trace.write(
                json.dumps({"question": args.question, "status": result.status, "messages": result.messages}) + "\n"
            )
        print(result.answer)
        print(f"[{result.status}] tools: {[c['name'] for c in result.calls]}", file=sys.stderr)
    elif args.what == "retrieval":
        questions = corpus.load_questions()[: args.limit]
        print(json.dumps(evaluate.retrieval_metrics(index, questions, k=args.k)))
    else:
        questions = evaluate.sample_questions(corpus.load_questions(), args.sample, args.seed)
        out = Path(args.out) / f"{args.mode}_{args.index}_seed{args.seed}.jsonl"
        if args.mode == "agent":
            metrics = evaluate.run_eval(lambda q: agent_row(chat, index, docs, q), questions, out)
        else:
            metrics = evaluate.run_eval(lambda q: rag_row(chat, index, q), questions, out)
        print(json.dumps(metrics, indent=1))
        print(f"rows: {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="newsrag")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("ask", "single-shot RAG answer"), ("agent", "tool-using agent")):
        p = sub.add_parser(name, help=text)
        p.add_argument("question")
        p.add_argument("--index", choices=INDEXES, default="bm25")
        p.add_argument("--k", type=int, default=6)
        p.add_argument("--max-steps", type=int, default=6)
    ev = sub.add_parser("eval", help="retrieval metrics or answer accuracy")
    ev.add_argument("what", choices=("retrieval", "answers"))
    ev.add_argument("--index", choices=INDEXES, default="bm25")
    ev.add_argument("--k", type=int, default=10)
    ev.add_argument("--mode", choices=("rag", "agent"), default="rag")
    ev.add_argument("--sample", type=int, default=200)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--limit", type=int, default=None, help="retrieval: first N questions only")
    ev.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except NotImplementedError as todo:
        print(f"{todo}: см. README, раздел «Что написать»", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
