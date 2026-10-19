"""Repeatable dense/hybrid/rerank comparison without an LLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.dense_index import search_dense
from retrieval.fuse import reciprocal_rank_fusion
from retrieval.hybrid_search import HybridSearcher
from retrieval.rerank import rerank
from retrieval.sparse_index import search_sparse

DEFAULT_QUERIES = [
    "robust reinforcement learning under dynamics uncertainty",
    "transfer learning sample efficiency in deep reinforcement learning",
    "safety constraints and critic based policy repair",
]


def _view(items: list[dict], score_key: str, top_k: int) -> list[dict]:
    return [
        {"rank": rank, "chunk_id": item["chunk_id"], "paper_id": item["paper_id"],
         "title": item["title"], "section": item["section"], "score": item.get(score_key)}
        for rank, item in enumerate(items[:top_k], start=1)
    ]


def compare_query(query: str, config: RetrievalConfig, top_k: int = 5) -> dict:
    searcher = HybridSearcher(config)
    dense_hits = search_dense(query, config.dense_top_k, config=config)
    sparse_hits = search_sparse(query, config.sparse_top_k, config=config)
    dense_rows = [{**searcher.chunks[chunk_id], "dense_score": score} for chunk_id, score in dense_hits]
    fused = reciprocal_rank_fusion(
        [[chunk_id for chunk_id, _ in dense_hits], [chunk_id for chunk_id, _ in sparse_hits]], config.rrf_k
    )
    hybrid_rows = [{**searcher.chunks[chunk_id], "rrf_score": score} for chunk_id, score in fused
                   if chunk_id in searcher.chunks][:config.rerank_top_n]
    ranked = rerank(
        query, [(item["chunk_id"], item["text"]) for item in hybrid_rows], top_k,
        model_name=config.reranker_model, device=config.device, cache_dir=config.cache_dir,
    )
    by_id = {item["chunk_id"]: item for item in hybrid_rows}
    reranked_rows = [{**by_id[chunk_id], "rerank_score": score} for chunk_id, score in ranked]
    return {
        "query": query,
        "dense": _view(dense_rows, "dense_score", top_k),
        "hybrid": _view(hybrid_rows, "rrf_score", top_k),
        "hybrid_rerank": _view(reranked_rows, "rerank_score", top_k),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = RetrievalConfig(corpus_dir=args.corpus_dir, device=args.device)
    payload = {"config": config.to_dict(), "results": [
        compare_query(query, config, args.top_k) for query in (args.query or DEFAULT_QUERIES)
    ]}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
