"""Standalone dense/sparse/RRF/Cross-Encoder retrieval services."""
from __future__ import annotations

from dataclasses import dataclass

from retrieval.chunking import load_chunks
from retrieval.config import RetrievalConfig
from retrieval.dense_index import search_dense
from retrieval.fuse import reciprocal_rank_fusion
from retrieval.rerank import rerank
from retrieval.sparse_index import search_sparse


@dataclass
class HybridSearcher:
    config: RetrievalConfig

    def __post_init__(self) -> None:
        self._chunks: dict[str, dict] | None = None

    @property
    def chunks(self) -> dict[str, dict]:
        if self._chunks is None:
            self._chunks = {chunk["chunk_id"]: chunk for chunk in load_chunks(self.config.chunks_path)}
        return self._chunks

    def reset(self) -> None:
        self._chunks = None

    def _allowed(self, chunk: dict, filters: dict | None) -> bool:
        if not filters:
            return True
        paper_ids = filters.get("paper_ids")
        sections = filters.get("sections")
        if paper_ids and chunk["paper_id"] not in set(map(str, paper_ids)):
            return False
        return not sections or chunk["section"].casefold() in {str(value).casefold() for value in sections}

    def candidates(self, query: str, *, mode: str = "hybrid",
                   filters: dict | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        if mode not in {"dense", "hybrid"}:
            raise ValueError("candidate mode must be 'dense' or 'hybrid'")
        dense_hits = search_dense(query, self.config.dense_top_k, config=self.config)
        sparse_hits = search_sparse(query, self.config.sparse_top_k, config=self.config) if mode == "hybrid" else []
        dense_scores = dict(dense_hits)
        sparse_scores = dict(sparse_hits)
        if mode == "dense":
            fused = [(chunk_id, score) for chunk_id, score in dense_hits]
        else:
            fused = reciprocal_rank_fusion(
                [[chunk_id for chunk_id, _ in dense_hits], [chunk_id for chunk_id, _ in sparse_hits]],
                self.config.rrf_k,
            )
        results: list[dict] = []
        seen: set[str] = set()
        for chunk_id, fused_score in fused:
            chunk = self.chunks.get(chunk_id)
            if chunk is None or chunk_id in seen or not self._allowed(chunk, filters):
                continue
            seen.add(chunk_id)
            results.append({
                **chunk,
                "dense_score": dense_scores.get(chunk_id),
                "sparse_score": sparse_scores.get(chunk_id),
                "rrf_score": fused_score if mode == "hybrid" else None,
                "score": fused_score,
            })
            if len(results) >= self.config.rerank_top_n:
                break
        return results

    def rerank_candidates(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.config.result_top_k
        by_id = {item["chunk_id"]: item for item in candidates}
        ranked = rerank(
            query, [(item["chunk_id"], item["text"]) for item in candidates], top_k,
            model_name=self.config.reranker_model, device=self.config.device,
            cache_dir=self.config.cache_dir, on_error="raise",
        )
        return [{**by_id[chunk_id], "rerank_score": score, "score": score} for chunk_id, score in ranked]

    def search(self, query: str, top_k: int | None = None, *,
               mode: str = "hybrid_rerank", filters: dict | None = None) -> list[dict]:
        if mode not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("mode must be dense, hybrid, or hybrid_rerank")
        top_k = top_k or self.config.result_top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        candidates = self.candidates(query, mode="dense" if mode == "dense" else "hybrid", filters=filters)
        if mode == "hybrid_rerank":
            return self.rerank_candidates(query, candidates, top_k)
        return candidates[:top_k]

    def retrieve(self, chunk_id: str, context: int = 1) -> dict | None:
        if not isinstance(context, int) or isinstance(context, bool) or context < 0 or context > 10:
            raise ValueError("context must be an integer between 0 and 10")
        target = self.chunks.get((chunk_id or "").strip())
        if target is None:
            return None
        neighbors = sorted(
            (chunk for chunk in self.chunks.values()
             if chunk["paper_id"] == target["paper_id"]
             and abs(int(chunk["chunk_index"]) - int(target["chunk_index"])) <= context),
            key=lambda chunk: (int(chunk["chunk_index"]), chunk["chunk_id"]),
        )
        return {**target, "text": "\n\n".join(chunk["text"] for chunk in neighbors),
                "context_chunk_ids": [chunk["chunk_id"] for chunk in neighbors]}


_default_searchers: dict[tuple, HybridSearcher] = {}


def get_searcher(config: RetrievalConfig | None = None) -> HybridSearcher:
    cfg = config or RetrievalConfig()
    key = (str(cfg.corpus_dir), cfg.dense_model, cfg.reranker_model, cfg.device)
    if key not in _default_searchers:
        _default_searchers[key] = HybridSearcher(cfg)
    return _default_searchers[key]


def search_papers(query: str, top_k: int = 5, *, config: RetrievalConfig | None = None,
                  mode: str = "hybrid_rerank", filters: dict | None = None) -> list[dict]:
    return get_searcher(config).search(query, top_k, mode=mode, filters=filters)


def retrieve_chunk(chunk_id: str, context: int = 1, *,
                   config: RetrievalConfig | None = None) -> dict | None:
    return get_searcher(config).retrieve(chunk_id, context)
