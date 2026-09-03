"""Deterministic Cross-Encoder reranking with an explicit fallback."""
from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder

_models: dict[tuple[str, str, str], CrossEncoder] = {}


def get_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = "cpu",
                 cache_dir: Path | None = None) -> CrossEncoder:
    key = (model_name, device, str(cache_dir or ""))
    if key not in _models:
        _models[key] = CrossEncoder(
            model_name, device=device, cache_folder=str(cache_dir) if cache_dir else None
        )
    return _models[key]


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 5, *,
           model_name: str = "BAAI/bge-reranker-base", device: str = "cpu",
           cache_dir: Path | None = None, model=None,
           on_error: str = "fallback") -> list[tuple[str, float]]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk_id, text in candidates:
        if chunk_id not in seen:
            unique.append((chunk_id, text))
            seen.add(chunk_id)
    if not query.strip() or not unique:
        return []
    try:
        scorer = model or get_reranker(model_name, device, cache_dir)
        raw_scores = scorer.predict([(query, text) for _, text in unique])
        scored = [(chunk_id, float(score)) for (chunk_id, _), score in zip(unique, raw_scores)]
        scored.sort(key=lambda hit: (-hit[1], hit[0]))
        return scored[:top_k]
    except Exception:
        if on_error == "raise":
            raise
        # Preserve the deterministic fused order; decreasing values make the fallback visible.
        return [(chunk_id, -float(index)) for index, (chunk_id, _) in enumerate(unique[:top_k])]
