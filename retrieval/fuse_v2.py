"""Deterministic Reciprocal Rank Fusion."""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("RRF k must be a positive integer")
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for list_index, ranking in enumerate(rankings):
        seen_in_ranking: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if document_id in seen_in_ranking:
                continue
            seen_in_ranking.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(document_id, (list_index, rank))
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))
