from __future__ import annotations

import pytest

import retrieval.hybrid_v2 as hybrid_module
from retrieval.config import RetrievalConfig
from retrieval.fuse_v2 import reciprocal_rank_fusion
from retrieval.hybrid_v2 import HybridSearcher
from retrieval.rerank_v2 import rerank


def test_rrf_deduplicates_within_each_ranking_and_is_deterministic():
    first = reciprocal_rank_fusion([["a", "a", "b"], ["b", "a"]], k=60)
    second = reciprocal_rank_fusion([["a", "a", "b"], ["b", "a"]], k=60)
    assert first == second
    assert [item[0] for item in first] == ["a", "b"]
    expected_a = 1 / 61 + 1 / 62
    assert first[0][1] == pytest.approx(expected_a)


def test_rrf_rejects_invalid_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], 0)


class FakeReranker:
    def predict(self, pairs):
        return [len(text) for _, text in pairs]


class BrokenReranker:
    def predict(self, _pairs):
        raise RuntimeError("model failed")


def test_cross_encoder_rerank_order_dedup_and_fallback():
    candidates = [("short", "x"), ("long", "xxxx"), ("long", "duplicate")]
    assert rerank("q", candidates, 2, model=FakeReranker())[0][0] == "long"
    fallback = rerank("q", candidates, 2, model=BrokenReranker())
    assert [item[0] for item in fallback] == ["short", "long"]


def test_hybrid_metadata_filter_and_scores(monkeypatch, tmp_path):
    config = RetrievalConfig(corpus_dir=tmp_path, rerank_top_n=5)
    searcher = HybridSearcher(config)
    searcher._chunks = {
        "a": {"chunk_id": "a", "paper_id": "p1", "title": "A", "section": "Method", "source": "s1", "chunk_index": 0, "text": "alpha"},
        "b": {"chunk_id": "b", "paper_id": "p2", "title": "B", "section": "Results", "source": "s2", "chunk_index": 0, "text": "beta"},
    }
    monkeypatch.setattr(hybrid_module, "search_dense", lambda *a, **k: [("a", .9), ("b", .8)])
    monkeypatch.setattr(hybrid_module, "search_sparse", lambda *a, **k: [("b", 4.0), ("a", 2.0)])
    results = searcher.candidates("query", filters={"paper_ids": ["p2"]})
    assert [item["chunk_id"] for item in results] == ["b"]
    assert results[0]["dense_score"] == .8
    assert results[0]["sparse_score"] == 4.0


def test_retrieve_invalid_id_and_context(monkeypatch, tmp_path):
    searcher = HybridSearcher(RetrievalConfig(corpus_dir=tmp_path))
    searcher._chunks = {}
    assert searcher.retrieve("missing") is None
    with pytest.raises(ValueError):
        searcher.retrieve("missing", -1)
