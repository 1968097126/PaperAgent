"""Tests for retrieval/fuse.py — RRF (Reciprocal Rank Fusion)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.fuse import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_known_example_matches_hand_calculation(self):
        # A = [d1, d2, d3]，B = [d2, d1, d4]，k=60
        # d1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        # d2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61   (和d1同分，因为两路排名互换了d1/d2)
        # d3: 1/(60+3)                              (只在A里出现)
        # d4: 1/(60+3)                              (只在B里出现，和d3同分)
        rankings = [["d1", "d2", "d3"], ["d2", "d1", "d4"]]
        result = reciprocal_rank_fusion(rankings, k=60)

        scores = dict(result)
        expected_d1_d2 = 1 / 61 + 1 / 62
        expected_d3_d4 = 1 / 63

        assert scores["d1"] == scores["d2"]
        assert abs(scores["d1"] - expected_d1_d2) < 1e-9
        assert scores["d3"] == scores["d4"]
        assert abs(scores["d3"] - expected_d3_d4) < 1e-9

        # 并列名次靠稳定排序保留原始相对顺序：d1在A里比d2先出现 -> 并列时d1排前面；
        # d3在处理完A之后就已经在scores字典里，比B引入的d4更早 -> 并列时d3排前面
        ids = [doc_id for doc_id, _ in result]
        assert ids == ["d1", "d2", "d3", "d4"]

    def test_document_only_in_one_ranking_still_included(self):
        rankings = [["a", "b"], ["c"]]
        result = reciprocal_rank_fusion(rankings)
        ids = {doc_id for doc_id, _ in result}
        assert ids == {"a", "b", "c"}

    def test_top_rank_scores_higher_than_lower_rank(self):
        rankings = [["a", "b", "c", "d", "e"]]
        result = reciprocal_rank_fusion(rankings)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_document_appearing_in_both_rankings_beats_appearing_in_one(self):
        # x 在两路里都排名靠后，y 只在一路里但排名靠前 -> 验证"出现在多路"不是无条件占优，
        # 而是看两种情况各自的RRF分数总和（这里特意让x的双路命中分数总和仍然更高，
        # 因为 1/(60+5)+1/(60+5) = 2/65 ≈ 0.0308 大于 1/(60+1) = 1/61 ≈ 0.0164）
        rankings = [["_", "_", "_", "_", "x"], ["_", "_", "_", "_", "x"]]
        rankings_y = [["y"]]
        combined = reciprocal_rank_fusion(rankings)
        y_only = reciprocal_rank_fusion(rankings_y)
        assert dict(combined)["x"] > dict(y_only)["y"]

    def test_empty_rankings_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []
