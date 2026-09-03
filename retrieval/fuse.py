"""RRF（Reciprocal Rank Fusion）：把稠密检索和稀疏检索各自的排名列表融合成一个排名。

为什么不能直接把dense的相似度分数和BM25的分数加权相加：两者量纲完全不同
（dense分数是[-1,1]的余弦相似度，BM25分数是无固定范围、和语料库统计特性相关的
数值），直接相加没有数学意义，权重也没有一个"正确"的设定方式。RRF的思路是
只看"排名"不看"分数"——一个chunk在某个检索器的排名列表里排第几名，
不管两个检索器的分数量纲差多远，排名都是可比的序数。
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """对多路排名列表做RRF融合。

    RRF公式：对每个文档d，融合分数 = 该文档在每一路排名列表r中贡献的
    1/(k + rank_r(d)) 之和（rank从1开始计数；文档没出现在某一路列表里就
    不贡献那一项，不是记0分）。k是一个平滑常数（论文原始设定+业界常用值都是60），
    作用是压低"排名靠后的文档"对融合分数的影响力——排名越靠后，1/(k+rank)
    的增量越小，且k越大整体曲线越平缓（不同排名之间分数差异更小）。

    Args:
        rankings: 多路排名列表，每一路自身已经是按相关性从高到低排好序的chunk_id列表
                  （比如 [dense_search的结果, sparse_search的结果]）
        k: RRF平滑常数，默认60（沿用RRF原论文和大多数生产系统的经验值）
    Returns:
        [(chunk_id, fused_score), ...]，按融合分数从高到低排序
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):   # enumerate从1开始计数，rank直接就是名次
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # dict.items()转成列表后按分数(x[1])降序排序；分数相同时Python的sort是稳定排序，
    # 会保留字典原有的相对顺序（也就是先出现在scores里的排前面，通常对应第一路排名更靠前的文档）
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


from retrieval.fuse_v2 import reciprocal_rank_fusion
