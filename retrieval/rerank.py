"""Cross-Encoder重排序：对召回+融合后的候选chunk做精排（"分块-召回-精排"的"精排"）。

为什么召回阶段（dense+sparse+RRF）不够、还需要单独一个精排阶段：
dense/sparse都是"双塔"式检索——query和chunk分别独立编码成向量，再比较向量的
相似度，这样做能支持对几千上万个chunk做快速检索（因为chunk的向量可以预先算好），
但代价是query和chunk之间没有真正"交互"过，语义匹配的精度有限。
Cross-Encoder不同：把query和chunk拼在一起一次性输入模型，让模型对两段文本做
逐token的双向注意力计算，能捕捉更细粒度的语义关联，精度明显更高，但计算量也
大得多（不能预先算好，每个候选都要现算一次），所以只能用来对"已经缩小到二三十个"
的候选做精排，不能直接对整个语料库做检索。
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"   # 智源开源的Cross-Encoder重排序模型，和BGE-M3同源配套

# 和dense_index.py的get_model()同样的模块级单例缓存思路，避免重复加载模型权重
_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """惰性加载并缓存 Cross-Encoder 重排序模型。"""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """对候选chunk按和query的相关性做Cross-Encoder精排。

    Args:
        query: 查询文本
        candidates: [(chunk_id, chunk_text), ...]，通常是RRF融合后的前20个候选
        top_k: 精排后保留几个
    Returns:
        [(chunk_id, rerank_score), ...]，按精排分数从高到低排序，只保留前top_k个。
        rerank_score是CrossEncoder的原始输出（未过sigmoid的logit，不是0~1的概率），
        只在同一批候选内部的相对大小有意义，用来跟其它候选比较排序，不代表绝对置信度。
    """
    if not candidates:
        return []

    model = get_reranker()
    # CrossEncoder.predict()接受一批(query, text)二元组，一次前向传播批量算完，
    # 比逐个候选单独调用快得多——这也是为什么精排只对"已经缩小的候选集"做，
    # 而不是对全部chunk做：批量算二三十个候选很快，批量算几千个就很慢了
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)

    scored = list(zip((cid for cid, _ in candidates), scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(cid, float(score)) for cid, score in scored[:top_k]]
