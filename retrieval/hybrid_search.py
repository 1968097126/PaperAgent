"""三阶段混合检索主入口："分块-召回-精排"流水线的最后一站，把前面几个模块串起来。

search_papers()：稠密召回(dense_index) + 稀疏召回(sparse_index) -> RRF融合(fuse) ->
                 Cross-Encoder精排(rerank)，返回最终证据列表。这是简历里
                 "search_papers"工具的核心实现；Phase 4 会把它包装成 ToolDef
                 注册进 tool_registry，供七阶段编排链路里的"混合检索"阶段调用。

retrieve_chunk()：按chunk_id查完整内容并可选取相邻chunk扩展上下文，对应简历里
                  "retrieve_chunk"工具——用于search_papers()给出的证据片段
                  因为chunk边界切得不巧、需要看更完整上下文的场景。
"""
from __future__ import annotations

import json

from retrieval.dense_index import search_dense
from retrieval.sparse_index import search_sparse
from retrieval.fuse import reciprocal_rank_fusion
from retrieval.rerank import rerank
from retrieval.chunking import CHUNKS_PATH

RECALL_TOP_K = 20   # dense召回和sparse召回各自取多少候选，喂给RRF融合
FUSE_TOP_K   = 20   # RRF融合后，取前多少名送去做Cross-Encoder精排（精排开销大，不能对全部候选做）

# chunk_id -> chunk字典 的内存缓存，search_papers/retrieve_chunk都要按id查chunk正文，
# 懒加载一次后常驻内存，避免每次调用都重新读一遍chunks.jsonl（语料库规模小，一次性
# 全量缓存完全可以接受；如果以后语料库规模涨到几十万chunk，这里要改成按需查询）
_chunk_store: dict[str, dict] | None = None


def _get_chunk_store() -> dict[str, dict]:
    global _chunk_store
    if _chunk_store is None:
        _chunk_store = {}
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                _chunk_store[c["chunk_id"]] = c
    return _chunk_store


def search_papers(query: str, top_k: int = 5) -> list[dict]:
    """三阶段混合检索：稠密召回 + 稀疏召回 -> RRF融合 -> Cross-Encoder精排。

    Args:
        query: 独立完整、不含未消解代词的检索查询（指代消解是更上游NLU阶段的职责，
               这里不做任何指代处理，传进来的query被当作已经是完整问题）
        top_k: 最终返回几条证据
    Returns:
        [{"chunk_id","paper_id","title","section","text","score"}, ...]，
        按精排分数从高到低排序；query为空或语料库无匹配时返回空列表
    """
    if not query or not query.strip():
        return []

    store = _get_chunk_store()
    dense_hits  = search_dense(query, top_k=RECALL_TOP_K)
    sparse_hits = search_sparse(query, top_k=RECALL_TOP_K)

    fused = reciprocal_rank_fusion([
        [cid for cid, _ in dense_hits],
        [cid for cid, _ in sparse_hits],
    ])
    candidates = fused[:FUSE_TOP_K]

    reranked = rerank(
        query,
        [(cid, store[cid]["text"]) for cid, _ in candidates if cid in store],
        top_k=top_k,
    )

    results = []
    for cid, score in reranked:
        c = store[cid]
        results.append({
            "chunk_id": cid,
            "paper_id": c["paper_id"],
            "title":    c["title"],
            "section":  c["section"],
            "text":     c["text"],
            "score":    score,
        })
    return results


def retrieve_chunk(chunk_id: str, context: int = 1) -> dict | None:
    """按chunk_id取回完整chunk内容，并带上同一篇论文里相邻的chunk扩展上下文。

    典型用途：search_papers()返回的证据片段可能因为chunk切分边界正好切在
    关键论述中间，需要看更完整上下文时，用这个函数按chunk_id展开。

    Args:
        chunk_id: search_papers()返回结果里的chunk_id
        context: 在目标chunk前后各取多少个相邻chunk（按chunk_index顺序），默认1
    Returns:
        {"chunk_id","paper_id","title","section","text"}，text是拼接了上下文后的正文；
        找不到该chunk_id时返回None
    """
    store = _get_chunk_store()
    target = store.get(chunk_id)
    if target is None:
        return None

    paper_id = target["paper_id"]
    center_idx = target["chunk_index"]
    # 筛出同一篇论文里、chunk_index落在[center-context, center+context]区间的chunk，
    # 按chunk_index排序后拼接，还原出一段比单个chunk更完整的上下文
    neighbors = sorted(
        (c for c in store.values()
         if c["paper_id"] == paper_id and abs(c["chunk_index"] - center_idx) <= context),
        key=lambda c: c["chunk_index"],
    )

    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title":    target["title"],
        "section":  target["section"],
        "text":     "\n\n".join(c["text"] for c in neighbors),
    }


from retrieval.hybrid_v2 import HybridSearcher, get_searcher, retrieve_chunk, search_papers
