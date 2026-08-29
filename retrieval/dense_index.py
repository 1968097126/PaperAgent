"""用 BGE-M3 把chunk编码成稠密向量，用FAISS建索引和检索（"分块-召回-精排"里的"召回"半边）。

Phase 1 实现了离线建索引（build_dense_index）和索引读取（load_dense_index）；
Phase 2 在此基础上加了 search_dense()——真正拿一个query去检索最相似chunk的
查询接口，复用同一套 get_model() / embed_texts() 编码逻辑，保证query向量和
建索引时chunk向量落在同一个向量空间里（这是能够比较相似度的前提）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss                                   # Facebook AI Similarity Search：向量近邻检索库
from sentence_transformers import SentenceTransformer

from retrieval.chunking import load_chunks, CHUNKS_PATH

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
INDEX_PATH = CORPUS_DIR / "faiss.index"     # FAISS索引文件本身只存"向量"，不存文本
IDS_PATH   = CORPUS_DIR / "chunk_ids.json"  # 所以另存一份"第i个向量对应哪个chunk_id"的映射

MODEL_NAME = "BAAI/bge-m3"   # 智源开源的多语言/多粒度embedding模型，简历里点名的稠密检索模型

# 模块级"单例"：SentenceTransformer加载模型要读取几个GB的权重文件，很慢，
# 用一个模块级变量缓存已加载的模型，同一进程内多次调用get_model()只加载一次。
# （对比：如果每次embed_texts()内部都new一个SentenceTransformer()，建索引时会重复加载模型，
#  查询时如果每个query都重新加载模型，延迟会高到无法接受。）
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """惰性加载并缓存 BGE-M3 模型。"""
    global _model   # 声明要修改的是模块级变量_model，而不是在函数内新建一个同名局部变量
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """把一批文本编码成向量矩阵，形状 (len(texts), embedding维度)。

    normalize_embeddings=True：把每个向量都缩放成单位长度（L2范数=1）。
    这样做的意义是——两个单位向量的内积 = 两个向量夹角的余弦值，
    所以"归一化向量 + IndexFlatIP(内积索引)"就完全等价于"用余弦相似度检索"，
    是稠密检索最常见的标准组合（比直接用未归一化的内积更符合直觉、更稳定）。
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,     # 编码几百上千个chunk时在终端打印进度条，方便观察是否卡住
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")   # faiss要求输入向量是float32类型，sentence-transformers默认可能是float64


def build_dense_index(chunks: list[dict] | None = None) -> None:
    """读取全部chunk，编码后建FAISS索引并落盘。

    Args:
        chunks: 如果调用方已经在内存里有chunk列表（比如 build_index.py 里刚跑完分块），
                可以直接传进来，省一次读盘；不传就自己从 CHUNKS_PATH 读。
    """
    if chunks is None:
        chunks = load_chunks(CHUNKS_PATH)

    texts     = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print(f"[dense_index] 用 {MODEL_NAME} 编码 {len(texts)} 个chunk……")
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]   # embeddings.shape是(N,D)元组，[1]取出向量维度D（BGE-M3是1024维）
    # IndexFlatIP：不做任何近似/量化的暴力内积检索索引。"Flat"=不压缩、"IP"=Inner Product(内积)。
    # 检索时对query向量和索引里每一条都算一次内积，语料库几千条规模下这个开销可以忽略，
    # 换来的是100%精确的检索结果（不像HNSW/IVF那样为了速度牺牲一点召回率）。
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)       # 把所有向量塞进索引；faiss内部按插入顺序给每条向量编号0..N-1

    faiss.write_index(index, str(INDEX_PATH))
    # index里的向量只有编号(0..N-1)，没有chunk_id这种业务语义的信息，
    # 所以额外存一份"编号 -> chunk_id"的列表（按插入顺序，第i个元素就是编号i对应的chunk_id）
    IDS_PATH.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")

    print(f"[dense_index] FAISS索引已写入 {INDEX_PATH}"
          f"（{index.ntotal}条向量，维度{dim}），id映射写入 {IDS_PATH}")


def load_dense_index() -> tuple[faiss.Index, list[str]]:
    """把build_dense_index()落盘的索引读回内存，返回(索引对象, 编号->chunk_id映射)。

    Phase 2 的检索逻辑会调用这个函数：拿到index后用 index.search(query_vec, k)
    做近邻检索，返回的是"编号"，再用这里返回的chunk_ids列表把编号翻译回chunk_id。
    """
    index = faiss.read_index(str(INDEX_PATH))
    chunk_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    return index, chunk_ids


def search_dense(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """稠密检索：把query编码成向量，去FAISS索引里找最相似的top_k个chunk。

    Args:
        query: 查询文本（应是独立完整的问题，指代消解在更上游的NLU阶段做）
        top_k: 返回多少个候选
    Returns:
        [(chunk_id, score), ...]，按相似度从高到低排序。score就是FAISS返回的内积值，
        因为embed_texts()和build_dense_index()两边的向量都做过归一化，内积在数值上
        等价于余弦相似度，理论范围是[-1, 1]，语义相关的学术文本通常落在0.3~0.8之间。
    """
    index, chunk_ids = load_dense_index()
    query_vec = embed_texts([query])   # 形状(1, D)，复用建索引时同一套编码逻辑，保证向量空间一致

    # faiss的search()是批量接口：输入N个query一次性检索，这里只有1个query，
    # 所以返回的scores/idxs形状都是(1, top_k)，取第0行拿到这一个query的结果
    scores, idxs = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:   # 索引里实际数据不足top_k条时，faiss用-1填充占位，要过滤掉
            continue
        results.append((chunk_ids[idx], float(score)))
    return results


if __name__ == "__main__":
    build_dense_index()
