"""Phase 1：用 BM25 给chunk建稀疏（关键词）索引，和 dense_index.py 是"召回"阶段的另一条腿。

为什么稠密检索之外还要BM25：稠密向量擅长捕捉语义相似（"降低方差"和"variance reduction"
能匹配上），但对精确的专有名词/缩写（比如论文里的算法名"PPO"、"SAC"）有时反而不如
关键词精确匹配敏感。混合检索（dense+sparse）比单用其中一种召回率更高，这也是简历里
"结合稠密检索与稀疏检索，通过RRF融合"这句话的动机来源。RRF融合本身是Phase 2的工作，
这里只负责把BM25索引建出来、能读回来。
"""
from __future__ import annotations

import re
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi   # 纯Python实现的BM25算法库

from retrieval.chunking import load_chunks, CHUNKS_PATH

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
BM25_PATH  = CORPUS_DIR / "bm25.pkl"

# 匹配连续2个及以上的英文字母作为一个token（数字、标点都不算token的一部分）。
# 语料库目前是纯英文的强化学习论文，这个简单正则分词器已经够用；
# BM25算法本身是基于词频统计的，对分词粒度不算敏感，不需要上重量级的分词器。
# 如果以后语料库混入中文摘要/标题，可以在这里判断文本是否含中文字符，
# 含中文的部分改用 jieba.lcut() 分词——requirements.txt里预留的jieba依赖就是为这个场景准备的。
_TOKEN_PATTERN = re.compile(r"[a-zA-Z]{2,}")


def tokenize(text: str) -> list[str]:
    """把一段文本切成小写单词列表，供BM25统计词频用。"""
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


def build_bm25_index(chunks: list[dict] | None = None) -> None:
    """读取全部chunk，分词后建BM25索引并用pickle落盘。

    为什么用pickle而不是像FAISS那样有专门的write_index：BM25Okapi是一个普通Python对象
    （内部存着词频统计表等数据结构），rank_bm25库本身没有提供文件序列化接口，
    pickle是Python标准库里"把任意对象存成二进制文件、以后原样读回来"的通用方案。
    """
    if chunks is None:
        chunks = load_chunks(CHUNKS_PATH)

    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    print(f"[sparse_index] 对 {len(tokenized_corpus)} 个chunk分词并构建BM25索引……")
    bm25 = BM25Okapi(tokenized_corpus)

    # 索引对象和chunk_ids必须绑在一起存：BM25Okapi内部是按"语料库列表的顺序"存储统计信息的，
    # 后面查询时 bm25.get_scores(query_tokens) 返回的分数数组也是按这个顺序排列，
    # 所以"第i个分数对应哪个chunk_id"必须靠这个顺序一致的chunk_ids列表才能翻译回去。
    with open(BM25_PATH, "wb") as f:   # "wb" = 写二进制模式，pickle存的是字节流不是文本
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    print(f"[sparse_index] BM25索引已写入 {BM25_PATH}")


def load_bm25_index() -> tuple[BM25Okapi, list[str]]:
    """读回build_bm25_index()落盘的索引，供 Phase 2 的检索逻辑调用 bm25.get_scores(tokens)。"""
    with open(BM25_PATH, "rb") as f:   # "rb" = 读二进制模式，要和写入时的模式对应
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


if __name__ == "__main__":
    build_bm25_index()
