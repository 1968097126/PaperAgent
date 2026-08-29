"""检索层：分块 - 召回 - 精排 三阶段混合检索流水线。

这个文件是 Python package 的标志文件（__init__.py）。一个目录下只要放了这个文件，
Python 就会把这个目录当成一个"包"（package），外面就能用
    from retrieval.dense_index import build_faiss_index
这种带路径的写法导入里面的模块，而不是把每个 .py 文件都堆在项目根目录用同一个命名空间。

本包后续会包含（按开发顺序，对应工作规划里的 Phase 1 / Phase 2）：
    chunking.py     Phase 1  论文正文切分成 chunk（滑动窗口 + overlap），并保留
                             paper_id / title / section 等元信息，供后面做"引用准确率"评测用
    dense_index.py  Phase 1  用 sentence-transformers 加载 BAAI/bge-m3，把 chunk 编码成向量，
                             用 faiss 建 IndexFlatIP（内积索引，配合归一化向量等价于余弦相似度）
    sparse_index.py Phase 1  用 rank_bm25 对 chunk 分词后建 BM25 倒排索引
    fuse.py          Phase 2  RRF（Reciprocal Rank Fusion）：把 dense_index 和 sparse_index
                             各自召回的排名列表融合成一个排名
    rerank.py        Phase 2  用 sentence-transformers 的 CrossEncoder（BAAI/bge-reranker-base）
                             对融合后的候选做精排，得到最终 top_k 证据

Phase 0 阶段这里先只放这份说明性的 __init__.py，具体实现留到 Phase 1/2 再写，
避免"目录里全是空文件却让人误以为功能已经做完"。
"""
