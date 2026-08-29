"""检索层：分块 - 召回 - 精排 三阶段混合检索流水线。

这个文件是 Python package 的标志文件（__init__.py）。一个目录下只要放了这个文件，
Python 就会把这个目录当成一个"包"（package），外面就能用
    from retrieval.dense_index import build_faiss_index
这种带路径的写法导入里面的模块，而不是把每个 .py 文件都堆在项目根目录用同一个命名空间。

本包包含（Phase 1 / Phase 2 已完成）：
    chunking.py      论文正文切分成 chunk（按物理行打包+overlap），保留paper_id/title/section
                     等元信息，供后面做"引用准确率"评测用；也负责建索引前的通用chunk读取
    dense_index.py   用 sentence-transformers 加载 BAAI/bge-m3 把 chunk 编码成向量，
                     用 faiss 建 IndexFlatIP 索引（build_dense_index），并提供 search_dense()
                     查询接口
    sparse_index.py  用 rank_bm25 对 chunk 分词建 BM25 索引（build_bm25_index），并提供
                     search_sparse() 查询接口
    fuse.py          RRF（Reciprocal Rank Fusion）：把dense/sparse各自召回的排名列表
                     融合成一个排名，只看排名不看原始分数（两边分数量纲不同不可比）
    rerank.py        用 CrossEncoder（BAAI/bge-reranker-base）对融合后的候选做精排
    hybrid_search.py 主入口：search_papers() 把 dense_search+sparse_search -> fuse ->
                     rerank 串成完整的"分块-召回-精排"三阶段流水线；retrieve_chunk() 按
                     chunk_id 查完整内容并扩展相邻上下文

Phase 4（七阶段编排链路）会把 search_papers/retrieve_chunk 包装成 tool_registry 里的
ToolDef（已在 tools.py 里注册），由 pipeline/orchestrator.py 按固定顺序调用。
"""
