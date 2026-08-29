# corpus/

存放原始论文语料 + 离线建库脚本。这个目录**不是** Python 包（没有 `__init__.py`），
因为里面的 `build_index.py`（Phase 1 添加）是一个独立运行的离线脚本——
用 `python corpus/build_index.py` 直接执行一次，把论文切块、编码、建索引，
不需要被其他模块 `import`，所以不用包的形式组织。

## 后续规划（Phase 1）

- `papers/`　原始论文文件（PDF/HTML/txt），30~80 篇，自建的小规模示例集
- `chunks.jsonl`　`retrieval.chunking` 切分后的结果：每行一个 chunk，
  带 `chunk_id` / `paper_id` / `title` / `section` / `text` 字段
- `faiss.index`　`retrieval.dense_index` 用 BGE-M3 编码后构建的稠密向量索引（FAISS）
- `bm25.pkl`　`retrieval.sparse_index` 构建的 BM25 稀疏索引（序列化后的 `BM25Okapi` 对象）
- `build_index.py`　离线建库脚本：读 `papers/` → 调用 `retrieval.chunking` 切块
  → 分别调用 `retrieval.dense_index` / `retrieval.sparse_index` 建两套索引 → 落盘

Phase 0 阶段先只建这个空目录 + 这份说明，实际论文和建库脚本在 Phase 1 加入。
