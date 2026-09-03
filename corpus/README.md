# corpus/

存放原始论文语料 + 离线建库脚本。这个目录**不是** Python 包（没有 `__init__.py`），
因为里面的 `build_index.py`（Phase 1 添加）是一个独立运行的离线脚本——
用 `python corpus/build_index.py` 直接执行一次，把论文切块、编码、建索引，
不需要被其他模块 `import`，所以不用包的形式组织。

## 当前实现（Phase 1 已验证）

- `papers/`　配置指定的本地 PDF、Markdown 或 TXT 论文；也支持把 JSONL 作为输入
- `chunks.jsonl`　`retrieval.chunking` 切分后的结果：每行一个 chunk，
  带 `chunk_id` / `paper_id` / `title` / `section` / `text` 字段
- `faiss.index`　`retrieval.dense_index` 用 BGE-M3 编码后构建的稠密向量索引（FAISS）
- `bm25.pkl`　`retrieval.sparse_index` 构建的 BM25 稀疏索引（序列化后的 `BM25Okapi` 对象）
- `build_index.py`　离线建库脚本：读 `papers/` → 调用 `retrieval.chunking` 切块
  → 分别调用 `retrieval.dense_index` / `retrieval.sparse_index` 建两套索引 → 落盘

当前仓库的真实索引来自 30 篇本地论文正文，共生成 1,800 个 chunk；测试 fixture 不计入该统计。

## 建库

```bash
conda run -n nanocc python corpus/build_index.py --corpus-dir corpus --device cpu
```

指定另一份本地语料并强制重建：

```bash
conda run -n nanocc python corpus/build_index.py \
  --source /path/to/papers --corpus-dir corpus --chunk-strategy sliding \
  --chunk-size 1000 --overlap 200 --batch-size 16 --device cpu --force
```

索引产物包括：

- `chunks.jsonl`、`papers_meta.jsonl`
- `faiss.index`、`chunk_ids.json`
- `bm25.pkl`、`index_manifest.json`
