#!/usr/bin/env python3
"""Phase 1 总入口：分块 -> 建稠密索引(FAISS) -> 建稀疏索引(BM25)。

前置条件：corpus/build_corpus.py 已经跑过，corpus/papers_meta.jsonl 和
corpus/papers/text/*.txt 已经存在（本脚本不负责抓论文，只负责"从已有正文
到可检索的索引"这一段）。

用法：
    python corpus/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 sys.path：这样不管从哪个目录执行这个脚本，
# `from retrieval.chunking import build_all_chunks` 这种"从项目根开始"的导入路径都能找到。
# （直接 `python corpus/build_index.py` 运行时，Python默认只把 corpus/ 目录本身加进sys.path，
#  不会自动包含它的上一级，所以要手动加，否则 import retrieval.xxx 会报 ModuleNotFoundError）
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.chunking import build_all_chunks
from retrieval.dense_index import build_dense_index
from retrieval.sparse_index import build_bm25_index


def main() -> None:
    print("=" * 60)
    print("[1/3] 分块")
    print("=" * 60)
    chunks = build_all_chunks()

    if not chunks:
        print("没有生成任何chunk，请先确认 corpus/build_corpus.py 已经成功跑过。")
        return

    print("\n" + "=" * 60)
    print("[2/3] 构建稠密索引（BGE-M3 + FAISS）")
    print("=" * 60)
    build_dense_index(chunks)   # 直接传内存里的chunks，不用重新读盘

    print("\n" + "=" * 60)
    print("[3/3] 构建稀疏索引（BM25）")
    print("=" * 60)
    build_bm25_index(chunks)

    print("\n全部完成。产出文件：")
    print("  corpus/chunks.jsonl   分块结果")
    print("  corpus/faiss.index    稠密向量索引")
    print("  corpus/chunk_ids.json 索引编号 -> chunk_id 映射")
    print("  corpus/bm25.pkl       BM25稀疏索引")


if __name__ == "__main__":
    main()
