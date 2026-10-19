"""Reusable index build command for local research corpora."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.chunking import build_all_chunks, build_chunks_from_source
from retrieval.config import RetrievalConfig
from retrieval.dense_index import build_dense_index
from retrieval.manifest import (
    build_manifest,
    corpus_fingerprint,
    index_status,
    write_manifest,
)
from retrieval.sparse_index import build_bm25_index


def _portable_source(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"external:{resolved.name}"


def build_indexes(config: RetrievalConfig, *, source: Path | None = None,
                  force: bool = False) -> dict:
    if source is not None:
        chunks, load_report = build_chunks_from_source(source, config)
    else:
        chunks = build_all_chunks(
            config.papers_meta_path, config.corpus_dir / "papers" / "text",
            config.chunks_path, config=config,
        )
        load_report = {"documents": len({chunk["paper_id"] for chunk in chunks}),
                       "skipped_empty": [], "skipped_duplicates": [], "errors": []}
    if not chunks:
        raise ValueError("no usable paper text was found; inspect the corpus load report")

    fingerprint = corpus_fingerprint(chunks)
    current = index_status(config)
    if not force and current["ready"] and current["manifest"].get("corpus_fingerprint") == fingerprint:
        manifest = {**current["manifest"], "source": _portable_source(source or config.papers_meta_path)}
        write_manifest(manifest, config.manifest_path)
        return {"rebuilt": False, "load_report": load_report, **manifest}

    dense = build_dense_index(chunks, config=config)
    sparse = build_bm25_index(chunks, config=config)
    manifest = build_manifest(chunks, config)
    manifest["dense"] = dense
    manifest["sparse"] = sparse
    manifest["source"] = _portable_source(source or config.papers_meta_path)
    write_manifest(manifest, config.manifest_path)
    return {"rebuilt": True, "load_report": load_report, **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build versioned BGE-M3, FAISS, and BM25 indexes")
    parser.add_argument("--source", type=Path, help="PDF/Markdown/TXT/JSONL file or directory")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    parser.add_argument("--chunk-strategy", choices=("sliding", "paragraph"), default="sliding")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-chunk-chars", type=int, default=40)
    parser.add_argument("--dense-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RetrievalConfig(
        corpus_dir=args.corpus_dir, source_dir=args.source,
        chunk_strategy=args.chunk_strategy, chunk_size=args.chunk_size,
        overlap=args.overlap, min_chunk_chars=args.min_chunk_chars,
        dense_model=args.dense_model, reranker_model=args.reranker_model,
        batch_size=args.batch_size, device=args.device,
        cache_dir=args.cache_dir, random_seed=args.seed,
    )
    try:
        result = build_indexes(config, source=args.source, force=args.force)
    except Exception as exc:  # noqa: BLE001 - CLI must turn dependency/model failures into JSON
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
