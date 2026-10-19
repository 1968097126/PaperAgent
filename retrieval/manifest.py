"""Index versioning, corpus fingerprints, and rebuild checks."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.model_versions import MODEL_REVISIONS


class IndexVersionError(RuntimeError):
    pass


def corpus_fingerprint(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item["chunk_id"]):
        digest.update(json.dumps(chunk, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(chunks: list[dict], config: RetrievalConfig) -> dict:
    return {
        "index_version": config.index_version,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_fingerprint": corpus_fingerprint(chunks),
        "paper_count": len({chunk["paper_id"] for chunk in chunks}),
        "chunk_count": len(chunks),
        "dense_model": config.dense_model,
        "model_revisions": MODEL_REVISIONS,
        "reranker_model": config.reranker_model,
        "chunk_strategy": config.chunk_strategy,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "random_seed": config.random_seed,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    Path(path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path, expected_version: int | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise IndexVersionError(f"index manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if expected_version is not None and manifest.get("index_version") != expected_version:
        raise IndexVersionError(
            f"index version {manifest.get('index_version')!r} is incompatible with {expected_version}"
        )
    return manifest


def index_status(config: RetrievalConfig) -> dict:
    required = {
        "chunks": config.chunks_path,
        "dense": config.dense_index_path,
        "dense_ids": config.dense_ids_path,
        "sparse": config.sparse_index_path,
        "manifest": config.manifest_path,
    }
    present = {name: path.exists() for name, path in required.items()}
    manifest: dict = {}
    error = None
    if present["manifest"]:
        try:
            manifest = load_manifest(config.manifest_path, config.index_version)
        except (OSError, ValueError, IndexVersionError) as exc:
            error = str(exc)
    stale = True
    if all(present.values()) and not error:
        try:
            from retrieval.chunking import load_chunks
            stale = corpus_fingerprint(load_chunks(config.chunks_path)) != manifest.get("corpus_fingerprint")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
    return {"ready": all(present.values()) and not stale and not error,
            "stale": stale, "files": present, "manifest": manifest, "error": error}
