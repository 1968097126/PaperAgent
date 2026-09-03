"""Versioned BM25 indexing with mixed Chinese/English tokenization."""
from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from retrieval.chunking_v2 import load_chunks
from retrieval.config import RetrievalConfig
from retrieval.manifest import corpus_fingerprint, load_manifest

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)?")
ZH_PATTERN = re.compile(r"[\u3400-\u9fff]")


def tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = WORD_PATTERN.findall(lowered)
    if ZH_PATTERN.search(lowered):
        try:
            import jieba  # type: ignore[import-untyped]
            tokens.extend(token.strip() for token in jieba.lcut(lowered) if token.strip() and ZH_PATTERN.search(token))
        except ImportError:
            tokens.extend(ZH_PATTERN.findall(lowered))
    return tokens


def build_bm25_index(chunks: list[dict] | None = None, *,
                     config: RetrievalConfig | None = None) -> dict:
    cfg = config or RetrievalConfig()
    chunks = chunks if chunks is not None else load_chunks(cfg.chunks_path)
    if not chunks:
        raise ValueError("cannot build BM25 index from an empty corpus")
    tokenized = [tokenize(chunk["text"]) or ["__empty__"] for chunk in chunks]
    payload = {
        "version": cfg.index_version,
        "corpus_fingerprint": corpus_fingerprint(chunks),
        "bm25": BM25Okapi(tokenized),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
    }
    with cfg.sparse_index_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return {"documents": len(chunks), "token_count": sum(map(len, tokenized))}


def load_bm25_index(*, config: RetrievalConfig | None = None,
                    validate_manifest: bool = True) -> tuple[BM25Okapi, list[str]]:
    cfg = config or RetrievalConfig()
    manifest = load_manifest(cfg.manifest_path, cfg.index_version) if validate_manifest else None
    with cfg.sparse_index_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != cfg.index_version:
        raise ValueError("BM25 index version is incompatible")
    ids = payload.get("chunk_ids")
    if not isinstance(ids, list):
        raise TypeError("BM25 index is missing chunk IDs")
    if manifest and payload.get("corpus_fingerprint") != manifest.get("corpus_fingerprint"):
        raise ValueError("BM25 index does not match the corpus manifest")
    return payload["bm25"], [str(value) for value in ids]


def search_sparse(query: str, top_k: int = 20, *,
                  config: RetrievalConfig | None = None) -> list[tuple[str, float]]:
    query = (query or "").strip()
    if not query:
        return []
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    tokens = tokenize(query)
    if not tokens:
        return []
    bm25, ids = load_bm25_index(config=config)
    hits = [(chunk_id, float(score)) for chunk_id, score in zip(ids, bm25.get_scores(tokens)) if float(score) > 0]
    hits.sort(key=lambda hit: (-hit[1], hit[0]))
    return hits[:top_k]
