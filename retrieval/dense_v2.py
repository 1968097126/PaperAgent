"""Configurable BGE-M3 embedding and FAISS search."""
from __future__ import annotations

import json
import random
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.chunking_v2 import load_chunks
from retrieval.config import RetrievalConfig
from retrieval.manifest import load_manifest

_models: dict[tuple[str, str, str], SentenceTransformer] = {}


def get_model(model_name: str = "BAAI/bge-m3", device: str = "cpu",
              cache_dir: Path | None = None) -> SentenceTransformer:
    key = (model_name, device, str(cache_dir or ""))
    if key not in _models:
        _models[key] = SentenceTransformer(
            model_name, device=device, cache_folder=str(cache_dir) if cache_dir else None
        )
    return _models[key]


def embed_texts(texts: list[str], batch_size: int = 16, *,
                model_name: str = "BAAI/bge-m3", device: str = "cpu",
                cache_dir: Path | None = None, random_seed: int = 42,
                show_progress: bool = False) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch
        torch.manual_seed(random_seed)
    except ImportError:
        pass
    embeddings = get_model(model_name, device, cache_dir).encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_dense_index(chunks: list[dict] | None = None, *,
                      config: RetrievalConfig | None = None,
                      embedder=embed_texts) -> dict:
    cfg = config or RetrievalConfig()
    chunks = chunks if chunks is not None else load_chunks(cfg.chunks_path)
    if not chunks:
        raise ValueError("cannot build dense index from an empty corpus")
    vectors = embedder(
        [chunk["text"] for chunk in chunks], batch_size=cfg.batch_size,
        model_name=cfg.dense_model, device=cfg.device, cache_dir=cfg.cache_dir,
        random_seed=cfg.random_seed, show_progress=True,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
        raise ValueError("embedder returned an invalid matrix")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype="float32"))
    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.dense_index_path))
    cfg.dense_ids_path.write_text(
        json.dumps([chunk["chunk_id"] for chunk in chunks], ensure_ascii=False), encoding="utf-8"
    )
    return {"vectors": int(index.ntotal), "dimension": int(vectors.shape[1])}


def load_dense_index(*, config: RetrievalConfig | None = None,
                     validate_manifest: bool = True) -> tuple[faiss.Index, list[str]]:
    cfg = config or RetrievalConfig()
    if validate_manifest:
        load_manifest(cfg.manifest_path, cfg.index_version)
    index = faiss.read_index(str(cfg.dense_index_path))
    ids = json.loads(cfg.dense_ids_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or index.ntotal != len(ids):
        raise ValueError("FAISS vector count does not match chunk_ids.json")
    return index, [str(value) for value in ids]


def search_dense(query: str, top_k: int = 20, *, config: RetrievalConfig | None = None,
                 embedder=embed_texts) -> list[tuple[str, float]]:
    cfg = config or RetrievalConfig()
    query = (query or "").strip()
    if not query:
        return []
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    index, ids = load_dense_index(config=cfg)
    vector = embedder(
        [query], batch_size=cfg.batch_size, model_name=cfg.dense_model,
        device=cfg.device, cache_dir=cfg.cache_dir, random_seed=cfg.random_seed,
        show_progress=False,
    )
    scores, indices = index.search(np.asarray(vector, dtype="float32"), min(top_k, len(ids)))
    hits = [(ids[index_value], float(score)) for score, index_value in zip(scores[0], indices[0]) if index_value >= 0]
    return sorted(hits, key=lambda hit: (-hit[1], hit[0]))
