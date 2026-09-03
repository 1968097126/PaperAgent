"""Configuration shared by corpus building and online retrieval."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INDEX_FORMAT_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RetrievalConfig:
    """Serializable retrieval settings with portable, project-relative defaults."""

    corpus_dir: Path = PROJECT_ROOT / "corpus"
    source_dir: Path | None = None
    chunk_strategy: str = "sliding"
    chunk_size: int = 1000
    overlap: int = 200
    min_chunk_chars: int = 40
    dense_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"
    batch_size: int = 16
    device: str = "cpu"
    cache_dir: Path | None = None
    random_seed: int = 42
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 20
    result_top_k: int = 5
    max_second_retrievals: int = 1
    index_version: int = INDEX_FORMAT_VERSION

    def __post_init__(self) -> None:
        corpus_dir = Path(self.corpus_dir).expanduser().resolve()
        source_dir = Path(self.source_dir).expanduser().resolve() if self.source_dir else corpus_dir / "papers"
        cache_dir = Path(self.cache_dir).expanduser().resolve() if self.cache_dir else None
        object.__setattr__(self, "corpus_dir", corpus_dir)
        object.__setattr__(self, "source_dir", source_dir)
        object.__setattr__(self, "cache_dir", cache_dir)
        if self.chunk_strategy not in {"sliding", "paragraph"}:
            raise ValueError("chunk_strategy must be 'sliding' or 'paragraph'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        if self.min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be non-negative")
        for name in ("batch_size", "dense_top_k", "sparse_top_k", "rrf_k", "rerank_top_n", "result_top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_second_retrievals < 0:
            raise ValueError("max_second_retrievals must be non-negative")

    @property
    def chunks_path(self) -> Path:
        return self.corpus_dir / "chunks.jsonl"

    @property
    def papers_meta_path(self) -> Path:
        return self.corpus_dir / "papers_meta.jsonl"

    @property
    def dense_index_path(self) -> Path:
        return self.corpus_dir / "faiss.index"

    @property
    def dense_ids_path(self) -> Path:
        return self.corpus_dir / "chunk_ids.json"

    @property
    def sparse_index_path(self) -> Path:
        return self.corpus_dir / "bm25.pkl"

    @property
    def manifest_path(self) -> Path:
        return self.corpus_dir / "index_manifest.json"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("corpus_dir", "source_dir", "cache_dir"):
            if data[key] is not None:
                path = Path(data[key]).resolve()
                try:
                    data[key] = path.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    data[key] = f"external:{path.name}"
        return data

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> RetrievalConfig:
        values = dict(values or {})
        nested = values.get("retrieval")
        if isinstance(nested, Mapping):
            values = dict(nested)
        allowed = set(cls.__dataclass_fields__)
        kwargs = {key: value for key, value in values.items() if key in allowed}
        env_corpus = os.environ.get("NANOCC_CORPUS_DIR")
        env_cache = os.environ.get("NANOCC_MODEL_CACHE_DIR")
        if env_corpus and "corpus_dir" not in kwargs:
            kwargs["corpus_dir"] = env_corpus
        if env_cache and "cache_dir" not in kwargs:
            kwargs["cache_dir"] = env_cache
        return cls(**kwargs)
