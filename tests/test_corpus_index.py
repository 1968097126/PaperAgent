from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.chunking_v2 import build_chunks_for_paper, build_chunks_from_source
from retrieval.config import RetrievalConfig
from retrieval.corpus import load_local_corpus
from retrieval.dense_v2 import build_dense_index, load_dense_index, search_dense
from retrieval.manifest import build_manifest, index_status, write_manifest
from retrieval.sparse_v2 import (
    build_bm25_index,
    load_bm25_index,
    search_sparse,
    tokenize,
)

FIXTURE = Path(__file__).parent / "fixtures" / "corpus"


def fake_embed(texts, **_kwargs):
    rows = []
    for text in texts:
        lowered = text.lower()
        row = np.array([
            1.0 if "policy" in lowered else 0.0,
            1.0 if "value" in lowered else 0.0,
            1.0 if "安全" in text else 0.0,
        ], dtype="float32")
        if not row.any():
            row[0] = 0.1
        row /= np.linalg.norm(row)
        rows.append(row)
    return np.stack(rows)


def test_local_loader_handles_formats_empty_and_mixed_text():
    report = load_local_corpus(FIXTURE)
    assert len(report.documents) == 3
    assert report.skipped_empty == ["fixture://empty"]
    assert any("安全" in document["text"] for document in report.documents)
    assert all({"paper_id", "title", "source", "text"} <= set(document) for document in report.documents)


def test_chunk_ids_are_stable_and_metadata_is_preserved():
    meta = {"paper_id": "p1", "title": "Paper", "source": "fixture://p1"}
    text = "Abstract\n" + "policy optimization evidence. " * 20
    first = build_chunks_for_paper(meta, text, chunk_size=120, overlap=20, min_chunk_chars=20)
    second = build_chunks_for_paper(meta, text, chunk_size=120, overlap=20, min_chunk_chars=20)
    assert first == second
    assert len(first) > 1
    assert all(chunk["source"] == "fixture://p1" for chunk in first)
    assert all(chunk["chunk_id"].startswith("p1:") for chunk in first)


def test_fixture_index_round_trip_and_rebuild_status(tmp_path):
    config = RetrievalConfig(corpus_dir=tmp_path, source_dir=FIXTURE, chunk_size=140, overlap=20,
                             min_chunk_chars=10, dense_model="fixture-embedder")
    chunks, report = build_chunks_from_source(FIXTURE, config)
    assert report["documents"] == 3
    dense = build_dense_index(chunks, config=config, embedder=fake_embed)
    sparse = build_bm25_index(chunks, config=config)
    manifest = build_manifest(chunks, config)
    manifest.update({"dense": dense, "sparse": sparse})
    write_manifest(manifest, config.manifest_path)

    index, ids = load_dense_index(config=config)
    bm25, sparse_ids = load_bm25_index(config=config)
    assert index.ntotal == len(chunks) == len(ids) == len(sparse_ids)
    assert bm25 is not None
    assert search_dense("policy", 2, config=config, embedder=fake_embed)[0][0] in ids
    assert search_sparse("安全", 3, config=config)
    assert index_status(config)["ready"] is True

    with config.chunks_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**chunks[0], "chunk_id": "changed"}) + "\n")
    assert index_status(config)["stale"] is True


def test_mixed_language_tokenization():
    tokens = tokenize("BGE-M3 支持中文 retrieval 2026")
    assert "bge-m3" in tokens
    assert "retrieval" in tokens
    assert any("中文" in token or token in {"中", "文"} for token in tokens)
