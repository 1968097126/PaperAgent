from __future__ import annotations

import pytest

from retrieval.config import RetrievalConfig
from retrieval.corpus import load_local_corpus
from retrieval.manifest import IndexVersionError, load_manifest, write_manifest


def test_duplicate_and_parse_failure_are_reported(tmp_path):
    (tmp_path / "one.txt").write_text("same non-empty paper text", encoding="utf-8")
    (tmp_path / "two.txt").write_text("same non-empty paper text", encoding="utf-8")
    (tmp_path / "broken.jsonl").write_text("{broken\n", encoding="utf-8")
    report = load_local_corpus(tmp_path)
    assert len(report.documents) == 1
    assert len(report.skipped_duplicates) == 1
    assert report.errors[0]["source"].endswith("broken.jsonl")


def test_manifest_version_mismatch_is_rejected(tmp_path):
    config = RetrievalConfig(corpus_dir=tmp_path)
    write_manifest({"index_version": config.index_version + 1}, config.manifest_path)
    with pytest.raises(IndexVersionError):
        load_manifest(config.manifest_path, config.index_version)
