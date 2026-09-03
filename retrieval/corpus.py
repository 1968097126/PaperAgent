"""Load local research papers from the formats used by this project."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".jsonl"}


@dataclass
class CorpusLoadReport:
    documents: list[dict] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    skipped_duplicates: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents": len(self.documents),
            "skipped_empty": self.skipped_empty,
            "skipped_duplicates": self.skipped_duplicates,
            "errors": self.errors,
        }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_paper_id(source: str, title: str, text: str) -> str:
    material = "\x1f".join((source, title.strip(), normalize_text(text))).encode("utf-8")
    return "paper-" + hashlib.sha256(material).hexdigest()[:16]


def _pdf_text(path: Path) -> str:
    import pymupdf

    parts: list[str] = []
    with pymupdf.open(path) as document:
        for page in document:
            parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)


def _markdown_title(text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else fallback


def _records_from_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} must contain a JSON object")
            yield value


def _document_from_record(record: dict, source: Path, suffix: str = "") -> dict:
    text = record.get("text") or record.get("content") or record.get("abstract") or ""
    title = str(record.get("title") or source.stem).strip()
    source_value = str(record.get("source") or record.get("pdf_url") or source)
    paper_id = str(record.get("paper_id") or stable_paper_id(source_value + suffix, title, text))
    return {
        **record,
        "paper_id": paper_id,
        "title": title,
        "source": source_value,
        "text": normalize_text(str(text)),
    }


def load_local_corpus(source: Path) -> CorpusLoadReport:
    """Load a file or directory, retaining parse failures in a structured report."""
    source = Path(source).expanduser().resolve()
    report = CorpusLoadReport()
    if not source.exists():
        report.errors.append({"source": str(source), "error": "path does not exist"})
        return report
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file()
        and not path.name.casefold().startswith("readme") and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for path in paths:
        try:
            if path.suffix.lower() == ".jsonl":
                documents = [
                    _document_from_record(record, path, f":{index}")
                    for index, record in enumerate(_records_from_jsonl(path))
                ]
            else:
                raw = _pdf_text(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
                title = _markdown_title(raw, path.stem) if path.suffix.lower() == ".md" else path.stem
                documents = [_document_from_record({"title": title, "text": raw}, path)]
        except Exception as exc:  # noqa: BLE001 - one bad paper must not abort a batch
            report.errors.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        for document in documents:
            if not document["text"]:
                report.skipped_empty.append(document["source"])
                continue
            text_hash = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
            if document["paper_id"] in seen_ids or text_hash in seen_text:
                report.skipped_duplicates.append(document["source"])
                continue
            seen_ids.add(document["paper_id"])
            seen_text.add(text_hash)
            report.documents.append(document)
    return report
