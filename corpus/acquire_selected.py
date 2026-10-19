"""Download a reviewed paper list, preserving full texts and reproducible hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import fitz
import httpx

from corpus.pdf_text import extract_text_from_pdf


class CitationMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values: dict[str, list[str]] = {}

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if tag == "meta" and fields.get("name", "").startswith("citation_"):
            self.values.setdefault(fields["name"], []).append(fields.get("content", ""))


def acquire(paper: dict, root: Path) -> dict:
    pid = paper["paper_id"]
    pdf_path = root / "papers/pdf" / f"{pid}.pdf"
    source_path = root / "papers/metadata" / f"{pid}.html"
    text_path = root / "papers/text" / f"{pid}.txt"
    for path in (pdf_path, source_path, text_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        if not pdf_path.exists():
            response = client.get(paper["pdf_url"])
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"{pid}: download is not a PDF")
            pdf_path.write_bytes(response.content)
        # Metadata pages can be blocked even when the publisher PDF is public.
        if not source_path.exists():
            try:
                response = client.get(paper["source_url"])
                response.raise_for_status()
                source_path.write_text(response.text, encoding="utf-8")
            except httpx.HTTPError:
                pass
    body = extract_text_from_pdf(pdf_path)
    if len(body) < 2000:
        raise ValueError(f"{pid}: insufficient extracted text; no abstract fallback")
    text_path.write_text(body, encoding="utf-8")
    metadata = CitationMetadata()
    if source_path.exists():
        metadata.feed(source_path.read_text(encoding="utf-8"))
    with fitz.open(pdf_path) as doc:
        pages = len(doc)
    result = {
        **paper,
        "authors": metadata.values.get("citation_author", []),
        "abstract": "",
        "pages": pages,
        "text_chars": len(body),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "publisher_title": metadata.values.get("citation_title", [""])[0],
    }
    print(f"{pid}: {pages} pages, {len(body)} characters", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("corpus/selection.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus"))
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(lambda paper: acquire(paper, args.corpus_dir), selected["papers"])
        )
    # Publish metadata only after the entire frozen selection has been acquired.
    (args.corpus_dir / "papers_meta.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "corpus_id": selected["corpus_id"],
        "papers": len(rows),
        "pages": sum(row["pages"] for row in rows),
        "full_text_only": True,
    }
    (args.corpus_dir / "acquisition_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
