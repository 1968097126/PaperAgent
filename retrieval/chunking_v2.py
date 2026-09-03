"""Configurable deterministic chunking used by the research pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.corpus import load_local_corpus, normalize_text

EXCLUDED_SECTIONS = {"references", "appendix", "参考文献", "附录"}
SECTION_PATTERN = re.compile(
    r"^(?:(?:\d{1,2}|[IVX]{1,5})[.)]?\s*)?"
    r"(Abstract|Introduction|Related\s+Work|Background|Preliminaries|"
    r"Method(?:ology)?|Approach|Experiments?|Results?|Evaluation|"
    r"Discussion|Conclusion|Limitations|References|Appendix|"
    r"摘要|引言|相关工作|背景|方法|实验|结果|讨论|结论|局限|参考文献|附录)\s*$",
    re.IGNORECASE,
)


def split_into_lines_with_sections(raw_text: str) -> list[tuple[str, str]]:
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", normalize_text(raw_text))
    section = "Unknown"
    lines: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SECTION_PATTERN.fullmatch(line) if len(line) <= 60 else None
        if match:
            section = match.group(1).title()
        lines.append((line, section))
    return lines


def _window_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            yield piece
        if start + chunk_size >= len(text):
            break


def split_into_chunks(
    lines: list[tuple[str, str]], chunk_size: int = 1000, overlap: int = 200,
    strategy: str = "sliding", min_chunk_chars: int = 0,
) -> list[tuple[str, str]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    if strategy not in {"sliding", "paragraph"}:
        raise ValueError("strategy must be 'sliding' or 'paragraph'")
    if not lines:
        return []

    sections: list[tuple[str, str]] = []
    section = lines[0][1]
    buffer: list[str] = []
    for text, current_section in lines:
        if current_section != section and buffer:
            sections.append(("\n".join(buffer), section))
            buffer = []
        section = current_section
        buffer.append(text)
    if buffer:
        sections.append(("\n".join(buffer), section))

    pieces: list[tuple[str, str]] = []
    for section_text, section_name in sections:
        if section_name.casefold() in {value.casefold() for value in EXCLUDED_SECTIONS}:
            continue
        if strategy == "sliding":
            flattened = re.sub(r"\s+", " ", section_text).strip()
            pieces.extend((piece, section_name) for piece in _window_text(flattened, chunk_size, overlap))
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.)\s*\n", section_text) if p.strip()]
        packed = ""
        for paragraph in paragraphs:
            candidate = f"{packed}\n\n{paragraph}" if packed else paragraph
            if len(candidate) <= chunk_size:
                packed = candidate
            else:
                if packed:
                    pieces.append((packed, section_name))
                if len(paragraph) > chunk_size:
                    pieces.extend((piece, section_name) for piece in _window_text(paragraph, chunk_size, overlap))
                    packed = ""
                else:
                    packed = paragraph
        if packed:
            pieces.append((packed, section_name))

    if min_chunk_chars <= 0:
        return pieces
    kept: list[tuple[str, str]] = []
    for text, section_name in pieces:
        if len(text) >= min_chunk_chars:
            kept.append((text, section_name))
        elif kept and kept[-1][1] == section_name:
            previous, previous_section = kept[-1]
            kept[-1] = (f"{previous} {text}".strip(), previous_section)
    return kept


def stable_chunk_id(paper_id: str, chunk_index: int, section: str, text: str) -> str:
    digest = hashlib.sha256(f"{paper_id}\x1f{section}\x1f{text}".encode()).hexdigest()[:12]
    return f"{paper_id}:{chunk_index:04d}:{digest}"


def build_chunks_for_paper(
    paper_meta: dict, full_text: str, *, chunk_size: int = 1000, overlap: int = 200,
    strategy: str = "sliding", min_chunk_chars: int = 0,
) -> list[dict]:
    text = normalize_text(full_text)
    if not text:
        return []
    pieces = split_into_chunks(
        split_into_lines_with_sections(text), chunk_size, overlap, strategy, min_chunk_chars
    )
    paper_id = str(paper_meta["paper_id"])
    source = str(paper_meta.get("source") or paper_meta.get("pdf_url") or "")
    return [
        {
            "chunk_id": stable_chunk_id(paper_id, index, section, piece),
            "paper_id": paper_id,
            "title": str(paper_meta.get("title") or paper_id),
            "section": section,
            "source": source,
            "chunk_index": index,
            "text": piece,
        }
        for index, (piece, section) in enumerate(pieces)
    ]


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {"chunk_id", "paper_id", "title", "section", "chunk_index", "text"}
            missing = required - set(value)
            if missing:
                raise ValueError(f"chunk line {line_number} missing fields: {sorted(missing)}")
            chunks.append(value)
    return chunks


def write_chunks(chunks: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")


def build_chunks_from_source(source: Path, config: RetrievalConfig) -> tuple[list[dict], dict]:
    report = load_local_corpus(source)
    config.corpus_dir.mkdir(parents=True, exist_ok=True)
    with config.papers_meta_path.open("w", encoding="utf-8") as handle:
        for document in report.documents:
            metadata = {key: value for key, value in document.items() if key != "text"}
            handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
    chunks: list[dict] = []
    for document in report.documents:
        chunks.extend(build_chunks_for_paper(
            document, document["text"], chunk_size=config.chunk_size, overlap=config.overlap,
            strategy=config.chunk_strategy, min_chunk_chars=config.min_chunk_chars,
        ))
    write_chunks(chunks, config.chunks_path)
    return chunks, report.to_dict()


def build_all_chunks(meta_path: Path, text_dir: Path, out_path: Path,
                     config: RetrievalConfig | None = None) -> list[dict]:
    """Build original metadata/text layout; real abstracts are the text fallback."""
    cfg = config or RetrievalConfig(corpus_dir=Path(out_path).parent)
    chunks: list[dict] = []
    seen: set[str] = set()
    if not Path(meta_path).exists():
        return chunks
    with Path(meta_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            meta = json.loads(line)
            paper_id = str(meta.get("paper_id") or "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            text_path = Path(text_dir) / f"{paper_id}.txt"
            full_text = text_path.read_text(encoding="utf-8") if text_path.exists() else str(meta.get("text") or meta.get("abstract") or "")
            meta = {**meta, "source": str(meta.get("source") or meta.get("pdf_url") or text_path)}
            chunks.extend(build_chunks_for_paper(
                meta, full_text, chunk_size=cfg.chunk_size, overlap=cfg.overlap,
                strategy=cfg.chunk_strategy, min_chunk_chars=cfg.min_chunk_chars,
            ))
    write_chunks(chunks, out_path)
    return chunks
