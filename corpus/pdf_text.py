"""Extract normalized text from single- and double-column research PDFs."""

import unicodedata
from pathlib import Path


def _classify_block(block: tuple, mid_x: float, tolerance: float = 5.0) -> str:
    """Classify a text block as left-column, right-column, or full-width."""
    x0, x1 = block[0], block[2]
    if x1 <= mid_x + tolerance:
        return "left"
    if x0 >= mid_x - tolerance:
        return "right"
    return "full"


def _order_page_text(page) -> str:
    """Read each column top-to-bottom, separated by full-width headings."""
    mid_x = page.rect.width / 2
    text_blocks = sorted(
        (b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()),
        key=lambda b: b[1],
    )
    bands: list[list[tuple[str, tuple]]] = []
    current: list[tuple[str, tuple]] = []
    for block in text_blocks:
        column = _classify_block(block, mid_x)
        if column == "full":
            if current:
                bands.append(current)
                current = []
            bands.append([("full", block)])
        else:
            current.append((column, block))
    if current:
        bands.append(current)

    # Reading left then right avoids interleaving sentences from parallel columns.
    lines = []
    for band in bands:
        for wanted in ("full", "left", "right"):
            lines.extend(block[4] for column, block in band if column == wanted)
    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract page text and normalize typographic ligatures such as ﬁ to fi."""
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        text = "\n".join(_order_page_text(page) for page in document)
    return unicodedata.normalize("NFKC", text)
