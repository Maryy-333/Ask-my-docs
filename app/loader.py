import re
from pathlib import Path

import pymupdf

from app.models import Page


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join hyphenated line breaks
    text = re.sub(r"\n{2,}", "\n\n", text)          # normalize paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _pages_from_doc(doc: "pymupdf.Document", name: str) -> list[Page]:
    """Extract cleaned text pages from an open document; skip empty pages."""
    if doc.needs_pass and not doc.authenticate(""):
        raise ValueError(f"{name} is password-protected")
    pages: list[Page] = []
    for i, page in enumerate(doc, start=1):
        text = _clean(page.get_text("text"))
        if text:
            pages.append(Page(source=name, page=i, text=text))
    if not pages:
        raise ValueError(
            f"No extractable text in {name}. It may be a scanned PDF (needs OCR)."
        )
    return pages


def load_pdf(path: str | Path) -> list[Page]:
    """Load a PDF from disk. Source name is the file name."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    with pymupdf.open(path) as doc:
        return _pages_from_doc(doc, path.name)


def load_pdf_bytes(data: bytes, name: str) -> list[Page]:
    """Load a PDF from memory (e.g. a Streamlit upload). `name` becomes the source."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return _pages_from_doc(doc, name)