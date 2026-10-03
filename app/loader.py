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


def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError(f"{path.name} is password-protected")
        for i, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append(Page(source=path.name, page=i, text=text))

    if not pages:
        raise ValueError(
            f"No extractable text in {path.name}. It may be a scanned PDF (needs OCR)."
        )
    return pages