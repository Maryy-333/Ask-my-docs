"""PDF ingestion: validate -> load -> chunk -> embed -> replace-by-source upsert.

No Streamlit imports: the UI and the CLI share this one code path.
Bad input (corrupt, encrypted, scanned, too big) returns a result; it never raises.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import PureWindowsPath
from typing import Any

import pymupdf

from app.chunker import chunk_pages
from app.config import MAX_PAGES, MAX_UPLOAD_MB
from app.embedder import embed_texts
from app.loader import load_pdf_bytes
from app.store import delete_source, upsert_chunks

EmbedFn = Callable[[list[str]], list[list[float]]]
ProgressFn = Callable[[float, str], None]  # (fraction 0..1, message)

EMBED_BATCH = 64
MB = 1024 * 1024


class IngestStatus(str, Enum):
    OK = "ok"
    TOO_LARGE = "too_large"
    TOO_MANY_PAGES = "too_many_pages"
    ENCRYPTED = "encrypted"
    NO_TEXT = "no_text"
    INVALID = "invalid"


@dataclass(frozen=True)
class IngestLimits:
    max_bytes: int = MAX_UPLOAD_MB * MB
    max_pages: int = MAX_PAGES


@dataclass(frozen=True)
class IngestResult:
    status: IngestStatus
    message: str
    source: str
    pages: int = 0
    chunks_indexed: int = 0
    replaced: int = 0  # old chunks of the same source that were removed

    @property
    def ok(self) -> bool:
        return self.status is IngestStatus.OK


def _fail(status: IngestStatus, message: str, source: str, pages: int = 0) -> IngestResult:
    return IngestResult(status, message, source, pages=pages)


def _report(cb: ProgressFn | None, frac: float, msg: str) -> None:
    if cb is not None:
        cb(min(max(frac, 0.0), 1.0), msg)


def _inspect(data: bytes) -> tuple[bool, int]:
    """Return (is_encrypted, page_count). Raises if the bytes are not a PDF."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass and not doc.authenticate(""):
            return True, 0  # page_count is unavailable on locked documents
        return False, doc.page_count


def ingest_pdf(
    data: bytes,
    filename: str,
    collection: Any,
    embed_fn: EmbedFn = embed_texts,
    limits: IngestLimits = IngestLimits(),
    on_progress: ProgressFn | None = None,
) -> IngestResult:
    """Index a PDF, replacing any previously indexed chunks with the same source name.

    Order is embed first, then delete old chunks, then upsert, so a failure while
    embedding leaves the existing index untouched.
    """
    source = PureWindowsPath(filename).name or "uploaded.pdf"  # strip any directory part

    if len(data) > limits.max_bytes:
        return _fail(
            IngestStatus.TOO_LARGE,
            f"{source} is {len(data) / MB:.1f} MB; the limit is {limits.max_bytes // MB} MB.",
            source,
        )

    _report(on_progress, 0.05, "Reading PDF...")
    try:
        encrypted, n_pages = _inspect(data)
    except Exception:  # PyMuPDF raises several error types for corrupt input
        return _fail(
            IngestStatus.INVALID,
            f"{source} could not be opened as a PDF. It may be corrupt or not a PDF.",
            source,
        )
    if encrypted:
        return _fail(
            IngestStatus.ENCRYPTED,
            f"{source} is password-protected. Remove the password and upload it again.",
            source,
        )
    if n_pages > limits.max_pages:
        return _fail(
            IngestStatus.TOO_MANY_PAGES,
            f"{source} has {n_pages} pages; the limit is {limits.max_pages}.",
            source,
            pages=n_pages,
        )

    no_text = _fail(
        IngestStatus.NO_TEXT,
        f"No extractable text in {source}. It may be a scanned PDF "
        "(OCR is not supported).",
        source,
        pages=n_pages,
    )
    try:
        pages = load_pdf_bytes(data, source)
    except ValueError:
        return no_text

    _report(on_progress, 0.15, "Splitting into chunks...")
    chunks = chunk_pages(pages)
    if not chunks:
        return no_text

    texts = [c.text for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(embed_fn(texts[i : i + EMBED_BATCH]))
        done = min(i + EMBED_BATCH, len(texts))
        _report(on_progress, 0.15 + 0.75 * done / len(texts),
                f"Embedding {done}/{len(texts)} chunks...")

    _report(on_progress, 0.95, "Writing to index...")
    replaced = delete_source(collection, source)
    upsert_chunks(collection, chunks, vectors)
    _report(on_progress, 1.0, "Done")

    if replaced:
        msg = (f"Re-indexed {source}: replaced {replaced} old chunks "
               f"with {len(chunks)} new ones.")
    else:
        msg = f"Indexed {source}: {len(chunks)} chunks from {n_pages} pages."
    return IngestResult(IngestStatus.OK, msg, source, pages=n_pages,
                        chunks_indexed=len(chunks), replaced=replaced)