"""Ingest a PDF via the same code path as the Streamlit UI. Safe to re-run.

Usage: python -m scripts.day3_demo path\\to\\file.pdf ["optional test question"]
"""
import sys
from pathlib import Path

from app.embedder import embed_query
from app.ingest import IngestLimits, ingest_pdf
from app.store import get_collection, list_sources, query

NO_LIMITS = IngestLimits(max_bytes=10**10, max_pages=10**6)  # the CLI is trusted input


def main(pdf_path: str, question: str | None) -> None:
    path = Path(pdf_path)
    col = get_collection()
    result = ingest_pdf(
        path.read_bytes(), path.name, col, limits=NO_LIMITS,
        on_progress=lambda frac, msg: print(f"[{frac:4.0%}] {msg}"),
    )
    print(result.message)
    if not result.ok:
        sys.exit(1)
    print(f"collection count: {col.count()}  sources: {list_sources(col)}")

    if question:
        for hit in query(col, embed_query(question)):
            print(f"\n[{hit['distance']:.3f}] {hit['meta']}\n{hit['text'][:200]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)