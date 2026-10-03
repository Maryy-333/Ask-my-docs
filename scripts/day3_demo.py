"""Day 3: embed chunks and upsert into Chroma. Safe to run repeatedly.

Usage: python day3_demo.py path\\to\\file.pdf ["optional test question"]
"""
import sys
import time

from app.chunker import chunk_pages
from app.embedder import embed_query, embed_texts
from app.loader import load_pdf
from app.store import get_collection, query, upsert_chunks, TEXT_FIELD


def main(pdf_path: str, question: str | None) -> None:
    chunks = chunk_pages(load_pdf(pdf_path))
    print(f"{len(chunks)} chunks")

    t = time.perf_counter()
    vectors = embed_texts([getattr(c, TEXT_FIELD) for c in chunks], show_progress=True)
    print(f"embedded in {time.perf_counter() - t:.1f}s (dim={len(vectors[0])})")

    col = get_collection()
    upsert_chunks(col, chunks, vectors)
    print(f"collection count: {col.count()}  (should equal {len(chunks)} on every re-run)")

    if question:
        for hit in query(col, embed_query(question)):
            print(f"\n[{hit['distance']:.3f}] {hit['meta']}\n{hit['text'][:200]}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)