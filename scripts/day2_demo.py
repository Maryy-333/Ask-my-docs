import sys
from statistics import mean

from app.chunker import chunk_pages
from app.loader import load_pdf

sys.stdout.reconfigure(encoding="utf-8")  # avoids Windows cp1252 print errors

path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
pages = load_pdf(path)
chunks = chunk_pages(pages)

lengths = [len(c.text) for c in chunks]
print(f"Pages with text: {len(pages)}")
print(f"Chunks: {len(chunks)}  (avg {mean(lengths):.0f} chars, min {min(lengths)}, max {max(lengths)})")
print(f"Unique IDs: {len({c.id for c in chunks}) == len(chunks)}")

for c in chunks[50:53]:
    print(f"\n--- {c.source} p.{c.page} #{c.chunk_index} [{c.id}] ---\n{c.text}")