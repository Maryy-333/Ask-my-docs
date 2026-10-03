"""Demo: python day4_demo.py "question" [-k 4] [--max-distance 0.7 | --no-threshold]"""
from __future__ import annotations

import argparse
import sys

from app.config import DEFAULT_K, MAX_DISTANCE
from app.rag import answer


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # PowerShell defaults to cp1252

    parser = argparse.ArgumentParser(description="Ask a question about your indexed docs.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="chunks to retrieve")
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                        help=f"drop chunks above this cosine distance (default {MAX_DISTANCE})")
    parser.add_argument("--no-threshold", action="store_true", help="disable the distance cutoff")
    args = parser.parse_args()

    max_distance = None if args.no_threshold else args.max_distance
    result = answer(args.question, k=args.k, max_distance=max_distance)

    print(f"\nQuestion: {args.question}\n")
    print("Answer:")
    print(result.answer)

    if result.cited:
        print("\nSources cited:")
        shown = [(n, result.chunks[n - 1]) for n in result.cited]
    else:
        print("\nSources retrieved (answer has no citations):" if result.chunks else "\nSources:")
        shown = list(enumerate(result.chunks, start=1))
    if not shown:
        print("  (none)")
    for n, c in shown:
        snippet = " ".join(c.text.split())[:100]
        print(f"  [{n}] {c.source} p.{c.page}  distance={c.distance:.3f}")
        print(f"      {snippet}...")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())