"""Retrieval-only eval (no LLM calls): python -m evals.run_eval [--k 4]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import DEFAULT_K, MAX_DISTANCE
from app.retriever import RetrievedChunk, retrieve
from app.store import get_collection

QUESTIONS_PATH = Path(__file__).with_name("questions.json")


def score_question(item: dict[str, Any], chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Score one question. Hit = any retrieved chunk is on an expected page."""
    expected = set(item["expected_pages"])
    gold = [c.distance for c in chunks if c.page in expected]
    return {
        "id": item["id"],
        "in_scope": bool(expected),
        "hit": bool(gold),
        "top_distance": chunks[0].distance if chunks else None,
        "gold_distance": min(gold) if gold else None,  # best chunk on an expected page
    }


def summarize(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Aggregate hit rate and threshold diagnostics."""
    inside = [r for r in results if r["in_scope"]]
    outside = [r for r in results if not r["in_scope"]]
    golds = [r["gold_distance"] for r in inside if r["gold_distance"] is not None]
    outs = [r["top_distance"] for r in outside if r["top_distance"] is not None]
    worst_gold = max(golds) if golds else None
    best_out = min(outs) if outs else None
    suggested = None
    if worst_gold is not None and best_out is not None and worst_gold < best_out:
        suggested = round((worst_gold + best_out) / 2, 2)
    return {
        "hit_rate": sum(r["hit"] for r in inside) / len(inside) if inside else 0.0,
        "n_in": len(inside),
        "worst_gold": worst_gold,
        "best_out": best_out,
        "oos_rejected": sum(1 for d in outs if d > threshold),
        "n_out": len(outside),
        "gold_lost": sum(1 for g in golds if g > threshold),
        "suggested": suggested,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()

    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    col = get_collection()
    results = [
        score_question(it, retrieve(it["question"], k=args.k, max_distance=None, collection=col))
        for it in items
    ]

    print(f"{'id':<15}{'scope':<6}{'hit':<5}{'top':>7}{'gold':>7}")
    for r in results:
        gold = "-" if r["gold_distance"] is None else f"{r['gold_distance']:.3f}"
        top = "-" if r["top_distance"] is None else f"{r['top_distance']:.3f}"
        print(f"{r['id']:<15}{'in' if r['in_scope'] else 'out':<6}"
              f"{'Y' if r['hit'] else ('-' if not r['in_scope'] else 'N'):<5}{top:>7}{gold:>7}")

    s = summarize(results, MAX_DISTANCE)
    print(f"\nHit@{args.k} (in-scope): {s['hit_rate']:.0%} of {s['n_in']}")
    print(f"Worst correct-chunk distance : {s['worst_gold']}")
    print(f"Best out-of-scope distance   : {s['best_out']}")
    print(f"Threshold {MAX_DISTANCE}: rejects {s['oos_rejected']}/{s['n_out']} out-of-scope, "
          f"would wrongly drop the best chunk for {s['gold_lost']} in-scope questions")
    print(f"Suggested threshold (midpoint of the gap): {s['suggested']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())