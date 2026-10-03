"""Eval scoring helpers: pure functions."""
from app.retriever import RetrievedChunk
from evals.run_eval import score_question, summarize

C = [RetrievedChunk("t", "s.pdf", 4, 0.4), RetrievedChunk("t", "s.pdf", 9, 0.6)]


def test_hit_when_expected_page_retrieved():
    r = score_question({"id": "x", "question": "q", "expected_pages": [9]}, C)
    assert r["hit"] and r["gold_distance"] == 0.6 and r["top_distance"] == 0.4


def test_miss_and_out_of_scope():
    miss = score_question({"id": "x", "question": "q", "expected_pages": [20]}, C)
    oos = score_question({"id": "y", "question": "q", "expected_pages": []}, C)
    assert not miss["hit"] and miss["gold_distance"] is None
    assert not oos["in_scope"]


def test_summary_suggests_midpoint_of_gap():
    results = [
        {"id": "a", "in_scope": True, "hit": True, "top_distance": 0.3, "gold_distance": 0.5},
        {"id": "b", "in_scope": False, "hit": False, "top_distance": 0.9, "gold_distance": None},
    ]
    s = summarize(results, 0.7)
    assert s["hit_rate"] == 1.0 and s["suggested"] == 0.7
    assert s["oos_rejected"] == 1 and s["gold_lost"] == 0