"""Retriever tests: fake embeddings and a fake store, no model, no network."""
import pytest

import app.retriever as retriever

HITS = [
    {"id": "a", "text": "alpha", "meta": {"source": "sample.pdf", "page": 4, "chunk_index": 0}, "distance": 0.41},
    {"id": "b", "text": "beta", "meta": {"source": "sample.pdf", "page": 5, "chunk_index": 1}, "distance": 0.52},
    {"id": "c", "text": "gamma", "meta": {"source": "sample.pdf", "page": 9, "chunk_index": 2}, "distance": 0.95},
]


@pytest.fixture
def calls(monkeypatch):
    seen: dict = {"embedded": []}

    def fake_embed(q):
        seen["embedded"].append(q)
        return [0.0] * 384

    def fake_query(col, emb, k=4):
        seen["k"] = k
        return HITS[:k]

    monkeypatch.setattr(retriever, "embed_query", fake_embed)
    monkeypatch.setattr(retriever, "query_collection", fake_query)
    monkeypatch.setattr(retriever, "get_collection", lambda *a, **kw: object())
    return seen


def test_maps_hits_to_dataclasses_in_order(calls):
    out = retriever.retrieve("q", k=3)
    assert [c.text for c in out] == ["alpha", "beta", "gamma"]
    assert out[0] == retriever.RetrievedChunk("alpha", "sample.pdf", 4, 0.41)


def test_passes_k_to_store(calls):
    out = retriever.retrieve("q", k=2)
    assert calls["k"] == 2
    assert len(out) == 2


def test_max_distance_none_keeps_everything(calls):
    assert len(retriever.retrieve("q", k=3)) == 3


def test_max_distance_filters_far_chunks(calls):
    out = retriever.retrieve("q", k=3, max_distance=0.6)
    assert [c.text for c in out] == ["alpha", "beta"]


def test_max_distance_is_inclusive(calls):
    out = retriever.retrieve("q", k=3, max_distance=0.52)
    assert [c.text for c in out] == ["alpha", "beta"]


def test_all_filtered_returns_empty(calls):
    assert retriever.retrieve("q", k=3, max_distance=0.1) == []


def test_blank_question_skips_embedding(calls):
    assert retriever.retrieve("   ") == []
    assert calls["embedded"] == []


def test_missing_metadata_gets_defaults(monkeypatch, calls):
    monkeypatch.setattr(
        retriever, "query_collection",
        lambda col, emb, k=4: [{"id": "x", "text": "t", "meta": None, "distance": 0.3}],
    )
    out = retriever.retrieve("q")
    assert (out[0].source, out[0].page) == ("unknown", 0)