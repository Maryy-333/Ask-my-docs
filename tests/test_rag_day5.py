"""Day 5 behavior: default threshold, cited indices, NO_ANSWER handling."""
import app.rag as rag
from app.config import MAX_DISTANCE
from app.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk("a", "sample.pdf", 1, 0.3),
    RetrievedChunk("b", "sample.pdf", 2, 0.4),
]


def _patch(monkeypatch, reply):
    seen = {}

    def fake_retrieve(q, k=4, max_distance=None):
        seen["max_distance"] = max_distance
        return CHUNKS

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "ask", lambda p: reply)
    return seen


def test_threshold_on_by_default(monkeypatch):
    seen = _patch(monkeypatch, "x [1]")
    rag.answer("What is RAG?")
    assert seen["max_distance"] == MAX_DISTANCE


def test_threshold_can_be_disabled(monkeypatch):
    seen = _patch(monkeypatch, "x [1]")
    rag.answer("What is RAG?", max_distance=None)
    assert seen["max_distance"] is None


def test_cited_indices_parsed(monkeypatch):
    _patch(monkeypatch, "Fact [2]. Another [1][9].")
    assert rag.answer("What is RAG?").cited == [1, 2]


def test_no_answer_has_no_citations(monkeypatch):
    _patch(monkeypatch, rag.NO_ANSWER + " [1]")
    assert rag.answer("What is RAG?").cited == []


def test_answer_without_citations_has_empty_cited(monkeypatch):
    _patch(monkeypatch, "plain")
    assert rag.answer("What is RAG?").cited == []


def test_too_short_skips_retrieval_and_llm(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("should not be called")

    monkeypatch.setattr(rag, "retrieve", boom)
    monkeypatch.setattr(rag, "ask", boom)
    result = rag.answer("hello")
    assert not result.ok and result.answer == rag.TOO_SHORT_MESSAGE
