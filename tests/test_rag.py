"""RAG orchestration tests: retrieval and LLM both mocked."""
import app.rag as rag
from app.llm import LLMError
from app.retriever import RetrievedChunk

CHUNKS = [RetrievedChunk("ColBERT uses late interaction.", "sample.pdf", 12, 0.4)]


def test_happy_path_builds_prompt_and_returns_answer(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(rag, "retrieve", lambda q, k=4, max_distance=None: CHUNKS)
    monkeypatch.setattr(rag, "ask", lambda p: prompts.append(p) or "  It uses late interaction [1].  ")

    result = rag.answer("What is ColBERT?")

    assert result.ok
    assert result.answer == "It uses late interaction [1]."
    assert result.chunks == CHUNKS
    assert "What is ColBERT?" in prompts[0]
    assert "[1] (sample.pdf, p.12)" in prompts[0]


def test_llm_error_returns_friendly_message(monkeypatch):
    def boom(prompt):
        raise LLMError("all models failed")

    monkeypatch.setattr(rag, "retrieve", lambda q, k=4, max_distance=None: CHUNKS)
    monkeypatch.setattr(rag, "ask", boom)

    result = rag.answer("What is RAG?")

    assert not result.ok
    assert result.answer == rag.LLM_ERROR_MESSAGE
    assert result.chunks == CHUNKS  # sources still shown


def test_no_chunks_skips_llm(monkeypatch):
    def must_not_call(prompt):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(rag, "retrieve", lambda q, k=4, max_distance=None: [])
    monkeypatch.setattr(rag, "ask", must_not_call)

    result = rag.answer("What is RAG?", max_distance=0.1)

    assert result.answer == rag.NO_ANSWER
    assert result.chunks == []


def test_blank_question_does_nothing(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: (_ for _ in ()).throw(AssertionError))
    result = rag.answer("   ")
    assert not result.ok
    assert result.answer == rag.EMPTY_QUESTION_MESSAGE
