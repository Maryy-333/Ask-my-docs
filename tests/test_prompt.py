"""Prompt tests: pure function, no fixtures needed."""
from app.prompt import NO_ANSWER, build_prompt
from app.retriever import RetrievedChunk

CHUNKS = [
    RetrievedChunk("Dense retrieval uses embeddings.", "sample.pdf", 4, 0.41),
    RetrievedChunk("ColBERT uses late interaction.", "sample.pdf", 12, 0.52),
]


def test_contains_question():
    assert "What is ColBERT?" in build_prompt("What is ColBERT?", CHUNKS)


def test_numbered_blocks_with_source_and_page():
    p = build_prompt("q", CHUNKS)
    assert "[1] (sample.pdf, p.4)" in p
    assert "[2] (sample.pdf, p.12)" in p


def test_chunk_text_included_and_ordered():
    p = build_prompt("q", CHUNKS)
    assert p.index("Dense retrieval") < p.index("ColBERT uses")


def test_grounding_and_fallback_instructions():
    p = build_prompt("q", CHUNKS)
    assert "ONLY" in p
    assert NO_ANSWER in p


def test_empty_chunks_still_builds_prompt():
    p = build_prompt("q", [])
    assert "(no context available)" in p
    assert NO_ANSWER in p