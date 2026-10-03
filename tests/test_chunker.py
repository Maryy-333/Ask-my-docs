from app.chunker import chunk_pages
from app.models import Page


def _pages():
    return [Page("a.pdf", 1, "Sentence one. " * 200), Page("a.pdf", 2, "Other text. " * 200)]


def test_chunks_respect_size_and_keep_page_metadata():
    chunks = chunk_pages(_pages(), chunk_size=500, chunk_overlap=50)
    assert chunks
    assert all(len(c.text) <= 500 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


def test_ids_are_unique_and_deterministic():
    a = chunk_pages(_pages())
    b = chunk_pages(_pages())
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)

def test_chunks_are_clean_sentences():
    chunks = chunk_pages(_pages(), chunk_size=300, chunk_overlap=50)
    assert all(c.text[0].isalnum() for c in chunks)
    assert all("(?<=" not in c.text for c in chunks)