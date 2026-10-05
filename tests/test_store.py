from dataclasses import dataclass

import pytest

from app.store import ID_FIELD, TEXT_FIELD, delete_source, get_collection, list_sources, query, upsert_chunks

@dataclass
class FakeChunk:
    id: str
    text: str
    source: str = "doc.pdf"
    page: int = 1
    chunk_index: int = 0

def _vec(i: int) -> list[float]:
    v = [0.0] * 384
    v[i % 384] = 1.0
    return v


def test_field_names_match_fake():
    assert ID_FIELD == "id" and TEXT_FIELD == "text"


def test_upsert_is_idempotent(tmp_path):
    col = get_collection(tmp_path)
    chunks = [FakeChunk(f"id{i}", f"text {i}") for i in range(5)]
    vecs = [_vec(i) for i in range(5)]
    upsert_chunks(col, chunks, vecs)
    upsert_chunks(col, chunks, vecs)
    assert col.count() == 5


def test_metadata_excludes_id_and_text(tmp_path):
    col = get_collection(tmp_path)
    upsert_chunks(col, [FakeChunk("a", "hello")], [_vec(0)])
    got = col.get(ids=["a"])
    assert got["metadatas"][0] == {"source": "doc.pdf", "page": 1, "chunk_index": 0}
    assert got["documents"][0] == "hello"


def test_length_mismatch_raises(tmp_path):
    col = get_collection(tmp_path)
    with pytest.raises(ValueError):
        upsert_chunks(col, [FakeChunk("a", "x")], [])


def test_model_mismatch_raises(tmp_path):
    get_collection(tmp_path, model_name="model-a")
    with pytest.raises(RuntimeError):
        get_collection(tmp_path, model_name="model-b")


def _two_docs(col):
    chunks = [FakeChunk(f"a{i}", f"a text {i}", source="a.pdf") for i in range(2)]
    chunks += [FakeChunk(f"b{i}", f"b text {i}", source="b.pdf") for i in range(2)]
    upsert_chunks(col, chunks, [_vec(0), _vec(1), _vec(2), _vec(3)])


def test_delete_source_removes_only_that_source(tmp_path):
    col = get_collection(tmp_path)
    _two_docs(col)
    assert delete_source(col, "a.pdf") == 2
    assert list_sources(col) == {"b.pdf": 2}
    assert delete_source(col, "a.pdf") == 0  # idempotent


def test_list_sources_counts(tmp_path):
    col = get_collection(tmp_path)
    _two_docs(col)
    assert list_sources(col) == {"a.pdf": 2, "b.pdf": 2}


def test_query_source_filter_excludes_other_documents(tmp_path):
    col = get_collection(tmp_path)
    _two_docs(col)
    # _vec(0) is a0's exact vector, so unfiltered the nearest hit is from a.pdf
    assert query(col, _vec(0), k=1)[0]["meta"]["source"] == "a.pdf"
    hits = query(col, _vec(0), k=2, source="b.pdf")
    assert {h["meta"]["source"] for h in hits} == {"b.pdf"} and len(hits) == 2


def test_query_k_larger_than_matches_does_not_raise(tmp_path):
    col = get_collection(tmp_path)
    _two_docs(col)
    assert len(query(col, _vec(0), k=6, source="a.pdf")) == 2
    assert query(col, _vec(0), k=6, source="missing.pdf") == []