from dataclasses import dataclass

import pytest

from app.store import ID_FIELD, TEXT_FIELD, get_collection, upsert_chunks


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