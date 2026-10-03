"""Persistent Chroma collection. Upserts are idempotent thanks to deterministic chunk IDs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import chromadb

from app.embedder import MODEL_NAME

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "ask_my_docs"

# >>> Adjust these two to match your Chunk dataclass in app/models.py <<<
ID_FIELD = "id"
TEXT_FIELD = "text"


def get_collection(path: Path | str = CHROMA_DIR, model_name: str = MODEL_NAME):
    client = chromadb.PersistentClient(path=str(path))
    col = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    # Guard: mixing vectors from different models silently ruins retrieval.
    existing = (col.metadata or {}).get("embedding_model")
    if existing and existing != model_name:
        raise RuntimeError(
            f"Collection was built with {existing!r}, not {model_name!r}. "
            "Delete data/chroma and re-ingest."
        )
    return col


def _metadata(chunk) -> dict:
    """All scalar fields except id/text. Chroma rejects None and nested values."""
    d = asdict(chunk)
    d.pop(ID_FIELD, None)
    d.pop(TEXT_FIELD, None)
    return {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool))}


def upsert_chunks(col, chunks: list, embeddings: list[list[float]], batch: int = 128) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks vs {len(embeddings)} embeddings")
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.upsert(
            ids=[getattr(c, ID_FIELD) for c in part],
            documents=[getattr(c, TEXT_FIELD) for c in part],
            embeddings=embeddings[i : i + batch],
            metadatas=[_metadata(c) for c in part],
        )
    return len(chunks)


def query(col, query_embedding: list[float], k: int = 4) -> list[dict]:
    res = col.query(query_embeddings=[query_embedding], n_results=k)
    return [
        {"id": i, "text": d, "meta": m, "distance": dist}
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]