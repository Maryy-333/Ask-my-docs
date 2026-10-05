"""Retrieval layer: question -> top-k chunks from Chroma, as plain dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.embedder import embed_query
from app.store import get_collection
from app.store import query as query_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by search. Lower distance = more similar (cosine)."""

    text: str
    source: str
    page: int
    distance: float


def retrieve(
    question: str,
    k: int = 4,
    max_distance: float | None = None,
    collection: Any = None,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """Return up to k chunks nearest to the question, best first.

    If max_distance is set, chunks with distance > max_distance are dropped.
    If `source` is set, only that document is searched (None = all documents).
    Pass `collection` to reuse an open Chroma collection (e.g. in Streamlit).
    """
    if not question.strip():
        return []

    col = collection if collection is not None else get_collection()
    extra = {"source": source} if source else {}  # unfiltered calls keep the old shape
    hits = query_collection(col, embed_query(question), k=k, **extra)

    results: list[RetrievedChunk] = []
    for hit in hits:
        distance = float(hit["distance"])
        if max_distance is not None and distance > max_distance:
            continue
        meta = hit.get("meta") or {}
        results.append(
            RetrievedChunk(
                text=hit["text"],
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", 0)),
                distance=distance,
            )
        )
    return results