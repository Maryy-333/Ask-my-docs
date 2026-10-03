"""Local embeddings via sentence-transformers (no API, no quota)."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only; never executed at runtime
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~90 MB
BATCH_SIZE = 64


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the model once per process, on first use (not at import time)."""
    # Lazy import: sentence_transformers pulls in torch, ~23s on this machine.
    from sentence_transformers import SentenceTransformer

    # First run downloads weights to the HF cache.
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Embed documents. Vectors are L2-normalized, so cosine == dot product."""
    if not texts:
        return []
    vecs = _model().encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]