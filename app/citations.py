"""Parse [n] citation markers from an LLM answer."""
from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(text: str, n_chunks: int) -> list[int]:
    """Return sorted unique 1-based chunk numbers cited in `text`.

    Handles [1], [2][3] and [1, 3]. Numbers outside 1..n_chunks are ignored,
    so a hallucinated [9] never breaks the lookup.
    """
    found: set[int] = set()
    for match in _CITE.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if 1 <= number <= n_chunks:
                found.add(number)
    return sorted(found)