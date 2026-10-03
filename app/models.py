from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    source: str   # file name
    page: int     # 1-based, matches what a human sees in the PDF viewer
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int  # index within the page