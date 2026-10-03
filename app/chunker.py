import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import Chunk, Page


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(splitter.split_text(page.text)):
            if len(text.strip()) < 30:
                continue
            key = f"{page.source}|{page.page}|{i}|{text}"
            chunk_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(id=chunk_id, text=text, source=page.source,
                      page=page.page, chunk_index=i)
            )
    return chunks