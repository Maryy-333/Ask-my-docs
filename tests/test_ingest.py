import pymupdf
import pytest

from app.ingest import IngestLimits, IngestStatus, ingest_pdf
from app.store import get_collection, list_sources


def _vec(i: int) -> list[float]:
    v = [0.0] * 384
    v[i % 384] = 1.0
    return v


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [_vec(len(t)) for t in texts]


def boom(_texts):  # proves rejected files never reach the embedder
    raise AssertionError("embedder must not be called")


def make_pdf(pages: list[str], **save_kw) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    return doc.tobytes(**save_kw)


ALPHA = ["alpha " * 60, "alpha bravo " * 50]
BETA = ["beta gamma " * 60]


@pytest.fixture
def col(tmp_path):
    return get_collection(tmp_path)


def test_ingest_ok(col):
    r = ingest_pdf(make_pdf(ALPHA), "a.pdf", col, embed_fn=fake_embed)
    assert r.ok and r.replaced == 0 and r.pages == 2
    assert r.chunks_indexed == col.count() > 0
    assert list_sources(col) == {"a.pdf": r.chunks_indexed}


def test_reingest_same_file_keeps_count(col):
    data = make_pdf(ALPHA)
    first = ingest_pdf(data, "a.pdf", col, embed_fn=fake_embed)
    second = ingest_pdf(data, "a.pdf", col, embed_fn=fake_embed)
    assert second.replaced == first.chunks_indexed
    assert col.count() == first.chunks_indexed


def test_reingest_edited_file_leaves_no_stale_chunks(col):
    first = ingest_pdf(make_pdf(ALPHA), "a.pdf", col, embed_fn=fake_embed)
    second = ingest_pdf(make_pdf(BETA), "a.pdf", col, embed_fn=fake_embed)
    assert second.replaced == first.chunks_indexed
    assert col.count() == second.chunks_indexed
    docs = col.get()["documents"]
    assert docs and all("beta" in d for d in docs) and not any("alpha" in d for d in docs)


def test_other_sources_untouched(col):
    ingest_pdf(make_pdf(BETA), "b.pdf", col, embed_fn=fake_embed)
    b_count = list_sources(col)["b.pdf"]
    ingest_pdf(make_pdf(ALPHA), "a.pdf", col, embed_fn=fake_embed)
    ingest_pdf(make_pdf(BETA), "a.pdf", col, embed_fn=fake_embed)  # edit a.pdf
    assert list_sources(col)["b.pdf"] == b_count


def test_failed_reingest_keeps_existing_index(col):
    ingest_pdf(make_pdf(ALPHA), "a.pdf", col, embed_fn=fake_embed)
    before = col.count()
    r = ingest_pdf(b"not a pdf", "a.pdf", col, embed_fn=boom)
    assert r.status is IngestStatus.INVALID
    assert col.count() == before


@pytest.mark.parametrize(
    "data, limits, status",
    [
        (make_pdf(ALPHA), IngestLimits(max_bytes=10, max_pages=100), IngestStatus.TOO_LARGE),
        (make_pdf(["x " * 40] * 3), IngestLimits(max_pages=2), IngestStatus.TOO_MANY_PAGES),
        (make_pdf(["secret " * 40], encryption=pymupdf.PDF_ENCRYPT_AES_256,
                  user_pw="user", owner_pw="owner"), IngestLimits(), IngestStatus.ENCRYPTED),
        (make_pdf([""]), IngestLimits(), IngestStatus.NO_TEXT),  # blank page = scanned stand-in
        (b"not a pdf", IngestLimits(), IngestStatus.INVALID),
    ],
)
def test_bad_input_returns_status_not_exception(col, data, limits, status):
    r = ingest_pdf(data, "bad.pdf", col, embed_fn=boom, limits=limits)
    assert r.status is status and not r.ok and r.message
    assert col.count() == 0


def test_directory_part_of_filename_is_stripped(col):
    r = ingest_pdf(make_pdf(ALPHA), "C:\\tmp\\doc.pdf", col, embed_fn=fake_embed)
    assert r.source == "doc.pdf" and list_sources(col) == {"doc.pdf": r.chunks_indexed}


def test_progress_is_monotonic_and_finishes(col):
    seen: list[float] = []
    ingest_pdf(make_pdf(ALPHA), "a.pdf", col, embed_fn=fake_embed,
               on_progress=lambda f, _m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0