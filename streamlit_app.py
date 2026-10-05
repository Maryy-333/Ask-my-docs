"""Ask My Docs: Streamlit UI. Run: streamlit run streamlit_app.py"""
from __future__ import annotations

import hashlib

import streamlit as st

from app.config import DEFAULT_K, MAX_DISTANCE, MAX_PAGES, MAX_UPLOAD_MB
from app.ingest import IngestResult, ingest_pdf
from app.rag import EMPTY_QUESTION_MESSAGE, TOO_SHORT_MESSAGE, RagResult, answer
from app.store import get_collection, list_sources

st.set_page_config(page_title="Ask My Docs", page_icon="📄")
ALL_DOCS = "All documents"


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def warm_up() -> bool:
    """Load the embedding model once per server process, not on the first question."""
    from app.embedder import embed_query

    embed_query("warm up")
    return True


@st.cache_resource
def open_collection():
    """Open the persistent Chroma collection once per server process."""
    return get_collection()


def render_result(result: RagResult) -> None:
    """Show an answer with the right message style and its source chunks."""
    if not result.ok:
        hint = result.answer in (TOO_SHORT_MESSAGE, EMPTY_QUESTION_MESSAGE)
        (st.info if hint else st.warning)(result.answer)
    else:
        st.markdown(result.answer)  # renders the LaTeX in answers

    if result.cited:
        label = "Sources cited"
        shown = [(n, result.chunks[n - 1]) for n in result.cited]
    else:
        label = "Sources retrieved"
        shown = list(enumerate(result.chunks, start=1))
    if shown:
        with st.expander(f"{label} ({len(shown)})"):
            for n, c in shown:
                st.markdown(f"**[{n}] {c.source}, p.{c.page}** · distance {c.distance:.3f}")
                st.caption(" ".join(c.text.split())[:400])


def show_ingest_result(result: IngestResult) -> None:
    (st.success if result.ok else st.warning)(result.message)


def handle_upload(uploaded, col) -> None:
    """Index an upload once per distinct (name, content); reruns reuse the stored result."""
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.digests.get(uploaded.name) != digest:
        bar = st.progress(0.0, text="Starting...")
        try:
            result = ingest_pdf(
                data, uploaded.name, col,
                on_progress=lambda frac, msg: bar.progress(frac, text=msg),
            )
        except Exception as exc:  # not bad input: model/DB failure. Don't record the digest.
            bar.empty()
            st.error(f"Indexing failed unexpectedly: {exc}")
            return
        bar.empty()
        st.session_state.digests[uploaded.name] = digest
        st.session_state.results[uploaded.name] = result
    show_ingest_result(st.session_state.results[uploaded.name])


# ---- state + warm-up -------------------------------------------------------
st.session_state.setdefault("messages", [])  # each: {"role", "content", "result"}
st.session_state.setdefault("digests", {})   # file name -> sha256 of last processed bytes
st.session_state.setdefault("results", {})   # file name -> IngestResult
warm_up()
collection = open_collection()

# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"],
        help=f"Max {MAX_UPLOAD_MB} MB and {MAX_PAGES} pages. Scanned PDFs (no text) are not supported.",
    )
    st.caption(f"Limits: {MAX_UPLOAD_MB} MB, {MAX_PAGES} pages. Text PDFs only (no OCR). "
               "Re-uploading a file with the same name replaces it.")
    if uploaded is not None:
        handle_upload(uploaded, collection)

    sources = list_sources(collection)
    choice = st.selectbox("Search in", [ALL_DOCS, *sources], disabled=not sources)
    source = None if choice == ALL_DOCS else choice
    for name, n in sources.items():
        st.caption(f"{name}: {n} chunks")

    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", 2, 8, DEFAULT_K)
    use_cutoff = st.checkbox(f"Apply distance cutoff ({MAX_DISTANCE})", value=True)
    st.caption("Lower distance = more similar. Chunks above the cutoff are dropped, "
               "and if none remain the LLM is not called.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- main ------------------------------------------------------------------
st.title("📄 Ask My Docs")
st.caption("Answers come only from the indexed documents, with sources.")
if not sources:
    st.info("No documents indexed yet. Upload a PDF in the sidebar to begin.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"])

if prompt := st.chat_input("Ask a question about your documents", disabled=not sources):
    st.session_state.messages.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = answer(
                prompt, k=k, max_distance=MAX_DISTANCE if use_cutoff else None, source=source
            )
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})