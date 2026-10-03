"""Ask My Docs: Streamlit UI. Run: streamlit run streamlit_app.py"""
from __future__ import annotations

import streamlit as st

from app.config import DEFAULT_K, MAX_DISTANCE
from app.rag import EMPTY_QUESTION_MESSAGE, TOO_SHORT_MESSAGE, RagResult, answer

st.set_page_config(page_title="Ask My Docs", page_icon="📄")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def warm_up() -> bool:
    """Load the embedding model once per server process, not on the first question."""
    from app.embedder import embed_query

    embed_query("warm up")
    return True


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


# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", 2, 8, DEFAULT_K)
    use_cutoff = st.checkbox(f"Apply distance cutoff ({MAX_DISTANCE})", value=True)
    st.caption("Lower distance = more similar. Chunks above the cutoff are dropped, "
               "and if none remain the LLM is not called.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- main ----------------------------------------------------------------
st.title("📄 Ask My Docs")
st.caption("Answers come only from the indexed document, with sources.")
warm_up()

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "result"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["result"])

if prompt := st.chat_input("Ask a question about the document"):
    st.session_state.messages.append({"role": "user", "content": prompt, "result": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = answer(prompt, k=k, max_distance=MAX_DISTANCE if use_cutoff else None)
        render_result(result)
    st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})