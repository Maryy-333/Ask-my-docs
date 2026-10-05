# Ask My Docs

A retrieval-augmented generation (RAG) app that answers questions about your PDFs, **cites the passages it used**, and says "I don't know" when the documents don't contain the answer. Upload one or more PDFs in the UI and search all of them or just one. Built entirely on free tools: local embeddings, a local vector store, and the Gemini free tier.

![UI screenshot](docs/ui.png)

## How it works

```
PDF upload -> validate -> pages -> chunks -> MiniLM embeddings (local) -> Chroma (persistent)
                                                                              |
question -> embed -> top-k chunks (all docs or one) -> distance cutoff -> grounded prompt -> Gemini -> answer with [n] citations
```

- **Ingestion:** one pure function (`app/ingest.py`) shared by the UI and the CLI. It checks size, page count, encryption and extractable text, then chunks, embeds, and replaces any existing chunks of the same source. Bad input returns a status and a message; it never raises.
- **Chunking:** deterministic chunk IDs (source, page, index, text), so re-ingesting the same file is idempotent.
- **Replace-by-source:** before upserting a document, its old chunks are deleted by metadata (`source`), so re-ingesting an edited file leaves no stale chunks. Order is embed, then delete, then upsert, so a failure during embedding leaves the existing index untouched.
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers, run locally (384-dim, normalized, cosine space). No API cost or quota.
- **Retrieval:** Chroma returns the k nearest chunks, optionally filtered to one source. Results are converted to plain dataclasses so the rest of the app never depends on Chroma's response shape.
- **Prompt:** numbered context blocks like `[1] (file.pdf, p.4)`. The model must answer only from the context, cite blocks by number, and reply with a fixed fallback sentence if the context is insufficient. The context is explicitly treated as data, not instructions (a basic prompt-injection mitigation).
- **Citations:** `[n]` markers in the answer are parsed and mapped back to chunks, so the UI shows only the sources the model actually used. Out-of-range numbers are ignored.
- **Failures are values:** if every Gemini model fails (rate limit, 5xx), `answer()` returns a friendly message with `ok=False` instead of raising. The LLM client retries transient errors with backoff and falls back across models.

## Evaluation

The retrieval layer is evaluated without calling the LLM, so it is free, fast and deterministic: `python -m evals.run_eval`.

Test set: **22 questions** over one PDF (a draft chapter of Jurafsky & Martin's *Speech and Language Processing*, "Information Retrieval and RAG"): 14 answerable (including 2 paraphrased) and 8 unanswerable (3 clearly off-topic, 5 adjacent or vague). Each answerable question has a hand-checked expected page. A hit means any retrieved chunk comes from an expected page.

| Metric | k=4 | k=6 | k=8 |
|---|---|---|---|
| Hit rate (14 answerable) | 93% | 100% | 100% |
| Worst distance of a correct chunk | 0.591 | 0.679 | 0.679 |

The single k=4 miss was a list-style answer ("steps of the basic RAG algorithm") that ranked 6th, behind chunks that discuss RAG in general. The default is **k=6**, the smallest value that hit everything.

These numbers were measured on a single-document index. After adding multi-document support, the unfiltered eval was re-run on that same index and reproduced them exactly (Hit@6 = 100%, worst correct distance 0.679, best out-of-scope 0.562), which confirms the new source filter does not change default retrieval.

**Finding: no distance threshold cleanly separates unanswerable questions.**

| Query type | Distance of top result |
|---|---|
| Clearly off-topic (capital of Australia, bread recipe, World Cup) | 0.78 to 0.87 |
| Adjacent or vague (attention, word2vec, Elasticsearch, "test", "hello") | 0.56 to 0.76 |
| Worst correct chunk | 0.59 to 0.68 |

The best unanswerable query (0.562) is closer than some correct chunks, so any cutoff that rejects it would also drop real answers. Instead of over-tuning one number, the app uses three layers:

1. **Length guard:** inputs under 8 characters are rejected before retrieval and before any LLM call.
2. **Distance cutoff (0.70):** on this set it rejects 4 of 8 unanswerable questions and drops no correct chunk. If nothing survives, the LLM is not called.
3. **Prompt rule:** the model must return an exact fallback sentence when the context doesn't answer the question. This catches the adjacent-topic questions that pass the cutoff (at the cost of one wasted LLM call).

### Re-checking the cutoff on a multi-document index

The cutoff was tuned on one document and **has not yet been validated on a multi-document index**. A correct chunk's distance depends on the model and chunking, not on corpus size, so correct-chunk distances should not move. But the nearest neighbour of an unanswerable question is a minimum over more candidates, so with more documents the best out-of-scope distance will tend to fall and the cutoff will reject fewer of them. Treat the cutoff as a weak first filter; the prompt rule is the real guard.

To re-check: back up `data/chroma`, ingest 2 to 3 other PDFs (one close in topic to the sample), run `python -m evals.run_eval --k 6`, and compare Hit@6 (crowding by other documents) and the number of out-of-scope questions still rejected (currently 4 of 8).

## Known limitations

- **Small eval:** 22 questions, written by the developer, one document. It shows where each layer breaks; it is not a validated benchmark.
- **Thin threshold margin:** the worst correct chunk (0.679) sits only 0.021 below the 0.70 cutoff. The cutoff would need re-checking after any change to chunking, k, the embedding model, or the number of indexed documents (see above).
- **Upload limits:** 10 MB and 100 pages per PDF. These protect local CPU time and memory, not API quota (embeddings are local). The 10 MB value is set in both `app/config.py` and `.streamlit/config.toml`; keep them in sync.
- **Text PDFs only.** Scanned PDFs are rejected with a message (no OCR). Password-protected PDFs are rejected.
- **A document's identity is its file name.** Re-uploading a file with the same name replaces its chunks; two different files with the same name overwrite each other.
- **Replacement is not atomic.** It runs embed, delete, upsert. A failure while embedding leaves the old index intact, but a crash between delete and upsert would leave that document missing until it is re-uploaded.
- **No delete button in the UI.** Documents can be replaced but not removed from the sidebar; remove one with `delete_source` from `app/store.py`, or delete `data/chroma/`.
- **PDF extraction noise:** PyMuPDF sometimes interleaves margin labels into the text, which can show up in chunks and snippets.
- **Dense retrieval weakness:** on list-style content, general discussion can outrank the passage with the actual list (the `rag_steps` miss at k=4). A reranker would be the next step.
- **Free-tier LLM:** the Gemini free tier can return 503/429 under load; the retry and fallback logic handles it, but responses can be slow.

## Setup

Requires Python 3.13. Commands are for Windows PowerShell.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# edit .env and set GEMINI_API_KEY (free key from Google AI Studio)
```

### Index a PDF

Easiest: run the app and upload PDFs in the sidebar (see below). From the command line, using the same ingestion code path (no size or page limits, since the CLI is trusted input):

```powershell
python -m scripts.day3_demo data\sample.pdf "What is BM25?"
```

This validates and loads the PDF, chunks it, embeds the chunks and stores them in `data/chroma/`, replacing any earlier chunks of the same file name (re-running is safe), then runs the question as a quick retrieval check.

The first run downloads the embedding model (~90 MB) to the Hugging Face cache.

### Run

```powershell
streamlit run streamlit_app.py                 # web UI: upload PDFs, choose "All documents" or one
python -m scripts.day4_demo "What is BM25?"    # CLI
python -m evals.run_eval --k 6                 # retrieval eval (needs the matching PDF indexed)
python -m pytest tests -q                      # 56 tests, no network or model download needed
```

The sidebar in the UI lets you upload PDFs, pick which document to search, change k, and switch the distance cutoff off, which makes the cutoff's effect visible: with it off, an off-topic question shows six retrieved chunks at distance 0.87 to 0.92.

## Project layout

```
app/
  loader.py      PDF (path or bytes) -> pages
  chunker.py     pages -> chunks with deterministic IDs
  embedder.py    local embeddings (lazy-loaded model)
  store.py       Chroma wrapper: upsert, query (optional source filter), delete_source, list_sources
  ingest.py      validate -> load -> chunk -> embed -> replace-by-source upsert (pure, no UI)
  retriever.py   question -> RetrievedChunk list
  prompt.py      grounded prompt builder
  citations.py   [n] marker parser
  llm.py         Gemini client with retry and model fallback
  rag.py         orchestration -> RagResult
  config.py      k, distance cutoff, minimum query length, upload limits
scripts/         CLI demos (day3_demo: ingest, day4_demo: ask); run with python -m scripts.<name>
evals/           retrieval eval (questions.json, run_eval.py)
tests/           unit tests; embeddings, store and LLM are faked
.streamlit/      config.toml (browser-side upload size limit)
streamlit_app.py
```

## Design decisions

- **Free stack end to end.** Local embeddings and a local vector store mean the only metered dependency is the Gemini free tier.
- **Lazy imports.** `sentence_transformers` takes about 23 seconds to import on the dev machine, so it is imported inside the cached model loader. Unit tests and module imports don't pay for it, and the UI warms the model once with `st.cache_resource`.
- **Retrieval-only eval.** Retrieval quality is measured separately from generation, so changing chunk size, k, or the embedding model shows up in numbers without spending LLM quota.
- **Pure, small functions.** `build_prompt`, `extract_citations` and `ingest_pdf` have no UI dependencies, and the UI contains no RAG logic, so each piece is unit-testable without mocks of a framework. `ingest_pdf` takes the embedder as a parameter, so its tests run offline.
- **Plain dataclasses at boundaries.** `RetrievedChunk`, `RagResult` and `IngestResult` keep the UI and tests independent of Chroma and Gemini response formats.
- **Bad input is a value, not an exception.** Corrupt, encrypted, scanned, oversized or too-long PDFs return a status and a friendly message, so the UI never shows a traceback for user error.
- **Optional source filter, default off.** `source=None` keeps the exact old call shape through `rag.answer`, `retriever.retrieve` and `store.query`, so adding multi-document search could not change single-document behaviour (verified by an unchanged eval).
- **Rerun-safe uploads.** Streamlit reruns the script on every interaction, so the UI remembers the SHA-256 of the last processed bytes per file name and skips re-indexing unless the content changed. Keying on name plus hash (not hash alone) means going v1 to v2 and back to v1 still re-indexes correctly.