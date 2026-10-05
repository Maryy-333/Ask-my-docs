"""Central tunables. Change here, not in call sites."""

DEFAULT_K = 6
# Cosine distance cutoff (lower = more similar). Starting value from 3 queries;
# validate with `python -m evals.run_eval`.
MAX_DISTANCE = 0.70

MIN_QUERY_CHARS = 8

# Upload limits. MiniLM runs on CPU: ~4.8 chunks/page in sample.pdf, so 100 pages ~ 480 chunks.
# These protect local CPU time and memory, not Gemini quota (embeddings are local).
MAX_UPLOAD_MB = 10
MAX_PAGES = 100