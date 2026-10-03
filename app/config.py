"""Central tunables. Change here, not in call sites."""

DEFAULT_K = 6
# Cosine distance cutoff (lower = more similar). Starting value from 3 queries;
# validate with `python -m evals.run_eval`.
MAX_DISTANCE = 0.70

MIN_QUERY_CHARS = 8