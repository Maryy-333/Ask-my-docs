"""RAG orchestration: question -> retrieve -> prompt -> LLM -> RagResult."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.citations import extract_citations
from app.config import DEFAULT_K, MAX_DISTANCE, MIN_QUERY_CHARS
from app.llm import LLMError, ask
from app.prompt import NO_ANSWER, build_prompt
from app.retriever import RetrievedChunk, retrieve

LLM_ERROR_MESSAGE = (
    "Sorry, the language model is unavailable right now "
    "(rate limit or service error). Please try again in a minute."
)
EMPTY_QUESTION_MESSAGE = "Please enter a question."
TOO_SHORT_MESSAGE = "Please ask a full question (a few words at least)."


@dataclass(frozen=True)
class RagResult:
    """Answer text, chunks shown to the LLM, success flag, and cited chunk numbers."""

    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    ok: bool = True
    cited: list[int] = field(default_factory=list)  # 1-based, matches [n] in the prompt


def answer(
    question: str,
    k: int = DEFAULT_K,
    max_distance: float | None = MAX_DISTANCE,
) -> RagResult:
    """Answer from the indexed documents. Pass max_distance=None to disable the cutoff.

    Never raises on LLM failure.
    """
    question = question.strip()
    if not question:
        return RagResult(EMPTY_QUESTION_MESSAGE, [], ok=False)
    if len(question) < MIN_QUERY_CHARS:
        # Vague inputs ("hello", "test") would burn quota on a guaranteed "I don't know".
        return RagResult(TOO_SHORT_MESSAGE, [], ok=False)

    chunks = retrieve(question, k=k, max_distance=max_distance)
    if not chunks:
        # Nothing close enough: skip the LLM call (saves free-tier quota).
        return RagResult(NO_ANSWER, [])

    try:
        text = ask(build_prompt(question, chunks)).strip()
    except LLMError:
        return RagResult(LLM_ERROR_MESSAGE, chunks, ok=False)

    cited = [] if NO_ANSWER in text else extract_citations(text, len(chunks))
    return RagResult(text, chunks, cited=cited)