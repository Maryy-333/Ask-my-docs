"""Gemini wrapper: retry on transient errors, fall back across models, raise on total failure."""
from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

log = logging.getLogger(__name__)
# Best-effort: hides the harmless "automatic function calling" notice.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
RETRYABLE = {429, 500, 502, 503, 504}

_client: genai.Client | None = None


class LLMError(RuntimeError):
    """Raised when every model and retry has failed."""


def _get_client() -> genai.Client:
    """Create the Gemini client lazily so importing this module needs no API key."""
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (check your .env file).")
        _client = genai.Client(api_key=key)
    return _client


def ask(prompt: str, retries: int = 3) -> str:
    """Return Gemini's text answer, or raise LLMError."""
    last_error: Exception | None = None
    for model in MODELS:
        for attempt in range(retries):
            try:
                resp = _get_client().models.generate_content(model=model, contents=prompt)
                if resp.text:
                    return resp.text
                last_error = LLMError(f"{model} returned empty text")
                break  # empty/blocked response: try next model
            except errors.APIError as e:
                last_error = e
                log.warning("[%s] attempt %d failed: %s %s", model, attempt + 1, e.code, str(e)[:80])
                if e.code not in RETRYABLE:
                    break  # 400/403/404: retrying the same model is pointless
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s; no sleep after the last try
    raise LLMError(f"All models failed. Last error: {last_error}")