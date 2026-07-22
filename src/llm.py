"""LLM factory.

Isolated so the model provider is swappable in one place. Everything else in the
graph depends on this, not on `langchain_openai` directly.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import SETTINGS


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a configured chat model. Cached so we build it once."""
    return ChatOpenAI(
        model=SETTINGS.model,
        temperature=SETTINGS.temperature,
        api_key=SETTINGS.openai_api_key,
        timeout=30,
        max_retries=2,  # built-in retry on transient API errors
    )
