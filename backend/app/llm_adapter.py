"""Flexible LLM / embedding provider switching.

Usage:
    from app.llm_adapter import get_llm, get_embed_model
    llm = get_llm()           # uses LLM_PROVIDER env var
    embed = get_embed_model()  # uses same provider for embeddings
"""

from __future__ import annotations

from functools import lru_cache

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import (
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    LLM_PROVIDER,
)


@lru_cache
def get_llm(provider: str | None = None) -> BaseLLM:
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "gemini":
        from llama_index.llms.gemini import Gemini

        return Gemini(
            model="models/gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
        )

    if provider == "openai":
        from llama_index.llms.openai import OpenAI  # type: ignore[import-untyped]

        return OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

    if provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic  # type: ignore[import-untyped]

        return Anthropic(model="claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY)

    raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache
def get_embed_model(provider: str | None = None) -> BaseEmbedding:
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "gemini":
        from llama_index.embeddings.gemini import GeminiEmbedding

        return GeminiEmbedding(
            model_name="models/text-embedding-004",
            api_key=GOOGLE_API_KEY,
        )

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore[import-untyped]

        return OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    raise ValueError(
        f"No embedding model configured for provider: {provider}. "
        "Gemini and OpenAI are supported for embeddings."
    )
