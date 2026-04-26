"""
LLM / Embedding middleware — single point of model selection.

Swap models by changing `config.model_ids[role]` or
`config.model_providers[role]`. No other code needs to change.

Usage:
    registry = LLMRegistry(config)
    llm = registry.get_llm("extraction")       # DeepSeek V4 Flash
    embed = registry.get_embed_model()          # Qwen3-Embedding-8B
    llm2 = registry.get_llm("retrieval_detailed")  # DeepSeek V4 Pro
"""

import logging
from typing import Any, Dict, Optional

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike

from config import PipelineConfig

logger = logging.getLogger(__name__)


class LLMRegistry:
    """
    Role-based model registry.

    Each *role* (e.g. "extraction", "retrieval_simple") maps to
    exactly one model + provider combination. Changing the mapping
    in config swaps the model everywhere it's used.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._llm_cache: Dict[str, LLM] = {}
        self._embed_model: Optional[BaseEmbedding] = None

    # ── Public API ──────────────────────────────────────────────

    def get_llm(self, role: str) -> LLM:
        """Get or create an LLM instance for the given role."""
        if role in self._llm_cache:
            return self._llm_cache[role]

        model_id = self.config.model_ids[role]
        provider = self.config.model_providers[role]
        provider_cfg = self._get_provider_config(provider)

        llm = OpenAILike(
            model=model_id,
            api_base=provider_cfg.base_url,
            api_key=provider_cfg.api_key,
            is_chat_model=True,
            context_window=131072,
            max_tokens=8192,
            temperature=0.0 if "extraction" in role or "community" in role else 0.1,
        )

        self._llm_cache[role] = llm
        logger.info(f"Registered LLM role={role!r} → {provider}/{model_id}")
        return llm

    def get_embed_model(self) -> BaseEmbedding:
        """Get or create the embedding model."""
        if self._embed_model is not None:
            return self._embed_model

        from embeddings import OpenAICompatibleEmbedding

        role = "embedding"
        model_id = self.config.model_ids[role]
        provider = self.config.model_providers[role]
        provider_cfg = self._get_provider_config(provider)

        self._embed_model = OpenAICompatibleEmbedding(
            model=model_id,
            api_key=provider_cfg.api_key,
            api_base=provider_cfg.base_url,
            dimensions=self.config.embedding_dimensions,
        )
        logger.info(
            f"Registered Embedding → {provider}/{model_id} "
            f"(dim={self.config.embedding_dimensions})"
        )
        return self._embed_model

    def clear_cache(self):
        """Force re-creation of all models (use after config change)."""
        self._llm_cache.clear()
        self._embed_model = None

    # ── Internals ───────────────────────────────────────────────

    def _get_provider_config(self, provider: str):
        if provider == "openrouter":
            return self.config.openrouter
        elif provider == "deepseek":
            return self.config.deepseek
        else:
            raise ValueError(
                f"Unknown provider {provider!r}. "
                f"Add its ProviderConfig to PipelineConfig and update this method."
            )
