"""
OpenAI-compatible embedding adapter.
Works with OpenRouter (Qwen3-Embedding-8B), DeepSeek, or any
OpenAI-compatible /embeddings endpoint.
"""

import logging
from typing import List, Optional

import httpx
from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    Embedding client for any OpenAI-compatible /embeddings endpoint.

    Tested with:
      - OpenRouter:  qwen/qwen3-embedding-8b
      - OpenAI:      text-embedding-3-small / large
      - DeepSeek:    (if they add an embedding endpoint)
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        api_key: str = "",
        api_base: str = "https://openrouter.ai/api/v1",
        dimensions: Optional[int] = None,
        embed_batch_size: int = 50,
        **kwargs,
    ):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        self._model = model
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")
        self._dimensions = dimensions

    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"

    # ── HTTP call ───────────────────────────────────────────────

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/docling-graphrag-pipeline",
        }
        payload: dict = {"model": self._model, "input": texts}
        if self._dimensions:
            payload["dimensions"] = self._dimensions

        resp = httpx.post(
            f"{self._api_base}/embeddings",
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings_sorted = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_sorted]

    # ── BaseEmbedding interface ─────────────────────────────────

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._call_api([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._call_api([text])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._call_api(texts)
