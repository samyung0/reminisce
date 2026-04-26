"""
Centralized configuration.
All model IDs and provider endpoints are configurable for easy swapping.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProviderConfig:
    """API provider configuration."""

    api_key: str = ""
    base_url: str = ""


@dataclass
class PipelineConfig:
    # ── Provider endpoints ──────────────────────────────────────
    openrouter: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    )
    deepseek: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    )

    # ── Model assignments by role ───────────────────────────────
    # Change these strings to swap models without touching any other code.
    model_ids: Dict[str, str] = field(
        default_factory=lambda: {
            "embedding": "qwen/qwen3-embedding-8b",  # OpenRouter
            "extraction": "deepseek-v4-flash",
            "image_caption": "deepseek-v4-flash",
            "community_summary": "deepseek-v4-flash",
            "enrichment": "deepseek-v4-flash",
            "retrieval_simple": "deepseek-v4-flash",
            "retrieval_detailed": "deepseek-v4-pro",
        }
    )

    # Which provider each role routes to ("openrouter" or "deepseek")
    model_providers: Dict[str, str] = field(
        default_factory=lambda: {
            "embedding": "openrouter",
            "extraction": "deepseek",
            "image_caption": "deepseek",
            "community_summary": "deepseek",
            "enrichment": "deepseek",
            "retrieval_simple": "deepseek",
            "retrieval_detailed": "deepseek",
        }
    )

    # ── Embedding dimensions ────────────────────────────────────
    embedding_dimensions: int = 4096

    # ── Neo4j ───────────────────────────────────────────────────
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = "neo4j"

    # ── Docling ─────────────────────────────────────────────────
    docling_ocr: bool = True
    docling_table_structure: bool = True
    docling_generate_images: bool = True

    # ── Chunking ────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # ── Graph Schema ────────────────────────────────────────────
    entity_types: List[str] = field(
        default_factory=lambda: [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "TECHNOLOGY",
            "CONCEPT",
            "EVENT",
            "PRODUCT",
            "DOCUMENT",
            "ABBREVIATION",
            "PROCESS",
            "METRIC",
            "REGULATION",
        ]
    )
    relation_types: List[str] = field(
        default_factory=lambda: [
            "MENTIONS",
            "PART_OF",
            "LOCATED_IN",
            "WORKS_FOR",
            "DEFINED_AS",
            "RELATED_TO",
            "DEPENDS_ON",
            "PRODUCES",
            "ABBREVIATED_AS",
            "ALSO_KNOWN_AS",
            "HAS_COMPONENT",
            "DESCRIBES",
            "PRECEDES",
            "FOLLOWS",
            "REGULATES",
        ]
    )

    # ── Enrichment ─────────────────────────────────────────────
    enable_summarization: bool = True
    enable_abbreviation_expansion: bool = True
    summary_context_window: int = 3

    # ── Community Management ────────────────────────────────────
    community_aggressive_threshold: int = 500  # below: rebuild all on insert
    community_max_levels: int = 3
    community_end_of_day_rebalance: bool = True

    # ── Retrieval ───────────────────────────────────────────────
    retrieval_top_k: int = 10
    retrieval_community_top_k: int = 5
    detailed_max_agent_iterations: int = 5

    # ── Indexes ─────────────────────────────────────────────────
    enable_vector_index: bool = True
    enable_fulltext_index: bool = True
