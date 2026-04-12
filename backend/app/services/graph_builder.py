"""Build and manage PropertyGraphIndex from parsed documents."""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import StorageContext, PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)
from llama_index.core.schema import Document

from app.config import STORAGE_DIR
from app.llm_adapter import get_llm, get_embed_model

_index_cache: dict[str, PropertyGraphIndex] = {}


def _source_storage_dir(source_id: str) -> Path:
    d = STORAGE_DIR / "indexes" / source_id
    d.mkdir(parents=True, exist_ok=True)
    return d


async def build_graph_for_source(
    source_id: str, documents: list[Document]
) -> int:
    """Build a PropertyGraphIndex for a source and persist it locally.

    Returns the number of graph nodes extracted.
    """
    llm = get_llm()
    embed_model = get_embed_model()
    persist_dir = _source_storage_dir(source_id)

    graph_store = SimplePropertyGraphStore()

    storage_context = StorageContext.from_defaults(
        property_graph_store=graph_store,
    )

    kg_extractors = [
        SimpleLLMPathExtractor(llm=llm, max_paths_per_chunk=20),
        ImplicitPathExtractor(),
    ]

    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=kg_extractors,
        llm=llm,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(persist_dir))

    _index_cache[source_id] = index

    triplets = graph_store.get_triplets(entity_names=[])
    return len(triplets)


def load_index(source_id: str) -> PropertyGraphIndex:
    """Load a persisted PropertyGraphIndex from disk (with caching)."""
    if source_id in _index_cache:
        return _index_cache[source_id]

    persist_dir = _source_storage_dir(source_id)
    if not (persist_dir / "docstore.json").exists():
        raise FileNotFoundError(f"No index found for source {source_id}")

    graph_store = SimplePropertyGraphStore()
    storage_context = StorageContext.from_defaults(
        persist_dir=str(persist_dir),
        property_graph_store=graph_store,
    )

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        llm=get_llm(),
        embed_model=get_embed_model(),
        storage_context=storage_context,
    )

    _index_cache[source_id] = index
    return index


def get_graph_data(source_id: str) -> dict:
    """Return nodes and edges suitable for frontend graph visualization."""
    index = load_index(source_id)
    graph_store: SimplePropertyGraphStore = (
        index.storage_context.property_graph_store  # type: ignore[assignment]
    )

    triplets = graph_store.get_triplets(entity_names=[])

    nodes_set: dict[str, dict] = {}
    edges: list[dict] = []

    for subj, rel, obj in triplets:
        subj_name = subj.name if hasattr(subj, "name") else str(subj)
        obj_name = obj.name if hasattr(obj, "name") else str(obj)
        rel_label = rel.label if hasattr(rel, "label") else str(rel)

        if subj_name not in nodes_set:
            props = subj.properties if hasattr(subj, "properties") else {}
            nodes_set[subj_name] = {
                "id": subj_name,
                "label": subj_name,
                "type": props.get("entity_type", "concept"),
            }
        if obj_name not in nodes_set:
            props = obj.properties if hasattr(obj, "properties") else {}
            nodes_set[obj_name] = {
                "id": obj_name,
                "label": obj_name,
                "type": props.get("entity_type", "concept"),
            }

        edges.append({
            "source": subj_name,
            "target": obj_name,
            "label": rel_label,
        })

    return {
        "nodes": list(nodes_set.values()),
        "edges": edges,
    }
