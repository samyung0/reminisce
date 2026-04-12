"""Knowledge graph data endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.graph_builder import get_graph_data

router = APIRouter()


@router.get("/{source_id}")
async def graph_for_source(source_id: str):
    """Return nodes + edges for the knowledge graph of a source."""
    try:
        return get_graph_data(source_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index not found for this source")
