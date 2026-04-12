"""RAG query engine with citation support."""

from __future__ import annotations

from llama_index.core.response_synthesizers import ResponseMode

from app.services.graph_builder import load_index
from app.llm_adapter import get_llm, get_embed_model


async def query_source(source_id: str, question: str) -> dict:
    """Query a source's index and return an answer with source citations."""
    index = load_index(source_id)

    query_engine = index.as_query_engine(
        llm=get_llm(),
        include_text=True,
        response_mode=ResponseMode.TREE_SUMMARIZE,
        embedding_mode="hybrid",
        similarity_top_k=5,
    )

    response = await query_engine.aquery(question)

    citations: list[dict] = []
    for node in response.source_nodes:
        meta = node.metadata or {}
        citations.append({
            "text": node.get_content()[:300],
            "page_number": meta.get("page_number"),
            "source_file": meta.get("source_file"),
            "source_id": meta.get("source_id"),
            "score": node.score,
        })

    return {
        "answer": str(response),
        "citations": citations,
    }
