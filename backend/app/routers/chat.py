"""Q&A chat endpoint with citations."""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from app.services.rag import query_source

router = APIRouter()


class ChatRequest(BaseModel):
    source_id: str
    question: str


@router.post("/")
async def chat(req: ChatRequest):
    try:
        result = await query_source(req.source_id, req.question)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source index not found")
