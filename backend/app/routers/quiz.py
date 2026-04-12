"""Quiz generation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from app.services.quiz import generate_quiz

router = APIRouter()


class QuizRequest(BaseModel):
    source_id: str
    difficulty: int = Field(default=1, ge=1, le=4)
    num_questions: int = Field(default=5, ge=1, le=20)
    scope: str | None = None


@router.post("/generate")
async def create_quiz(req: QuizRequest):
    try:
        result = await generate_quiz(
            source_id=req.source_id,
            difficulty=req.difficulty,
            num_questions=req.num_questions,
            scope=req.scope,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source index not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")
