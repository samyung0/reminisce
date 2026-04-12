"""Quiz generation using Bloom's taxonomy difficulty levels."""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.services.graph_builder import load_index
from app.llm_adapter import get_llm

BLOOM_LEVELS = {
    1: {
        "name": "Remember",
        "instruction": (
            "Ask factual recall questions. Target single definitions, names, "
            "dates, or formulas that appear directly in the source material."
        ),
    },
    2: {
        "name": "Understand",
        "instruction": (
            "Ask questions that require explaining concepts in the student's "
            "own words, comparing two ideas, or describing why something works."
        ),
    },
    3: {
        "name": "Apply",
        "instruction": (
            "Ask questions that require applying a concept to a new scenario, "
            "performing a calculation, or predicting an outcome."
        ),
    },
    4: {
        "name": "Analyze",
        "instruction": (
            "Ask questions that require breaking down complex relationships, "
            "comparing across multiple topics, or evaluating trade-offs. "
            "These should require multi-step reasoning."
        ),
    },
}


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: int
    explanation: str
    bloom_level: int
    related_nodes: list[str]


class Quiz(BaseModel):
    questions: list[QuizQuestion]


QUIZ_SYSTEM_PROMPT = """You are an expert educational quiz generator. Given study material, \
generate multiple-choice questions.

RULES:
- Each question must have exactly 4 options (A through D).
- correct_answer is the 0-based index of the correct option.
- explanation should cite specific concepts from the material.
- related_nodes should list the key concept names the question tests.
- Respond ONLY with valid JSON matching the schema below.

Schema:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": 0,
      "explanation": "...",
      "bloom_level": 1,
      "related_nodes": ["Concept A", "Concept B"]
    }}
  ]
}}"""


async def generate_quiz(
    source_id: str,
    difficulty: int = 1,
    num_questions: int = 5,
    scope: str | None = None,
) -> dict:
    """Generate a quiz from the knowledge graph of a source.

    Args:
        source_id: The source to quiz on.
        difficulty: Bloom's level 1-4.
        num_questions: How many questions to generate.
        scope: Optional topic/chapter to focus on.
    """
    difficulty = max(1, min(4, difficulty))
    bloom = BLOOM_LEVELS[difficulty]

    index = load_index(source_id)

    scope_query = scope or "all main topics and concepts"
    retriever = index.as_retriever(similarity_top_k=10)
    nodes = await retriever.aretrieve(scope_query)

    context_chunks = []
    for node in nodes:
        meta = node.metadata or {}
        page = meta.get("page_number", "?")
        context_chunks.append(f"[Page {page}]\n{node.get_content()[:600]}")

    context = "\n\n---\n\n".join(context_chunks)

    llm = get_llm()
    prompt = (
        f"{QUIZ_SYSTEM_PROMPT}\n\n"
        f"DIFFICULTY: Level {difficulty} ({bloom['name']})\n"
        f"{bloom['instruction']}\n\n"
        f"SCOPE: {scope_query}\n"
        f"NUMBER OF QUESTIONS: {num_questions}\n\n"
        f"STUDY MATERIAL:\n{context}"
    )

    response = await llm.acomplete(prompt)
    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

    quiz_data = json.loads(raw_text)
    quiz = Quiz(**quiz_data)

    return quiz.model_dump()
