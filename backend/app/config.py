from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
STORAGE_DIR = BASE_DIR / "storage"

UPLOADS_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
