# Reminisce

AI-powered study app that parses textbook PDFs into knowledge graphs, enables Q&A with citations, and generates quizzes.

## Prerequisites

- Python 3.12+
- Node.js 22+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## API Keys

You need two API keys (both have free tiers):

| Key | Where to get it |
|-----|----------------|
| **Google AI Studio** (Gemini) | https://aistudio.google.com/apikey |
| **LlamaCloud** (LlamaParse) | https://cloud.llamaindex.ai |

## Setup

```bash
# 1. Backend
cd backend
cp .env.example .env
# Edit .env with your API keys
uv sync

# 2. Frontend
cd ../frontend
npm install
```

## Running

```bash
# Terminal 1: Backend (from backend/)
uv run uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend (from frontend/)
npm run dev
```

Open http://localhost:3000

## Usage

1. Click **Upload PDF** and upload a textbook
2. Wait for parsing + knowledge graph construction
3. Explore the **Graph** tab to see extracted concepts and relationships
4. Use the **Chat** tab to ask questions — answers include clickable page citations
5. Use the **Quiz** tab to generate quizzes with configurable difficulty (Bloom's taxonomy levels 1-4)
# reminisce
