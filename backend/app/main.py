from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import sources, graph, chat, quiz

app = FastAPI(title="Reminisce", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sources.router, prefix="/api/sources", tags=["sources"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(quiz.router, prefix="/api/quiz", tags=["quiz"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}
