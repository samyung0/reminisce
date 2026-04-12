"""PDF upload + source management endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import UPLOADS_DIR
from app.services.parser import (
    parse_pdf,
    save_source_metadata,
    list_sources,
    get_source,
    generate_source_id,
)
from app.services.graph_builder import build_graph_for_source

router = APIRouter()


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    source_id = generate_source_id()
    dest = UPLOADS_DIR / f"{source_id}_{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    documents = await parse_pdf(dest)

    num_nodes = await build_graph_for_source(source_id, documents)

    meta = save_source_metadata(
        source_id=source_id,
        filename=file.filename,
        num_pages=len(documents),
        num_nodes=num_nodes,
    )

    return {"message": "Source uploaded and indexed", "source": meta}


@router.get("/")
async def get_sources():
    return list_sources()


@router.get("/{source_id}")
async def get_source_detail(source_id: str):
    source = get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.get("/{source_id}/pdf")
async def serve_pdf(source_id: str):
    """Return the original PDF file for rendering in the frontend."""
    from fastapi.responses import FileResponse

    source = get_source(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    for p in UPLOADS_DIR.glob(f"{source_id}_*"):
        return FileResponse(p, media_type="application/pdf")

    raise HTTPException(status_code=404, detail="PDF file not found on disk")
