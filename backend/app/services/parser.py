"""PDF parsing via LlamaParse → structured LlamaIndex Documents."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from llama_index.core.schema import Document

from app.config import LLAMA_CLOUD_API_KEY, UPLOADS_DIR, STORAGE_DIR


async def parse_pdf(file_path: Path) -> list[Document]:
    """Parse a PDF with LlamaParse and return LlamaIndex Documents with metadata.

    Uses the llama_cloud SDK directly:
    1. Create a parsing job (upload file).
    2. Poll until complete.
    3. Fetch markdown results per page.
    """
    from llama_cloud import AsyncLlamaCloud
    from llama_cloud.resources.parsing import poll_until_complete_async

    client = AsyncLlamaCloud(token=LLAMA_CLOUD_API_KEY)

    with open(file_path, "rb") as f:
        job = await client.parsing.create(
            tier="fast",
            version="latest",
            upload_file=(file_path.name, f, "application/pdf"),
        )

    job_id = job.id

    await poll_until_complete_async(
        get_status_fn=lambda: client.parsing.get(job_id),
        is_complete_fn=lambda r: r.job.status == "COMPLETED",
        is_error_fn=lambda r: r.job.status in ("FAILED", "CANCELLED"),
        get_error_message_fn=lambda r: r.job.error_message or "Parse failed",
        polling_interval=2.0,
        timeout=600.0,
    )

    result = await client.parsing.get(
        job_id,
        expand=["markdown"],
    )

    documents: list[Document] = []
    source_name = file_path.stem

    if result.markdown and result.markdown.pages:
        for page in result.markdown.pages:
            page_num = page.page_number if hasattr(page, "page_number") else 0
            text = page.markdown if hasattr(page, "markdown") else str(page)

            metadata = {
                "source_id": source_name,
                "source_file": file_path.name,
                "page_number": page_num,
            }

            documents.append(
                Document(
                    text=text,
                    metadata=metadata,
                    doc_id=f"{source_name}_page_{page_num}",
                )
            )
    elif result.markdown_full:
        documents.append(
            Document(
                text=result.markdown_full,
                metadata={
                    "source_id": source_name,
                    "source_file": file_path.name,
                    "page_number": 1,
                },
                doc_id=f"{source_name}_full",
            )
        )

    return documents


def save_source_metadata(
    source_id: str,
    filename: str,
    num_pages: int,
    num_nodes: int,
) -> dict:
    """Persist a small JSON manifest for each uploaded source."""
    meta_dir = STORAGE_DIR / "sources"
    meta_dir.mkdir(exist_ok=True)

    meta = {
        "id": source_id,
        "filename": filename,
        "num_pages": num_pages,
        "num_nodes": num_nodes,
    }

    (meta_dir / f"{source_id}.json").write_text(json.dumps(meta, indent=2))
    return meta


def list_sources() -> list[dict]:
    meta_dir = STORAGE_DIR / "sources"
    if not meta_dir.exists():
        return []
    sources = []
    for p in sorted(meta_dir.glob("*.json")):
        sources.append(json.loads(p.read_text()))
    return sources


def get_source(source_id: str) -> dict | None:
    path = STORAGE_DIR / "sources" / f"{source_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def generate_source_id() -> str:
    return uuid.uuid4().hex[:12]
