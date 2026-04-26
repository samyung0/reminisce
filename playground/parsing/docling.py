from typing import cast
import asyncio
import hashlib
import json
from io import BytesIO
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.doc_chunk import DocMeta
from docling.datamodel.base_models import InputFormat
from docling.datamodel.chart_extraction_options import (
    ChartExtractionModelKind,
    ChartExtractionModelOptions,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    FormulaItem,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.test.local")

OUT_DIR = Path(__file__).parent


def resolve_caption(doc: DoclingDocument, item) -> str | None:
    """Use the built-in caption_text method on FloatingItem subclasses."""
    if not hasattr(item, "caption_text"):
        return None
    text = item.caption_text(doc)
    return text if text else None


def build_section_path(
    item, level: int, section_stack: list[tuple[int, str]]
) -> list[str]:
    """Maintain a stack-based section hierarchy and return current path."""
    if isinstance(item, (SectionHeaderItem, TitleItem)):
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        text = item.text
        section_stack.append((level, text))
    return [s[1] for s in section_stack]


def extract_provenance(doc: DoclingDocument, item) -> dict | None:
    if not item.prov:
        return None
    p = item.prov[0]
    page_h = doc.pages[p.page_no].size.height
    bbox = p.bbox.to_top_left_origin(page_h)
    return {
        "page": p.page_no,
        "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b},
    }


def save_picture(item: PictureItem, doc: DoclingDocument, idx: int) -> str | None:
    """Save picture image to disk and return relative path."""
    img = item.get_image(doc)
    if img is None:
        return None
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_hash = hashlib.md5(buf.getvalue()).hexdigest()[:8]
    filename = f"pic_{idx}_{img_hash}.png"
    (img_dir / filename).write_bytes(buf.getvalue())
    return f"images/{filename}"


def extract_items(doc: DoclingDocument) -> list[dict]:
    items_out = []
    section_stack: list[tuple[int, str]] = []
    reading_order = 0

    for item, level in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue

        section_path = build_section_path(item, level, section_stack)

        entry: dict[str, object] = {
            "id": item.self_ref,
            "label": item.label.value,
            "level": level,
            "reading_order": reading_order,
            "section_path": list(section_path),
            "parent_ref": item.parent.cref if item.parent else None,
        }

        prov = extract_provenance(doc, item)
        if prov:
            entry.update(prov)

        if isinstance(item, (SectionHeaderItem, TitleItem)):
            entry["text"] = item.text
            if isinstance(item, SectionHeaderItem):
                entry["header_level"] = item.level

        elif isinstance(item, TextItem):
            entry["text"] = item.text

        elif isinstance(item, ListItem):
            entry["text"] = item.text
            entry["enumerated"] = item.enumerated
            entry["marker"] = item.marker

        elif isinstance(item, TableItem):
            entry["text_markdown"] = item.export_to_markdown(doc)
            entry["caption"] = resolve_caption(doc, item)

        elif isinstance(item, FormulaItem):
            entry["text"] = item.text or None

        elif isinstance(item, CodeItem):
            entry["text"] = item.text
            entry["code_language"] = (
                item.code_language.value if item.code_language else None
            )

        elif isinstance(item, PictureItem):
            entry["caption"] = resolve_caption(doc, item)
            entry["image_path"] = save_picture(item, doc, reading_order)
            entry["picture_classification"] = None
            for ann in item.annotations:
                if hasattr(ann, "provenance"):
                    entry["picture_classification"] = ann.provenance
                    break

        reading_order += 1
        items_out.append(entry)

    return items_out


def chunk_document(doc: DoclingDocument) -> list[dict]:
    """Use docling HybridChunker for token-aware semantic chunking."""
    chunker = HybridChunker()
    chunks_out = []
    for chunk in chunker.chunk(doc):
        meta: DocMeta = cast(DocMeta, chunk.meta)
        chunk_entry = {
            "text": chunk.text,
            "headings": meta.headings,
            "captions": meta.captions,
            "doc_items": [di.self_ref for di in meta.doc_items],
            "origin": meta.origin.filename if meta.origin else None,
        }
        if meta.doc_items:
            first = meta.doc_items[0]
            prov = extract_provenance(doc, first)
            if prov:
                chunk_entry["page"] = prov["page"]
        chunks_out.append(chunk_entry)
    return chunks_out


async def docling_parser(file_path: str) -> dict:
    print("Running docling parser...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.do_chart_extraction = True
    pipeline_options.chart_extraction_options = ChartExtractionModelOptions(
        model=ChartExtractionModelKind.GRANITE_VISION,
    )
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2
    pipeline_options.do_picture_classification = True
    # pipeline_options.do_picture_description = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    doc_result = converter.convert(file_path)
    doc = doc_result.document

    with open(OUT_DIR / "docling_output.raw.txt", "w") as f:
        f.write(json.dumps(doc.model_dump(), indent=4))

    items = extract_items(doc)
    chunks = chunk_document(doc)

    output = {
        "source": {
            "filename": doc.origin.filename if doc.origin else None,
            "hash": doc.origin.binary_hash if doc.origin else None,
            "page_count": len(doc.pages),
        },
        "items": items,
        "chunks": chunks,
    }

    with open(OUT_DIR / "docling_output.json", "w") as f:
        f.write(json.dumps(output, indent=4, default=str))

    print(f"Extracted {len(items)} items, {len(chunks)} chunks")
    return output


async def main():
    file_path = Path(__file__).parent / "data" / "test.pdf"
    await docling_parser(file_path.as_posix())


if __name__ == "__main__":
    asyncio.run(main())
