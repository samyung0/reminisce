from liteparse.types import ParseResult
from llama_cloud.types.parsing_get_response import (
    ItemsPageStructuredResultPage,
    ParsingGetResponse,
)
from llama_cloud.types.heading_item import HeadingItem
from llama_cloud.types.text_item import TextItem as LlamaTextItem
import os
import json
from dotenv import load_dotenv
from pathlib import Path
import pymupdf4llm

# 1. LlamaCloud (Managed SDK)
from llama_cloud import AsyncLlamaCloud

# 2. Docling (IBM Open Source)
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import (
    DocItem,
    FormulaItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

# 3. LiteParse (LlamaIndex Open Source)
# Note: LiteParse is CLI-first but has a library wrapper
from liteparse import LiteParse

import asyncio

load_dotenv(Path(__file__).parent.parent / ".env.test.local")
print(os.environ.get("LLAMA_CLOUD_API_KEY"))


async def llama_cloud_parser(file_path):
    print("Running LlamaCloud parser...")
    client = AsyncLlamaCloud(api_key=os.environ.get("LLAMA_CLOUD_API_KEY"))
    file_obj = await client.files.create(file=Path(file_path), purpose="parse")
    job: ParsingGetResponse = await client.parsing.parse(
        file_id=file_obj.id,
        version="latest",
        tier="agentic",
        output_options={
            "markdown": {
                "annotate_links": True,
            }
        },
        expand=["markdown", "text", "metadata", "items"],
    )

    with open("llama_cloud_output.raw.txt", "w") as f:
        f.write(json.dumps(job.model_dump(), indent=4, default=str))

    llama_output = []
    current_header = None
    if job.items:
        for page in job.items.pages:
            if not isinstance(page, ItemsPageStructuredResultPage):
                continue
            for item in page.items:
                entry: dict[str, object] = {
                    "type": item.type,
                    "page": page.page_number,
                    "parent_header": current_header,
                }
                if isinstance(item, HeadingItem):
                    entry["text"] = item.value
                    entry["heading_level"] = item.level
                    current_header = item.value
                elif isinstance(item, LlamaTextItem):
                    entry["text"] = item.value
                else:
                    entry["text"] = item.md if hasattr(item, "md") else None
                if item.bbox:
                    b = item.bbox[0]
                    entry["bbox"] = {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
                llama_output.append(entry)

    with open("llama_cloud_output.json", "w") as f:
        f.write(json.dumps(llama_output, indent=4, default=str))

    print("LlamaCloud parser completed")
    return llama_output


async def pymupdf4llm_parser(file_path):
    print("Running pymupdf4llm parser...")
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    with open("pymupdf4llm_output.raw.txt", "w") as f:
        f.write(json.dumps(pages, indent=4, default=str))

    structured_data = []
    current_header = None

    for page in pages:
        page_num = page["metadata"]["page_number"]
        toc_items = page.get("toc_items", [])
        if toc_items:
            current_header = toc_items[-1][1]

        boxes = page.get("page_boxes", [])
        if not boxes:
            structured_data.append(
                {
                    "text": page["text"][:200],
                    "type": "text",
                    "page": page_num,
                    "parent_header": current_header,
                    "toc_items": toc_items,
                }
            )
            continue

        for box in boxes:
            text = page["text"][box["pos"][0] : box["pos"][1]]
            if box["class"] == "text" and text.lstrip().startswith("#"):
                current_header = text.lstrip().lstrip("#").strip()
            structured_data.append(
                {
                    "text": text[:200],
                    "type": box["class"],
                    "bbox": box["bbox"],
                    "page": page_num,
                    "parent_header": current_header,
                    "toc_items": toc_items,
                }
            )

    with open("pymupdf4llm_output.json", "w") as f:
        f.write(json.dumps(structured_data, indent=4, default=str))

    print("pymupdf4llm parser completed")
    return structured_data


async def docling_parser(file_path):
    print("Running docling parser...")
    converter = DocumentConverter()
    doc_result = converter.convert(file_path)
    doc = doc_result.document

    with open("docling_output.raw.txt", "w") as f:
        f.write(json.dumps(doc.model_dump(), indent=4))

    docling_output = []
    current_header = None

    for item, level in doc.iterate_items():
        if not isinstance(item, DocItem):
            continue
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            current_header = item.text
        entry: dict[str, object] = {
            "label": item.label.value,
            "level": level,
            "parent_header": current_header,
        }
        if isinstance(item, TextItem):
            entry["text"] = item.text[:200]
        if isinstance(item, TableItem):
            entry["text"] = item.export_to_markdown(doc)
        if isinstance(item, FormulaItem):
            entry["formula_text"] = item.text or None
        if isinstance(item, SectionHeaderItem):
            entry["header_level"] = item.level
        if item.prov:
            p = item.prov[0]
            page_h = doc.pages[p.page_no].size.height
            bbox = p.bbox.to_top_left_origin(page_h)
            entry["page"] = p.page_no
            entry["bbox"] = {
                "l": bbox.l,
                "t": bbox.t,
                "r": bbox.r,
                "b": bbox.b,
            }
        docling_output.append(entry)

    with open("docling_output.json", "w") as f:
        f.write(json.dumps(docling_output, indent=4))

    print("docling parser completed")
    return docling_output


async def liteparse_parser(file_path):
    print("Running liteparse parser...")
    lite_parser = LiteParse()
    lite_result: ParseResult = lite_parser.parse(file_path)

    with open("liteparse_output.raw.txt", "w") as f:
        f.write(json.dumps(lite_result, indent=4, default=str))

    liteparse_output = []
    for page in lite_result.pages:
        for text_item in page.textItems:
            liteparse_output.append(
                {
                    "text": text_item.text[:200],
                    "page": page.pageNum,
                    "bbox": {
                        "x": text_item.x,
                        "y": text_item.y,
                        "w": text_item.width,
                        "h": text_item.height,
                    },
                    "font_name": text_item.fontName,
                    "font_size": text_item.fontSize,
                }
            )

    with open("liteparse_output.json", "w") as f:
        f.write(json.dumps(liteparse_output, indent=4))

    print("liteparse parser completed")
    return liteparse_output


async def main():
    file_path = Path(__file__).parent / "data" / "test.pdf"
    # llama_cloud_result = await llama_cloud_parser(file_path)
    # has no formulas
    # docling_result = await docling_parser(file_path)
    liteparse_result = await liteparse_parser(file_path)
    pymupdf4llm_result = await pymupdf4llm_parser(file_path)


if __name__ == "__main__":
    asyncio.run(main())
