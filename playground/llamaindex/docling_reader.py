"""
Docling parser with bounding-box provenance and image captioning.

Bounding boxes are attached to every TextNode so we can trace
back to the exact location on the page.

Images extracted by Docling are captioned using DeepSeek V4 Flash
(vision) and inserted as additional TextNodes.
"""

import base64
import io
import logging
from pathlib import Path
from typing import cast, Dict, List, Optional

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core.schema import TextNode
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)


class DoclingReaderWithBBox:
    """Parse documents via Docling with bbox metadata + image captioning."""

    def __init__(self, config, caption_llm: Optional[LLM] = None):
        self.config = config
        self.caption_llm = caption_llm
        self._converter = self._init_converter()

    # ── Converter init ──────────────────────────────────────────

    def _init_converter(self):
        from docling.document_converter import DocumentConverter

        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import InputFormat, PdfFormatOption

            pipeline_opts = PdfPipelineOptions(
                do_ocr=self.config.docling_ocr,
                do_table_structure=self.config.docling_table_structure,
                generate_page_images=self.config.docling_generate_images,
                generate_picture_images=self.config.docling_generate_images,
            )
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
                }
            )
        except Exception:
            logger.warning("Advanced Docling options unavailable; using defaults.")
            return DocumentConverter()

    # ── Public API ──────────────────────────────────────────────

    def parse_document(self, file_path: str) -> List[TextNode]:
        logger.info(f"Parsing: {file_path}")
        result = self._converter.convert(file_path)
        doc = result.document
        source_name = Path(file_path).name
        source_path = str(Path(file_path).resolve())

        # Text chunks with bbox
        try:
            nodes = self._chunk_with_docling(doc, source_name, source_path)
        except Exception as exc:
            logger.warning(f"Docling chunker failed ({exc}); falling back")
            nodes = self._chunk_with_splitter(doc, source_name, source_path)

        # Image captioning
        if self.config.docling_generate_images and self.caption_llm:
            image_nodes = self._extract_and_caption_images(
                doc, source_name, source_path
            )
            nodes.extend(image_nodes)

        self._link_siblings(nodes)
        logger.info(f"Extracted {len(nodes)} nodes from {source_name}")
        return nodes

    def parse_documents(self, file_paths: List[str]) -> List[TextNode]:
        all_nodes: List[TextNode] = []
        for fp in file_paths:
            all_nodes.extend(self.parse_document(fp))
        return all_nodes

    # ── Docling-native chunking ─────────────────────────────────

    def _chunk_with_docling(
        self, doc, source_name: str, source_path: str
    ) -> List[TextNode]:
        from docling.chunking import HybridChunker

        chunker = HybridChunker()
        doc_chunks = list(chunker.chunk(doc))

        nodes: List[TextNode] = []
        for idx, chunk in enumerate(doc_chunks):
            prov_list = self._collect_chunk_provenance(chunk)
            primary_page = prov_list[0]["page_number"] if prov_list else -1
            primary_bbox = prov_list[0]["bbox"] if prov_list else {}

            node = TextNode(
                text=chunk.text,
                metadata={
                    "source_document": source_name,
                    "source_path": source_path,
                    "page_number": primary_page,
                    "bbox": primary_bbox,
                    "all_provenance": prov_list,
                    "chunk_index": idx,
                    "headings": self._get_headings(chunk),
                    "node_type": "text",
                },
                excluded_llm_metadata_keys=[
                    "bbox", "all_provenance", "source_path",
                    "chunk_index", "node_type",
                ],
                excluded_embed_metadata_keys=[
                    "bbox", "all_provenance", "source_path",
                    "chunk_index", "page_number", "node_type",
                ],
            )
            nodes.append(node)
        return nodes

    # ── Fallback chunking ───────────────────────────────────────

    def _chunk_with_splitter(
        self, doc, source_name: str, source_path: str
    ) -> List[TextNode]:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document as LIDocument

        raw_nodes: List[TextNode] = []
        for item, _ in doc.iterate_items():
            text = getattr(item, "text", "")
            if not text or not text.strip():
                continue
            prov_list = []
            if hasattr(item, "prov"):
                for prov in item.prov:
                    prov_list.append(self._parse_provenance(prov))
            primary_page = prov_list[0]["page_number"] if prov_list else -1
            primary_bbox = prov_list[0]["bbox"] if prov_list else {}
            raw_nodes.append(TextNode(
                text=text.strip(),
                metadata={
                    "source_document": source_name,
                    "source_path": source_path,
                    "page_number": primary_page,
                    "bbox": primary_bbox,
                    "all_provenance": prov_list,
                    "node_type": "text",
                },
            ))

        combined = LIDocument(
            text="\n\n".join(n.text for n in raw_nodes),
            metadata={"source_document": source_name, "source_path": source_path},
        )
        splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunked = splitter.get_nodes_from_documents([combined])

        for chunk_node in chunked:
            best = self._find_best_raw_node(chunk_node.get_content(), raw_nodes)
            if best:
                chunk_node.metadata.update({
                    "page_number": best.metadata.get("page_number", -1),
                    "bbox": best.metadata.get("bbox", {}),
                    "all_provenance": best.metadata.get("all_provenance", []),
                    "source_document": source_name,
                    "source_path": source_path,
                    "node_type": "text",
                })
            chunk_node.excluded_llm_metadata_keys = [
                "bbox", "all_provenance", "source_path", "node_type",
            ]
            chunk_node.excluded_embed_metadata_keys = [
                "bbox", "all_provenance", "source_path", "page_number", "node_type",
            ]
        return cast(List[TextNode], list(chunked))

    # ── Image extraction + captioning ───────────────────────────

    def _extract_and_caption_images(
        self, doc, source_name: str, source_path: str
    ) -> List[TextNode]:
        """
        Extract images from Docling document, caption them with
        DeepSeek V4 Flash (vision), and return as TextNodes.
        """
        image_nodes: List[TextNode] = []

        try:
            for item, _ in doc.iterate_items():
                # Docling exposes pictures with image data
                if not hasattr(item, "image"):
                    continue
                if item.image is None:
                    continue

                # Get image bytes
                img = item.image
                if hasattr(img, "pil_image"):
                    buf = io.BytesIO()
                    img.pil_image.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                elif hasattr(img, "bytes"):
                    img_bytes = img.bytes
                else:
                    continue

                # Get provenance for the image
                prov_list = []
                if hasattr(item, "prov"):
                    for prov in item.prov:
                        prov_list.append(self._parse_provenance(prov))

                primary_page = prov_list[0]["page_number"] if prov_list else -1
                primary_bbox = prov_list[0]["bbox"] if prov_list else {}

                # Caption with DeepSeek V4 Flash (vision)
                caption = self._caption_image(img_bytes, source_name, primary_page)

                node = TextNode(
                    text=f"[Image Caption]: {caption}",
                    metadata={
                        "source_document": source_name,
                        "source_path": source_path,
                        "page_number": primary_page,
                        "bbox": primary_bbox,
                        "all_provenance": prov_list,
                        "node_type": "image",
                        "image_caption": caption,
                    },
                    excluded_llm_metadata_keys=[
                        "bbox", "all_provenance", "source_path", "node_type",
                    ],
                    excluded_embed_metadata_keys=[
                        "bbox", "all_provenance", "source_path",
                        "page_number", "node_type",
                    ],
                )
                image_nodes.append(node)

        except Exception as exc:
            logger.warning(f"Image extraction/captioning failed: {exc}")

        logger.info(f"Captioned {len(image_nodes)} images from {source_name}")
        return image_nodes

    def _caption_image(self, img_bytes: bytes, source_name: str, page: int) -> str:
        """Send image to DeepSeek V4 Flash for captioning."""
        if not self.caption_llm:
            return "No caption available"

        b64 = base64.b64encode(img_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"

        try:
            from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock

            messages = [
                ChatMessage(
                    role="user",
                    blocks=[
                        TextBlock(text=(
                            "Describe this image from a document in detail. "
                            "Focus on: charts/diagrams (describe axes, trends, legends), "
                            "tables (describe structure and key values), "
                            "or any other visual content. "
                            "Be specific and include any visible text or numbers."
                        )),
                        ImageBlock(url=data_url),
                    ],
                )
            ]
            resp = self.caption_llm.chat(messages)
            return (resp.message.content or "").strip()
        except Exception as exc:
            logger.warning(f"Image captioning failed: {exc}")
            return f"Image from {source_name} page {page} (caption unavailable)"

    # ── Provenance helpers ──────────────────────────────────────

    @staticmethod
    def _parse_provenance(prov) -> Dict:
        page_no = getattr(prov, "page_no", -1)
        bbox = getattr(prov, "bbox", None)
        bbox_dict = {}
        if bbox is not None:
            for key in ("l", "t", "r", "b"):
                if hasattr(bbox, key):
                    bbox_dict[key] = getattr(bbox, key)
            bbox_dict["unit"] = getattr(bbox, "unit", "pt")
        return {"page_number": page_no, "bbox": bbox_dict}

    @staticmethod
    def _collect_chunk_provenance(chunk) -> List[Dict]:
        prov_list: List[Dict] = []
        if not hasattr(chunk, "meta") or chunk.meta is None:
            return prov_list
        doc_items = getattr(chunk.meta, "doc_items", [])
        for item in doc_items:
            if hasattr(item, "prov"):
                for prov in item.prov:
                    prov_list.append(
                        DoclingReaderWithBBox._parse_provenance(prov)
                    )
        return prov_list

    @staticmethod
    def _get_headings(chunk) -> List[str]:
        if not hasattr(chunk, "meta") or chunk.meta is None:
            return []
        return list(getattr(chunk.meta, "headings", []) or [])

    @staticmethod
    def _find_best_raw_node(chunk_text: str, raw_nodes) -> Optional[TextNode]:
        best, best_overlap = None, 0
        for rn in raw_nodes:
            overlap = len(set(chunk_text.split()) & set(rn.text.split()))
            if overlap > best_overlap:
                best_overlap, best = overlap, rn
        return best

    @staticmethod
    def _link_siblings(nodes: List[TextNode]):
        for i, node in enumerate(nodes):
            if i > 0:
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=nodes[i - 1].node_id,
                )
            if i + 1 < len(nodes):
                node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=nodes[i + 1].node_id,
                )