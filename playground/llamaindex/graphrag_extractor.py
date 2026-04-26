"""
GraphRAG-style entity/relationship extractor following the LlamaIndex cookbook:
https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/

Key differences from SimpleLLMPathExtractor:
  - Extracts entity descriptions (richer node properties)
  - Extracts relationship descriptions
  - Uses chunk_summary and resolved_entities from enrichment
  - Handles abbreviation cross-references
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Sequence

from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

logger = logging.getLogger(__name__)


# ── Extraction prompt (GraphRAG cookbook style) ─────────────────

GRAPH_RAG_EXTRACT_PROMPT = """\
-Goal-
Given a text document that is potentially relevant to this activity and a list of \
entity types, identify all entities of those types from the text and all \
relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
   - entity_name: Name of the entity, capitalized
   - entity_type: One of the specified entity types: {entity_types}
   - entity_description: Comprehensive description of the entity's attributes and activities
   Format each entity as: ("entity"<TUPLE_SEP><entity_name><TUPLE_SEP><entity_type><TUPLE_SEP><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) \
that are clearly related to each other.
   - relationship_type: One of: {relation_types}
   - relationship_description: Explanation of why source and target are related
   Format: ("relationship"<TUPLE_SEP><source_entity><TUPLE_SEP><target_entity><TUPLE_SEP><relationship_type><TUPLE_SEP><relationship_description>)

3. ABBREVIATION HANDLING: When you see an expanded abbreviation like \
"Application Programming Interface (API)":
   - Create entity "Application Programming Interface" with appropriate type
   - Create entity "API" with type ABBREVIATION
   - Create relationship ("Application Programming Interface", "ABBREVIATED_AS", "API")

4. Return output as a single list of all entities and relationships. \
Use **{record_delimiter}** as the list delimiter.

5. When finished, output **{completion_delimiter}**

-Real Data-
######################
text: {text}
######################
chunk_summary: {chunk_summary}
resolved_entities: {resolved_entities}
source_document: {source_document}
page_number: {page_number}
######################
output:"""


ENTITY_TYPES_STR = ", ".join
RELATION_TYPES_STR = ", ".join


class GraphRAGExtractor(TransformComponent):
    """
    GraphRAG-style extractor that produces rich entity and relationship
    descriptions, handles abbreviations, and leverages chunk summaries
    for context-aware extraction.

    Follows the cookbook pattern but integrates with LlamaIndex's
    PropertyGraphIndex via the kg_extractors interface.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    max_paths_per_chunk: int
    num_workers: int
    entity_types: List[str]
    relation_types: List[str]

    def __init__(
        self,
        llm: LLM,
        entity_types: List[str],
        relation_types: List[str],
        max_paths_per_chunk: int = 50,
        num_workers: int = 4,
        **kwargs,
    ):
        prompt = GRAPH_RAG_EXTRACT_PROMPT.format(
            entity_types=", ".join(entity_types),
            relation_types=", ".join(relation_types),
            text="{text}",
            chunk_summary="{chunk_summary}",
            resolved_entities="{resolved_entities}",
            source_document="{source_document}",
            page_number="{page_number}",
            tuple_sep="<TUPLE_SEP>",
            record_delimiter="<REC_DELIM>",
            completion_delimiter="<COMPLETE>",
        )

        super().__init__(
            **{
                "llm": llm,
                "extract_prompt": PromptTemplate(prompt),
                "max_paths_per_chunk": max_paths_per_chunk,
                "num_workers": num_workers,
                "entity_types": entity_types,
                "relation_types": relation_types,
            }
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphRAGExtractor"

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting GraphRAG paths",
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract rich entities and relations into LlamaIndex metadata keys."""
        metadata = node.metadata
        text = node.get_content(metadata_mode=MetadataMode.LLM)

        try:
            response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                chunk_summary=metadata.get("chunk_summary", "N/A"),
                resolved_entities=str(metadata.get("resolved_entities", [])),
                source_document=metadata.get("source_document", "unknown"),
                page_number=metadata.get("page_number", -1),
            )
            entities, relations = self._parse_response(response)
        except Exception as exc:
            logger.warning(f"GraphRAG extraction failed for node {node.node_id}: {exc}")
            entities, relations = {}, []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        base_properties = node.metadata.copy()

        entity_nodes: Dict[str, EntityNode] = {}
        for name, props in entities.items():
            clean_name = self._normalize_name(name)
            if not clean_name:
                continue
            node_props = {
                **base_properties,
                "description": props.get("description", ""),
                "type": props.get("type", "CONCEPT"),
            }
            entity_nodes[clean_name] = EntityNode(
                name=clean_name,
                label=props.get("type", "CONCEPT"),
                properties=node_props,
            )

        for rel in relations:
            source = self._normalize_name(rel["source"])
            target = self._normalize_name(rel["target"])
            if not source or not target:
                continue

            if source not in entity_nodes:
                entity_nodes[source] = EntityNode(
                    name=source, label="CONCEPT", properties=base_properties.copy()
                )
            if target not in entity_nodes:
                entity_nodes[target] = EntityNode(
                    name=target, label="CONCEPT", properties=base_properties.copy()
                )

            existing_relations.append(
                Relation(
                    label=self._normalize_label(rel["type"]),
                    source_id=entity_nodes[source].id,
                    target_id=entity_nodes[target].id,
                    properties={
                        **base_properties,
                        "description": rel.get("description", ""),
                    },
                )
            )

        existing_nodes.extend(entity_nodes.values())
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    def _parse_response(self, response_text: str) -> tuple[Dict[str, Dict], List[Dict]]:
        """
        Parse GraphRAG-style response into ExtractedPath objects.

        Expected format:
          ("entity"<TUPLE_SEP>name<TUPLE_SEP>type<TUPLE_SEP>desc)
          ("relationship"<TUPLE_SEP>src<TUPLE_SEP>tgt<TUPLE_SEP>rel<TUPLE_SEP>desc)
        """
        entities: Dict[str, Dict] = {}  # name → {type, description}
        relations: List[Dict] = []

        # Split by record delimiter
        records = response_text.split("<REC_DELIM>")

        for record in records:
            record = record.strip()
            if not record or record == "<COMPLETE>":
                continue

            # Try to extract tuple content
            match = re.search(r'\("([^"]+)"<TUPLE_SEP>(.*?)\)', record, re.DOTALL)
            if not match:
                continue

            record_type = match.group(1)
            content = match.group(2)

            parts = content.split("<TUPLE_SEP>")
            parts = [p.strip().strip('"') for p in parts]

            if record_type == "entity" and len(parts) >= 3:
                entities[parts[0]] = {
                    "type": parts[1],
                    "description": parts[2],
                }

            elif record_type == "relationship" and len(parts) >= 4:
                source = parts[0]
                target = parts[1]
                rel_type = parts[2]
                description = parts[3] if len(parts) > 3 else ""

                relations.append(
                    {
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "description": description,
                    }
                )

        return entities, relations

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"\s+", " ", name or "").strip()

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = re.sub(r"[^0-9A-Za-z_]+", "_", label or "RELATED_TO")
        label = re.sub(r"_+", "_", label).strip("_")
        return label.upper() or "RELATED_TO"
