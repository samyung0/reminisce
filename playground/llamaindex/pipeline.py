"""
Main orchestration — wires everything together:
  Docling → Enrich → Extract → Neo4j → Communities → Indexes
"""

import logging
from typing import Any, List, Literal, Optional, Type, cast

from llama_index.core import Settings
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
    SchemaLLMPathExtractor,
)

from config import PipelineConfig
from llm_registry import LLMRegistry
from docling_reader import DoclingReaderWithBBox
from node_enrichment import NodeEnricher
from graphrag_extractor import GraphRAGExtractor
from graphrag_store import GraphRAGStore
from community_manager import CommunityManager
from retrieval import RetrievalEngine

logger = logging.getLogger(__name__)


class PropertyGraphPipeline:
    """End-to-end pipeline with LLM registry middleware."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.registry = LLMRegistry(config)

        # Set global defaults (used by PropertyGraphIndex internals)
        Settings.llm = self.registry.get_llm("extraction")
        Settings.embed_model = self.registry.get_embed_model()

        # Sub-components
        self.reader = DoclingReaderWithBBox(
            config,
            caption_llm=self.registry.get_llm("image_caption"),
        )
        self.enricher = NodeEnricher(self.registry.get_llm("enrichment"), config)
        self.graph_store: Optional[GraphRAGStore] = None
        self.community_mgr: Optional[CommunityManager] = None
        self.index: Optional[PropertyGraphIndex] = None
        self.retrieval: Optional[RetrievalEngine] = None

    # ── Build ───────────────────────────────────────────────────

    def build(self, file_paths: List[str]) -> PropertyGraphIndex:
        """
        Full build pipeline:
          1. Parse (Docling + bbox + image captioning)
          2. Enrich (summary + abbrev expansion)
          3. Build PropertyGraphIndex with all 3 extractors
          4. Community detection + summarization
          5. Neo4j indexes
          6. Wire up retrieval engine
        """
        # Step 1
        logger.info("Step 1/6 — Parsing documents …")
        nodes = self.reader.parse_documents(file_paths)

        # Step 2
        logger.info("Step 2/6 — Enriching nodes …")
        nodes = self.enricher.enrich_nodes(nodes)

        # Step 3
        logger.info("Step 3/6 — Building property graph …")
        self.graph_store = GraphRAGStore(
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            url=self.config.neo4j_uri,
            database=self.config.neo4j_database,
            community_llm=self.registry.get_llm("community_summary"),
        )

        extractors = self._create_extractors()

        self.index = PropertyGraphIndex(
            nodes=nodes,
            kg_extractors=extractors,
            property_graph_store=self.graph_store,
            embed_model=self.registry.get_embed_model(),
            llm=self.registry.get_llm("extraction"),
            show_progress=True,
        )

        # Step 4
        logger.info("Step 4/6 — Building communities …")
        self.community_mgr = CommunityManager(
            store=self.graph_store,
            summary_llm=self.registry.get_llm("community_summary"),
            aggressive_threshold=self.config.community_aggressive_threshold,
            max_levels=self.config.community_max_levels,
        )
        self.community_mgr.on_nodes_inserted(list(self._get_entity_names_from_store()))

        # Step 5
        logger.info("Step 5/6 — Creating Neo4j indexes …")
        self._create_neo4j_indexes()

        # Step 6
        logger.info("Step 6/6 — Wiring retrieval engine …")
        self.retrieval = RetrievalEngine(
            store=self.graph_store,
            community_manager=self.community_mgr,
            llm_registry=self.registry,
            config=self.config,
        )

        logger.info("✅ Pipeline build complete!")
        return self.index

    # ── Incremental insert ──────────────────────────────────────

    def insert_documents(self, file_paths: List[str]):
        """Insert new documents into the existing graph."""
        if self.index is None:
            raise RuntimeError("Pipeline not built yet. Call build() first.")

        logger.info(f"Inserting {len(file_paths)} new documents …")
        nodes = self.reader.parse_documents(file_paths)
        nodes = self.enricher.enrich_nodes(nodes)

        # Get entity names before insert (to detect new ones)
        existing = self._get_entity_names_from_store()

        # Insert into index
        self.index.insert_nodes(nodes)

        # Detect new entities and update communities
        current = self._get_entity_names_from_store()
        new_entities = list(current - existing)
        if self.community_mgr is None:
            raise RuntimeError("Community manager not initialized.")
        self.community_mgr.on_nodes_inserted(new_entities)

        logger.info(f"Inserted {len(nodes)} nodes, {len(new_entities)} new entities")

    # ── End-of-day maintenance ──────────────────────────────────

    def end_of_day_maintenance(self):
        """Re-run Leiden and rebalance communities."""
        if self.community_mgr:
            logger.info("Running end-of-day maintenance …")
            self.community_mgr.end_of_day_rebalance()

    # ── Extractors ──────────────────────────────────────────────

    def _create_extractors(self) -> list:
        extraction_llm = self.registry.get_llm("extraction")
        entity_literal = cast(
            Type[Any], Literal.__getitem__(tuple(self.config.entity_types))
        )
        relation_literal = cast(
            Type[Any], Literal.__getitem__(tuple(self.config.relation_types))
        )
        return [
            # 1. Implicit: structural doc → chunk links
            ImplicitPathExtractor(),
            # 2. Schema-constrained: typed entity/relation extraction
            SchemaLLMPathExtractor(
                llm=extraction_llm,
                possible_entities=entity_literal,
                possible_relations=relation_literal,
                max_triplets_per_chunk=30,
                num_workers=4,
            ),
            # 3. GraphRAG: context-aware with descriptions
            GraphRAGExtractor(
                llm=extraction_llm,
                entity_types=self.config.entity_types,
                relation_types=self.config.relation_types,
                max_paths_per_chunk=50,
                num_workers=4,
            ),
        ]

    # ── Neo4j indexes ───────────────────────────────────────────

    def _create_neo4j_indexes(self):
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )

        with driver.session(database=self.config.neo4j_database) as session:
            if self.config.enable_vector_index:
                try:
                    session.run(
                        """
                        CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                        FOR (e:__Entity__)
                        ON (e.embedding)
                        OPTIONS {
                            indexConfig: {
                                `vector.dimensions`: $embedding_dimensions,
                                `vector.similarity_function`: 'cosine'
                            }
                        }
                    """,
                        embedding_dimensions=self.config.embedding_dimensions,
                    )
                except Exception as exc:
                    logger.warning(f"Vector index: {exc}")

            if self.config.enable_fulltext_index:
                try:
                    session.run(
                        """
                        CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
                        FOR (e:__Entity__) ON EACH [e.name, e.description]
                    """
                    )
                    session.run(
                        """
                        CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
                        FOR (c:Chunk) ON EACH [c.text, c.chunk_summary]
                    """
                    )
                except Exception as exc:
                    logger.warning(f"Fulltext index: {exc}")

            try:
                session.run(
                    "CREATE INDEX entity_type_index IF NOT EXISTS "
                    "FOR (e:__Entity__) ON (e.type)"
                )
                session.run(
                    "CREATE INDEX source_doc_index IF NOT EXISTS "
                    "FOR (n:Chunk) ON (n.source_document)"
                )
                session.run(
                    "CREATE INDEX page_number_index IF NOT EXISTS "
                    "FOR (n:Chunk) ON (n.page_number)"
                )
                session.run(
                    "CREATE INDEX community_dirty_index IF NOT EXISTS "
                    "FOR (c:Community) ON (c.is_dirty)"
                )
            except Exception as exc:
                logger.warning(f"Property indexes: {exc}")

        driver.close()

    # ── Helpers ─────────────────────────────────────────────────

    def _get_entity_names_from_store(self) -> set:
        if self.graph_store is None:
            raise RuntimeError("Graph store not initialized.")
        with self.graph_store._driver.session(
            database=self.graph_store._database
        ) as session:
            result = session.run("MATCH (e:__Entity__) RETURN e.name AS name")
            return {r["name"] for r in result}
