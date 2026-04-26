"""
Dual-mode retrieval engine:

  SIMPLE  (DeepSeek V4 Flash):
    - Local search: vector similarity + Cypher neighborhood traversal
    - Global search: community profile lookup (GraphRAGQueryEngine pattern)
    - Single-pass synthesis

  DETAILED (DeepSeek V4 Pro + agentic loop):
    - Same local + global search
    - Agentic loop: evaluate sufficiency → drill deeper → iterate
    - Can fetch in-depth details, traverse more graph hops
    - Multi-pass synthesis with source verification

Both modes trace results back to source document + bbox.
"""

import json
import logging
import re
from typing import Any, Dict, List, LiteralString, Optional, cast

from neo4j import Query
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from community_manager import CommunityManager
from graphrag_store import GraphRAGStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Search result data classes
# ═══════════════════════════════════════════════════════════════


class LocalSearchResult:
    """Results from local (entity-centric) search."""

    def __init__(self):
        self.entities: List[Dict] = []
        self.chunks: List[Dict] = []
        self.relationships: List[Dict] = []


class GlobalSearchResult:
    """Results from global (community-centric) search."""

    def __init__(self):
        self.communities: List[Dict] = []


# ═══════════════════════════════════════════════════════════════
#  LOCAL SEARCH — vector + Cypher
# ═══════════════════════════════════════════════════════════════


class LocalSearch:
    """
    Entity-centric local search combining:
      1. Vector similarity on entity embeddings
      2. Cypher graph traversal to expand neighborhood
      3. Source chunk retrieval with bbox
    """

    def __init__(
        self,
        store: GraphRAGStore,
        embed_model: BaseEmbedding,
        llm: LLM,
        top_k: int = 10,
    ):
        self.store = store
        self.embed_model = embed_model
        self.llm = llm
        self.top_k = top_k

    def search(self, query: str, depth: int = 2) -> LocalSearchResult:
        result = LocalSearchResult()

        # Step 1: Vector search for seed entities
        query_embedding = self.embed_model.get_query_embedding(query)
        seed_entities = self._vector_entity_search(query_embedding)
        result.entities = seed_entities

        # Step 2: Graph traversal from seed entities
        if seed_entities:
            expanded = self._expand_neighborhood(
                [e["name"] for e in seed_entities], depth
            )
            result.entities.extend(expanded)
            result.relationships = self._get_relationships(
                [e["name"] for e in result.entities]
            )

        # Step 3: Fetch source chunks with bbox
        entity_names = list({e["name"] for e in result.entities})
        result.chunks = self._fetch_source_chunks(entity_names)

        return result

    def _vector_entity_search(self, query_embedding: List[float]) -> List[Dict]:
        """Vector similarity search on entity embeddings."""
        with self.store._driver.session(database=self.store._database) as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'entity_embedding', $top_k, $embedding
                    )
                    YIELD node, score
                    RETURN node.name AS name,
                           node.type AS type,
                           node.description AS description,
                           score
                    ORDER BY score DESC
                """,
                    top_k=self.top_k,
                    embedding=query_embedding,
                )
                return [dict(r) for r in result]
            except Exception as exc:
                logger.warning(f"Vector entity search failed: {exc}")
                return []

    def _expand_neighborhood(self, seed_names: List[str], depth: int) -> List[Dict]:
        """BFS expansion from seed entities via Cypher."""
        query = cast(
            LiteralString,
            """
                MATCH (seed:__Entity__)-[r*1..%d]-(neighbor:__Entity__)
                WHERE seed.name IN $names
                RETURN DISTINCT neighbor.name AS name,
                       neighbor.type AS type,
                       neighbor.description AS description
                LIMIT 50
            """
            % depth,
        )
        with self.store._driver.session(database=self.store._database) as session:
            result = session.run(
                Query(query),
                names=seed_names,
            )
            return [dict(r) for r in result]

    def _get_relationships(self, entity_names: List[str]) -> List[Dict]:
        """Get relationships between a set of entities."""
        with self.store._driver.session(database=self.store._database) as session:
            result = session.run(
                """
                MATCH (s:__Entity__)-[r]->(t:__Entity__)
                WHERE s.name IN $names AND t.name IN $names
                RETURN s.name AS source, type(r) AS rel_type,
                       t.name AS target
                LIMIT 100
            """,
                names=entity_names,
            )
            return [dict(r) for r in result]

    def _fetch_source_chunks(self, entity_names: List[str]) -> List[Dict]:
        """Trace entities back to source chunks with bbox."""
        with self.store._driver.session(database=self.store._database) as session:
            result = session.run(
                """
                MATCH (e:__Entity__)<-[:MENTIONS]-(chunk:Chunk)
                WHERE e.name IN $names
                RETURN DISTINCT chunk.text AS text,
                       chunk.source_document AS source_document,
                       chunk.page_number AS page_number,
                       chunk.bbox AS bbox,
                       chunk.chunk_summary AS chunk_summary
                ORDER BY chunk.source_document, chunk.page_number
                LIMIT 30
            """,
                names=entity_names,
            )
            return [dict(r) for r in result]


# ═══════════════════════════════════════════════════════════════
#  GLOBAL SEARCH — community profiles (GraphRAGQueryEngine pattern)
# ═══════════════════════════════════════════════════════════════


class GlobalSearch:
    """
    Community-centric global search following the GraphRAG cookbook's
    GraphRAGQueryEngine pattern.

    1. Find relevant communities via entity vector search
    2. Get community summaries (rebuilding dirty ones on demand)
    3. Use community summaries as macro-level context
    """

    def __init__(
        self,
        store: GraphRAGStore,
        community_manager: CommunityManager,
        embed_model: BaseEmbedding,
        llm: LLM,
        top_k: int = 5,
    ):
        self.store = store
        self.community_mgr = community_manager
        self.embed_model = embed_model
        self.llm = llm
        self.top_k = top_k

    def search(self, query: str) -> GlobalSearchResult:
        result = GlobalSearchResult()

        # Step 1: Find relevant entities via vector search
        query_embedding = self.embed_model.get_query_embedding(query)
        seed_entities = self._vector_search(query_embedding)

        if not seed_entities:
            return result

        # Step 2: Find communities these entities belong to
        community_ids = set()
        for entity in seed_entities:
            comms = self.store.get_entity_communities(entity["name"])
            for c in comms:
                community_ids.add(c["id"])

        # Step 3: Get summaries (rebuilding dirty ones on demand)
        for cid in list(community_ids)[: self.top_k]:
            summary = self.community_mgr.ensure_community_summary(cid)
            members = self.store.get_community_members(cid)
            comm_info = self.store.get_community_summaries().get(cid, {})

            result.communities.append(
                {
                    "id": cid,
                    "summary": summary,
                    "level": comm_info.get("level", 0),
                    "member_count": comm_info.get("member_count", len(members)),
                    "key_members": [
                        {"name": m["name"], "type": m["type"]} for m in members[:20]
                    ],
                }
            )

        return result

    def _vector_search(self, query_embedding: List[float]) -> List[Dict]:
        with self.store._driver.session(database=self.store._database) as session:
            try:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'entity_embedding', $top_k, $embedding
                    )
                    YIELD node, score
                    RETURN node.name AS name, node.type AS type, score
                    ORDER BY score DESC
                """,
                    top_k=self.top_k,
                    embedding=query_embedding,
                )
                return [dict(r) for r in result]
            except Exception:
                return []


# ═══════════════════════════════════════════════════════════════
#  SIMPLE MODE — single-pass with DeepSeek V4 Flash
# ═══════════════════════════════════════════════════════════════


class SimpleQueryEngine:
    """
    Simple retrieval: local + global search, single-pass synthesis.
    Uses DeepSeek V4 Flash for speed.
    """

    def __init__(
        self,
        local_search: LocalSearch,
        global_search: GlobalSearch,
        llm: LLM,
    ):
        self.local = local_search
        self.global_ = global_search
        self.llm = llm

    def query(self, query_str: str) -> Dict[str, Any]:
        # Gather evidence
        local_result = self.local.search(query_str)
        global_result = self.global_.search(query_str)

        # Build context
        context = self._build_context(local_result, global_result)

        # Single-pass synthesis
        answer = self._synthesize(query_str, context)

        return {
            "answer": answer,
            "local_entities": local_result.entities[:10],
            "local_chunks": local_result.chunks[:10],
            "global_communities": global_result.communities[:5],
            "mode": "simple",
        }

    def _build_context(
        self,
        local: LocalSearchResult,
        global_: GlobalSearchResult,
    ) -> str:
        parts = []

        # Local context
        if local.entities:
            parts.append("## Relevant Entities")
            for e in local.entities[:15]:
                parts.append(
                    f"- **{e['name']}** ({e.get('type', '?')}): "
                    f"{e.get('description', '')}"
                )

        if local.relationships:
            parts.append("\n## Key Relationships")
            for r in local.relationships[:20]:
                parts.append(f"- {r['source']} →[{r['rel_type']}]→ {r['target']}")

        if local.chunks:
            parts.append("\n## Source Passages")
            for c in local.chunks[:10]:
                src = c.get("source_document", "?")
                pg = c.get("page_number", "?")
                parts.append(f"- [📄 {src}, p.{pg}]: {c['text'][:300]}")

        # Global context
        if global_.communities:
            parts.append("\n## Community Context")
            for comm in global_.communities[:5]:
                members = ", ".join(m["name"] for m in comm.get("key_members", [])[:10])
                parts.append(
                    f"- **Community L{comm['level']}** "
                    f"({comm.get('member_count', 0)} members): "
                    f"{comm['summary']}\n"
                    f"  Key members: {members}"
                )

        return "\n".join(parts)

    def _synthesize(self, query: str, context: str) -> str:
        prompt = f"""Answer the question based on the provided context.
Cite sources with document name, page number, and location when possible.
Be concise but thorough.

CONTEXT:
{context}

QUESTION: {query}

Answer:"""
        try:
            return self.llm.complete(prompt).text.strip()
        except Exception as exc:
            logger.error(f"Synthesis failed: {exc}")
            return f"Error generating answer: {exc}"


# ═══════════════════════════════════════════════════════════════
#  DETAILED MODE — agentic loop with DeepSeek V4 Pro
# ═══════════════════════════════════════════════════════════════


class DetailedQueryEngine:
    """
    Detailed retrieval with agentic loop.
    Uses DeepSeek V4 Pro for deeper reasoning.

    The agent can:
      1. Search (local + global)
      2. Evaluate whether information is sufficient
      3. Identify gaps and drill deeper
      4. Iterate until sufficient or max iterations reached
      5. Synthesize comprehensive answer with verified sources
    """

    def __init__(
        self,
        local_search: LocalSearch,
        global_search: GlobalSearch,
        llm: LLM,  # DeepSeek V4 Pro
        embed_model: BaseEmbedding,
        store: GraphRAGStore,
        max_iterations: int = 5,
    ):
        self.local = local_search
        self.global_ = global_search
        self.llm = llm
        self.embed_model = embed_model
        self.store = store
        self.max_iterations = max_iterations

    def query(self, query_str: str) -> Dict[str, Any]:
        # Initial search
        context_parts: List[str] = []
        all_entities: List[Dict] = []
        all_chunks: List[Dict] = []
        all_communities: List[Dict] = []
        explored_entities: set = set()
        iteration_log: List[Dict] = []

        # Round 1: broad search
        local_result = self.local.search(query_str, depth=2)
        global_result = self.global_.search(query_str)

        all_entities.extend(local_result.entities)
        all_chunks.extend(local_result.chunks)
        all_communities.extend(global_result.communities)
        context_parts.append(
            self._format_local_context(local_result)
            + self._format_global_context(global_result)
        )

        iteration_log.append(
            {
                "iteration": 1,
                "action": "initial_search",
                "entities_found": len(local_result.entities),
                "communities_found": len(global_result.communities),
            }
        )

        # Agentic loop
        for i in range(2, self.max_iterations + 1):
            # Evaluate sufficiency
            evaluation = self._evaluate_sufficiency(query_str, "\n".join(context_parts))

            iteration_log.append(
                {
                    "iteration": i,
                    "action": "evaluation",
                    "is_sufficient": evaluation["is_sufficient"],
                    "gaps": evaluation.get("gaps", []),
                }
            )

            if evaluation["is_sufficient"]:
                break

            # Drill into gaps
            for gap in evaluation.get("gaps", []):
                gap_result = self._drill_into_gap(gap, query_str, explored_entities)
                if gap_result:
                    context_parts.append(gap_result["context"])
                    all_entities.extend(gap_result.get("entities", []))
                    all_chunks.extend(gap_result.get("chunks", []))

            iteration_log.append(
                {
                    "iteration": i,
                    "action": "drill_down",
                    "gaps_addressed": len(evaluation.get("gaps", [])),
                }
            )

        # Final synthesis
        answer = self._synthesize_final(
            query_str, "\n".join(context_parts), iteration_log
        )

        # Deduplicate
        unique_entities = self._deduplicate(all_entities, "name")
        unique_chunks = self._deduplicate(all_chunks, "text")

        return {
            "answer": answer,
            "entities": unique_entities[:20],
            "source_chunks": unique_chunks[:15],
            "communities": all_communities[:5],
            "iterations": len(iteration_log),
            "iteration_log": iteration_log,
            "mode": "detailed",
        }

    # ── Evaluation ──────────────────────────────────────────────

    def _evaluate_sufficiency(self, query: str, context: str) -> Dict:
        """
        Ask the LLM whether the current context is sufficient
        to answer the query, and if not, what's missing.
        """
        prompt = f"""You are evaluating whether the gathered information is sufficient to answer a question.

QUESTION: {query}

GATHERED INFORMATION:
{context[:6000]}

Evaluate:
1. Is the information sufficient to provide a comprehensive answer? (true/false)
2. If not, list specific gaps or missing information that needs to be found.

Respond in this exact JSON format:
{{
  "is_sufficient": true/false,
  "gaps": ["gap1", "gap2", ...],
  "reasoning": "brief explanation"
}}

JSON:"""

        try:
            resp = self.llm.complete(prompt)
            # Extract JSON
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", resp.text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as exc:
            logger.warning(f"Sufficiency evaluation failed: {exc}")

        # Default: assume sufficient after failed eval
        return {"is_sufficient": True, "gaps": [], "reasoning": "Evaluation failed"}

    # ── Gap drilling ────────────────────────────────────────────

    def _drill_into_gap(
        self, gap: str, original_query: str, explored: set
    ) -> Optional[Dict]:
        """
        Drill deeper into a specific gap by:
          1. Searching for entities related to the gap
          2. Traversing deeper (more hops)
          3. Fetching more source chunks
        """
        # Search for gap-related entities
        local_result = self.local.search(
            f"{original_query} {gap}", depth=3  # Deeper traversal
        )

        if not local_result.entities and not local_result.chunks:
            return None

        # Mark explored entities
        for e in local_result.entities:
            explored.add(e["name"])

        context = self._format_local_context(local_result)
        return {
            "context": f"\n### Deep Dive: {gap}\n{context}",
            "entities": local_result.entities,
            "chunks": local_result.chunks,
        }

    # ── Final synthesis ─────────────────────────────────────────

    def _synthesize_final(
        self, query: str, context: str, iteration_log: List[Dict]
    ) -> str:
        """Comprehensive multi-source synthesis with source verification."""
        prompt = f"""You are a research assistant providing a comprehensive, detailed answer.

You have gone through {len(iteration_log)} rounds of searching and evaluation.
Provide a thorough answer with:
1. Direct answer to the question
2. Supporting evidence with specific source citations (document, page, bbox)
3. Nuances and caveats
4. Related context that may be relevant

Always cite sources in format: [📄 Document Name, p.X]

GATHERED EVIDENCE:
{context[:12000]}

QUESTION: {query}

Detailed Answer:"""

        try:
            return self.llm.complete(prompt).text.strip()
        except Exception as exc:
            logger.error(f"Final synthesis failed: {exc}")
            return f"Error: {exc}"

    # ── Formatting helpers ──────────────────────────────────────

    def _format_local_context(self, local: LocalSearchResult) -> str:
        parts = []
        if local.entities:
            parts.append("### Entities")
            for e in local.entities[:15]:
                parts.append(
                    f"- **{e['name']}** ({e.get('type', '?')}): "
                    f"{e.get('description', '')}"
                )
        if local.relationships:
            parts.append("### Relationships")
            for r in local.relationships[:20]:
                parts.append(f"- {r['source']} →[{r['rel_type']}]→ {r['target']}")
        if local.chunks:
            parts.append("### Source Passages")
            for c in local.chunks[:10]:
                parts.append(
                    f"- [📄 {c.get('source_document', '?')}, "
                    f"p.{c.get('page_number', '?')}]: {c['text'][:300]}"
                )
        return "\n".join(parts)

    def _format_global_context(self, global_: GlobalSearchResult) -> str:
        parts = []
        if global_.communities:
            parts.append("\n### Community Profiles")
            for comm in global_.communities:
                members = ", ".join(m["name"] for m in comm.get("key_members", [])[:10])
                parts.append(
                    f"- **Community L{comm['level']}** "
                    f"({comm.get('member_count', 0)} members): "
                    f"{comm['summary']}\n"
                    f"  Members: {members}"
                )
        return "\n".join(parts)

    @staticmethod
    def _deduplicate(items: List[Dict], key: str) -> List[Dict]:
        seen = set()
        result = []
        for item in items:
            k = item.get(key, "")
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result


# ═══════════════════════════════════════════════════════════════
#  UNIFIED RETRIEVAL ENGINE
# ═══════════════════════════════════════════════════════════════


class RetrievalEngine:
    """
    Unified entry point for both simple and detailed retrieval.
    Delegates to the appropriate engine based on mode.
    """

    def __init__(
        self,
        store: GraphRAGStore,
        community_manager: CommunityManager,
        llm_registry,  # LLMRegistry
        config,
    ):
        self.store = store
        self.community_mgr = community_manager
        self.config = config

        embed_model = llm_registry.get_embed_model()

        # Flash LLM for simple mode + local/global search
        flash_llm = llm_registry.get_llm("retrieval_simple")

        # Pro LLM for detailed mode
        pro_llm = llm_registry.get_llm("retrieval_detailed")

        # Shared search components (use Flash for search, Pro for reasoning)
        self._local_search = LocalSearch(
            store=store,
            embed_model=embed_model,
            llm=flash_llm,
            top_k=config.retrieval_top_k,
        )
        self._global_search = GlobalSearch(
            store=store,
            community_manager=community_manager,
            embed_model=embed_model,
            llm=flash_llm,
            top_k=config.retrieval_community_top_k,
        )

        self._simple_engine = SimpleQueryEngine(
            local_search=self._local_search,
            global_search=self._global_search,
            llm=flash_llm,
        )
        self._detailed_engine = DetailedQueryEngine(
            local_search=self._local_search,
            global_search=self._global_search,
            llm=pro_llm,
            embed_model=embed_model,
            store=store,
            max_iterations=config.detailed_max_agent_iterations,
        )

    def query(
        self,
        query_str: str,
        mode: str = "simple",
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph.

        Args:
            query_str: Natural language question
            mode: "simple" (Flash, single-pass) or "detailed" (Pro, agentic)
        """
        if mode == "simple":
            return self._simple_engine.query(query_str)
        elif mode == "detailed":
            return self._detailed_engine.query(query_str)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'simple' or 'detailed'.")
