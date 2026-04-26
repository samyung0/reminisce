"""
Enhanced Neo4j property graph store with GraphRAG community building.

Follows the cookbook pattern:
  https://developers.llamaindex.ai/python/examples/cookbooks/graphrag_v2/#graphragstore

Extends Neo4jPropertyGraphStore with:
  - build_communities(): hierarchical Leiden + LLM summaries
  - get_community_summaries(): retrieve all community summaries
  - Community node CRUD

Uses graspologic for Leiden (no Neo4j GDS plugin required).
"""

import logging
from typing import Dict, List, Optional

import networkx as nx
from llama_index.core.llms import LLM
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    Neo4j property graph store with GraphRAG community detection.

    Follows the cookbook pattern for community building using
    hierarchical Leiden algorithm (via graspologic).
    """

    def __init__(self, *args, community_llm: Optional[LLM] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.community_llm = community_llm
        self._community_cache: Dict[int, Dict] = {}

    @property
    def _database(self) -> str:
        return getattr(self, "database", "neo4j")

    # ── Community Building (cookbook pattern) ───────────────────

    def build_communities(self, max_levels: int = 3) -> Dict[int, Dict]:
        """
        Full community build pipeline:
          1. Load entity graph from Neo4j
          2. Build networkx graph
          3. Run hierarchical Leiden
          4. Generate community summaries via LLM
          5. Store community nodes in Neo4j

        Returns dict mapping community_id → community info.
        """
        logger.info("Building communities …")

        # Step 1-2: Build networkx graph from Neo4j
        nx_graph = self._build_networkx_graph()
        if nx_graph.number_of_nodes() == 0:
            logger.warning("No entities found for community building")
            return {}

        # Step 3: Hierarchical Leiden
        communities = self._run_hierarchical_leiden(nx_graph, max_levels)
        logger.info(f"Detected {len(communities)} communities")

        # Step 4-5: Generate summaries and store
        self._generate_and_store_communities(communities, nx_graph)

        # Update cache
        self._community_cache = communities
        return communities

    # ── NetworkX graph construction ─────────────────────────────

    def _build_networkx_graph(self) -> nx.Graph:
        """
        Load all Entity nodes and their relationships from Neo4j
        and construct a NetworkX graph for community detection.
        """
        G = nx.Graph()

        with self._driver.session(database=self._database) as session:
            # Add entity nodes
            result = session.run(
                """
                MATCH (e:__Entity__)
                RETURN e.name AS name, e.type AS type,
                       e.description AS description,
                       e.embedding AS embedding
            """
            )
            for record in result:
                name = record["name"]
                G.add_node(
                    name,
                    type=record["type"],
                    description=record.get("description", ""),
                    embedding=record.get("embedding"),
                )

            # Add relationships as edges
            result = session.run(
                """
                MATCH (s:__Entity__)-[r]->(t:__Entity__)
                RETURN s.name AS source, t.name AS target,
                       type(r) AS rel_type
            """
            )
            for record in result:
                G.add_edge(
                    record["source"],
                    record["target"],
                    rel_type=record["rel_type"],
                )

        logger.info(
            f"Built NX graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G

    # ── Hierarchical Leiden ─────────────────────────────────────

    def _run_hierarchical_leiden(self, G: nx.Graph, max_levels: int) -> Dict[int, Dict]:
        """
        Run hierarchical Leiden community detection.

        Returns dict mapping community_id → {
            "level": int,
            "members": List[str],  # entity names
            "parent": Optional[int],
        }
        """
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.error("graspologic not installed. Run: pip install graspologic")
            return {}

        # hierarchical_leiden stops splitting once clusters are below this size.
        max_cluster_size = max(5, G.number_of_nodes() // max(1, max_levels))
        hierarchical_clusters = hierarchical_leiden(
            G,
            max_cluster_size=max_cluster_size,
        )

        communities: Dict[int, Dict] = {}

        for cluster in hierarchical_clusters:
            if cluster.level >= max_levels:
                continue

            cid = cluster.cluster
            level = cluster.level  # 0 = finest, higher = coarser

            if cid not in communities:
                communities[cid] = {
                    "level": level,
                    "members": [],
                    "summary": None,
                    "parent": None,
                    "is_dirty": False,
                }

            # Add the node to this community
            node_name = cluster.node
            communities[cid]["members"].append(node_name)

        # Build parent-child relationships between levels
        self._link_hierarchical_communities(communities)

        return communities

    def _link_hierarchical_communities(self, communities: Dict[int, Dict]):
        """
        For hierarchical Leiden, establish parent-community relationships.
        A level-L community's parent is the level-(L+1) community that
        contains the majority of its members.
        """
        levels = sorted(set(c["level"] for c in communities.values()))

        for level in levels[:-1]:  # Skip the coarsest level
            child_comms = {
                cid: c for cid, c in communities.items() if c["level"] == level
            }
            parent_comms = {
                cid: c for cid, c in communities.items() if c["level"] == level + 1
            }

            for child_id, child in child_comms.items():
                # Find which parent community has the most overlap
                best_parent = None
                best_overlap = 0
                for parent_id, parent in parent_comms.items():
                    overlap = len(set(child["members"]) & set(parent["members"]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent_id

                if best_parent is not None:
                    communities[child_id]["parent"] = best_parent

    # ── Community summarization ─────────────────────────────────

    def _generate_and_store_communities(
        self, communities: Dict[int, Dict], G: nx.Graph
    ):
        """Generate LLM summaries for each community and write to Neo4j."""
        if not self.community_llm:
            logger.warning("No community_llm set; skipping summaries")
            return

        with self._driver.session(database=self._database) as session:
            # Clear old communities
            session.run("MATCH (c:Community) DETACH DELETE c")

            for cid, comm in communities.items():
                # Generate summary
                summary = self._summarize_community(comm, G)
                comm["summary"] = summary

                # Get entity types and relationships within community
                entity_info = self._get_community_entity_info(comm, G)

                # Create Community node
                session.run(
                    """
                    MERGE (c:Community {id: $cid})
                    SET c.level = $level,
                        c.summary = $summary,
                        c.member_count = $member_count,
                        c.is_dirty = False,
                        c.entity_types = $entity_types,
                        c.rel_types = $rel_types
                """,
                    cid=cid,
                    level=comm["level"],
                    summary=summary,
                    member_count=len(comm["members"]),
                    entity_types=entity_info["entity_types"],
                    rel_types=entity_info["rel_types"],
                )

                # Link entities to community
                for entity_name in comm["members"]:
                    session.run(
                        """
                        MATCH (c:Community {id: $cid})
                        MATCH (e:__Entity__ {name: $name})
                        MERGE (e)-[:MEMBER_OF]->(c)
                    """,
                        cid=cid,
                        name=entity_name,
                    )

                # Link to parent community
                if comm.get("parent") is not None:
                    session.run(
                        """
                        MATCH (c:Community {id: $cid})
                        MATCH (p:Community {id: $parent_id})
                        MERGE (c)-[:CHILD_OF]->(p)
                    """,
                        cid=cid,
                        parent_id=comm["parent"],
                    )

        logger.info(f"Stored {len(communities)} community summaries")

    def _summarize_community(self, comm: Dict, G: nx.Graph) -> str:
        """Generate LLM summary for a community."""
        members = comm["members"]
        if not members:
            return "Empty community."

        # Gather entity descriptions and inter-relationships
        entity_details = []
        for name in members[:50]:  # Cap to avoid token overflow
            node_data = G.nodes.get(name, {})
            desc = node_data.get("description", "")
            etype = node_data.get("type", "UNKNOWN")
            entity_details.append(
                f"- {name} ({etype}): {desc}" if desc else f"- {name} ({etype})"
            )

        # Get intra-community relationships
        intra_rels = []
        for u, v, data in G.subgraph([m for m in members if m in G]).edges(data=True):
            intra_rels.append(f"  {u} --[{data.get('rel_type', 'RELATED_TO')}]--> {v}")

        rel_text = "\n".join(intra_rels[:30])
        entity_text = "\n".join(entity_details)

        prompt = f"""You are an assistant that summarizes community profiles for a knowledge graph.

ENTITY MEMBERS:
{entity_text}

RELATIONSHIPS WITHIN COMMUNITY:
{rel_text}

Write a comprehensive summary of this community in 3-5 sentences. Focus on:
1. What unites these entities
2. Key relationships and interactions
3. The domain or topic this community represents
4. Any notable patterns or insights

Community summary:"""

        try:
            if self.community_llm is None:
                raise RuntimeError("community_llm is not configured")
            resp = self.community_llm.complete(prompt)
            return resp.text.strip()
        except Exception as exc:
            logger.warning(f"Community summary failed for {comm}: {exc}")
            return f"Community of {len(members)} entities: {', '.join(members[:10])}"

    def _get_community_entity_info(self, comm: Dict, G: nx.Graph) -> Dict:
        """Get entity types and relationship types within a community."""
        entity_types = set()
        rel_types = set()
        members = [m for m in comm["members"] if m in G]

        for name in members:
            entity_types.add(G.nodes[name].get("type", "UNKNOWN"))

        for u, v, data in G.subgraph(members).edges(data=True):
            rel_types.add(data.get("rel_type", "UNKNOWN"))

        return {
            "entity_types": list(entity_types),
            "rel_types": list(rel_types),
        }

    # ── Query helpers ───────────────────────────────────────────

    def get_community_summaries(self) -> Dict[int, Dict]:
        """Get all community summaries from Neo4j."""
        summaries = {}
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (c:Community)
                RETURN c.id AS id, c.summary AS summary,
                       c.level AS level, c.is_dirty AS is_dirty,
                       c.member_count AS member_count
            """
            )
            for record in result:
                summaries[record["id"]] = {
                    "summary": record["summary"],
                    "level": record["level"],
                    "is_dirty": record.get("is_dirty", False),
                    "member_count": record.get("member_count", 0),
                }
        return summaries

    def get_entity_communities(self, entity_name: str) -> List[Dict]:
        """Get all communities an entity belongs to."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (e:__Entity__ {name: $name})-[:MEMBER_OF]->(c:Community)
                RETURN c.id AS id, c.summary AS summary,
                       c.level AS level, c.is_dirty AS is_dirty
            """,
                name=entity_name,
            )
            return [dict(r) for r in result]

    def get_community_members(self, community_id: int) -> List[Dict]:
        """Get all entities in a community."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (e:__Entity__)-[:MEMBER_OF]->(c:Community {id: $cid})
                RETURN e.name AS name, e.type AS type,
                       e.description AS description
            """,
                cid=community_id,
            )
            return [dict(r) for r in result]

    def update_community_summary(self, community_id: int, summary: str):
        """Update a single community's summary in Neo4j."""
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (c:Community {id: $cid})
                SET c.summary = $summary, c.is_dirty = False
            """,
                cid=community_id,
                summary=summary,
            )

    def mark_community_dirty(self, community_id: int):
        """Mark a community (and its parents) as needing summary rebuild."""
        with self._driver.session(database=self._database) as session:
            # Mark the community itself
            session.run(
                """
                MATCH (c:Community {id: $cid})
                SET c.is_dirty = True
            """,
                cid=community_id,
            )
            # Mark parent communities
            session.run(
                """
                MATCH (c:Community {id: $cid})-[:CHILD_OF*]->(parent:Community)
                SET parent.is_dirty = True
            """,
                cid=community_id,
            )

    def get_dirty_communities(self) -> List[int]:
        """Get IDs of all communities with is_dirty = True."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (c:Community {is_dirty: True})
                RETURN c.id AS id
            """
            )
            return [r["id"] for r in result]

    def attach_entities_to_existing_communities(
        self, entity_names: List[str], max_communities: int = 3
    ) -> List[int]:
        """
        Attach new entities to nearby existing communities.

        This keeps lazy community summaries useful between full Leiden runs.
        The end-of-day rebalance still owns final community membership.
        """
        if not entity_names:
            return []

        touched: set[int] = set()
        with self._driver.session(database=self._database) as session:
            for name in entity_names:
                result = session.run(
                    """
                    MATCH (e:__Entity__ {name: $name})--(n:__Entity__)-[:MEMBER_OF]->(c:Community)
                    RETURN c.id AS id, count(*) AS score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    name=name,
                    limit=max_communities,
                )
                ids = [r["id"] for r in result]
                for cid in ids:
                    session.run(
                        """
                        MATCH (e:__Entity__ {name: $name})
                        MATCH (c:Community {id: $cid})
                        MERGE (e)-[:MEMBER_OF]->(c)
                        """,
                        name=name,
                        cid=cid,
                    )
                    touched.add(cid)

        return list(touched)

    def get_total_entity_count(self) -> int:
        """Total number of entity nodes in the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH (e:__Entity__) RETURN count(e) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0
