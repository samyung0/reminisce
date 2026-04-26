"""
Community lifecycle manager with aggressive / lazy rebuild strategies.

Aggressive mode (total entities < threshold):
  - Rebuild ALL community summaries on every insert.
  - Re-run Leiden from scratch each time.

Lazy mode (total entities >= threshold):
  - Mark affected communities as dirty (is_dirty flag).
  - On retrieval: rebuild dirty community summaries on-demand.
  - End of day: re-run Leiden to rebalance the full hierarchy.

Uses DeepSeek V4 Flash for community summaries (via LLM registry).
"""

import logging
from typing import Dict, List, Set

from llama_index.core.llms import LLM

from graphrag_store import GraphRAGStore

logger = logging.getLogger(__name__)


class CommunityManager:
    """
    Manages community lifecycle with two strategies:
      - Aggressive (below threshold): full rebuild on every insert
      - Lazy (above threshold): dirty flags + on-demand rebuild
    """

    def __init__(
        self,
        store: GraphRAGStore,
        summary_llm: LLM,
        aggressive_threshold: int = 500,
        max_levels: int = 3,
    ):
        self.store = store
        self.summary_llm = summary_llm
        self.aggressive_threshold = aggressive_threshold
        self.max_levels = max_levels
        self._summary_cache: Dict[int, str] = {}

    # ── Main lifecycle hook ─────────────────────────────────────

    def on_nodes_inserted(self, new_entity_names: List[str]):
        """
        Called after new entities are inserted into the graph.
        Decides whether to do aggressive or lazy community update.
        """
        total = self.store.get_total_entity_count()
        logger.info(
            f"on_nodes_inserted: {len(new_entity_names)} new entities, "
            f"total now {total} (threshold={self.aggressive_threshold})"
        )

        if total < self.aggressive_threshold:
            self._aggressive_rebuild()
        else:
            self._lazy_mark_dirty(new_entity_names)

    # ── Aggressive strategy ─────────────────────────────────────

    def _aggressive_rebuild(self):
        """Full Leiden + full summary rebuild. Used when graph is small."""
        logger.info("AGGRESSIVE: full community rebuild")
        self.store.community_llm = self.summary_llm
        self.store.build_communities(max_levels=self.max_levels)
        self._refresh_summary_cache()

    # ── Lazy strategy ───────────────────────────────────────────

    def _lazy_mark_dirty(self, new_entity_names: List[str]):
        """
        Find communities that contain new entities and mark them
        (and their parents) as dirty. Summaries will be rebuilt
        on-demand during retrieval.
        """
        dirty_ids: Set[int] = set()

        attached_ids = self.store.attach_entities_to_existing_communities(
            new_entity_names
        )
        dirty_ids.update(attached_ids)

        for name in new_entity_names:
            communities = self.store.get_entity_communities(name)
            for comm in communities:
                dirty_ids.add(comm["id"])

        if new_entity_names and not dirty_ids:
            logger.info("LAZY: no nearby communities; running full rebuild")
            self._aggressive_rebuild()
            return

        for cid in dirty_ids:
            self.store.mark_community_dirty(cid)
            logger.debug(f"Marked community {cid} as dirty")

        # Invalidate cache for dirty communities
        for cid in dirty_ids:
            self._summary_cache.pop(cid, None)

        logger.info(f"LAZY: marked {len(dirty_ids)} communities as dirty")

    # ── On-demand dirty rebuild (called during retrieval) ───────

    def ensure_community_summary(self, community_id: int) -> str:
        """
        Get a community summary, rebuilding it first if dirty.
        This is the lazy evaluation point.
        """
        # Check cache first
        if community_id in self._summary_cache:
            return self._summary_cache[community_id]

        # Get from store
        summaries = self.store.get_community_summaries()
        comm_info = summaries.get(community_id)
        if comm_info is None:
            return "Community not found."

        # If dirty, rebuild on-demand
        if comm_info.get("is_dirty", False):
            logger.info(f"Rebuilding dirty community {community_id}")
            summary = self._rebuild_single_summary(community_id)
            self.store.update_community_summary(community_id, summary)
            self._summary_cache[community_id] = summary
            return summary

        # Clean — cache and return
        self._summary_cache[community_id] = comm_info["summary"]
        return comm_info["summary"]

    def ensure_all_dirty_summaries(self) -> int:
        """
        Rebuild summaries for ALL dirty communities.
        Called at retrieval time to ensure global search has
        up-to-date community profiles.

        Returns number of summaries rebuilt.
        """
        dirty_ids = self.store.get_dirty_communities()
        if not dirty_ids:
            return 0

        logger.info(f"Rebuilding {len(dirty_ids)} dirty community summaries")
        for cid in dirty_ids:
            summary = self._rebuild_single_summary(cid)
            self.store.update_community_summary(cid, summary)
            self._summary_cache[cid] = summary

        return len(dirty_ids)

    def _rebuild_single_summary(self, community_id: int) -> str:
        """Rebuild one community's summary using the LLM."""
        members = self.store.get_community_members(community_id)
        if not members:
            return "Empty community."

        member_details = []
        for m in members[:50]:
            desc = m.get("description", "")
            if desc:
                member_details.append(f"- {m['name']} ({m['type']}): {desc[:200]}")
            else:
                member_details.append(f"- {m['name']} ({m['type']})")

        prompt = f"""Update the summary for this community of related entities.
Focus on what unites them and key relationships.

MEMBERS:
{chr(10).join(member_details)}

Updated community summary (3-5 sentences):"""

        try:
            resp = self.summary_llm.complete(prompt)
            return resp.text.strip()
        except Exception as exc:
            logger.warning(
                f"Summary rebuild failed for community {community_id}: {exc}"
            )
            return f"Community of {len(members)} entities: {', '.join(m['name'] for m in members[:10])}"

    # ── End-of-day rebalance ────────────────────────────────────

    def end_of_day_rebalance(self):
        """
        Re-run full Leiden algorithm to rebalance communities.
        Should be called as a scheduled job (e.g., daily).
        """
        logger.info(
            f"END-OF-DAY: full Leiden rebalance "
            f"(total entities: {self.store.get_total_entity_count()})"
        )
        self._aggressive_rebuild()
        logger.info("End-of-day rebalance complete")

    # ── Cache management ────────────────────────────────────────

    def _refresh_summary_cache(self):
        """Load all community summaries into cache."""
        summaries = self.store.get_community_summaries()
        self._summary_cache = {
            cid: info["summary"]
            for cid, info in summaries.items()
            if info.get("summary")
        }

    def get_all_summaries(self) -> Dict[int, str]:
        """Get all community summaries, ensuring dirty ones are rebuilt."""
        self.ensure_all_dirty_summaries()
        self._refresh_summary_cache()
        return dict(self._summary_cache)
