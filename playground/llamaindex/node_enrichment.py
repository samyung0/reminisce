"""
Node enrichment — runs BEFORE graph extraction:
  1. Abbreviation dictionary building + expansion
  2. Contextual summarization
  3. Embedding-text construction

Uses the LLM registry so you can swap the enrichment model freely.
"""

import json
import logging
import re
from typing import Dict, List, Tuple

from llama_index.core.llms import LLM
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class NodeEnricher:
    def __init__(self, llm: LLM, config):
        self.llm = llm
        self.config = config
        self._abbrev_cache: Dict[str, str] = {}

    def enrich_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        logger.info("Starting node enrichment …")

        if self.config.enable_abbreviation_expansion:
            abbrev_dict = self._build_abbreviation_dictionary(nodes)
            self._abbrev_cache.update(abbrev_dict)
            logger.info(f"Discovered {len(abbrev_dict)} abbreviation mappings")

        for i, node in enumerate(nodes):
            neighbors = self._get_neighbor_context(nodes, i)

            if self.config.enable_summarization:
                summary = self._generate_contextual_summary(node, neighbors)
                node.metadata["chunk_summary"] = summary

            if self.config.enable_abbreviation_expansion:
                expanded_text, resolved = self._expand_abbreviations(node.text)
                node.metadata["original_text"] = node.text
                node.metadata["resolved_entities"] = resolved
                node.text = expanded_text

            node.metadata["embed_text"] = self._build_embedding_text(node)

        logger.info(f"Enriched {len(nodes)} nodes")
        return nodes

    # ── Abbreviation dictionary ─────────────────────────────────

    def _build_abbreviation_dictionary(self, nodes: List[TextNode]) -> Dict[str, str]:
        all_text = "\n".join(n.text for n in nodes)

        prompt = f"""Extract all abbreviations/acronyms and their full forms from the text below.
Only include abbreviations you are confident about.
Return a JSON object mapping abbreviation → full form. If none found, return {{}}.

Example:
{{"API": "Application Programming Interface", "LLM": "Large Language Model", "RAG": "Retrieval-Augmented Generation"}}

TEXT (first 8000 chars):
{all_text[:8000]}

JSON:"""

        abbrev_dict: Dict[str, str] = {}
        try:
            resp = self.llm.complete(prompt)
            json_str = self._extract_json(resp.text)
            if json_str:
                abbrev_dict = json.loads(json_str)
        except Exception as exc:
            logger.warning(f"LLM abbreviation extraction failed: {exc}")

        # Regex supplement
        for abbr, full in self._regex_abbreviation_extraction(all_text).items():
            if abbr not in abbrev_dict:
                abbrev_dict[abbr] = full

        return abbrev_dict

    @staticmethod
    def _regex_abbreviation_extraction(text: str) -> Dict[str, str]:
        abbrevs: Dict[str, str] = {}
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(([A-Z]{2,})\)'
        for match in re.finditer(pattern, text):
            abbrevs[match.group(2)] = match.group(1).strip()
        return abbrevs

    # ── Abbreviation expansion ──────────────────────────────────

    def _expand_abbreviations(self, text: str) -> Tuple[str, List[Dict]]:
        resolved: List[Dict] = []
        expanded = text
        for abbrev, full_form in self._abbrev_cache.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, expanded):
                expanded = re.sub(
                    pattern, f"{full_form} ({abbrev})", expanded, count=1,
                )
                resolved.append({"abbreviation": abbrev, "full_form": full_form})
        return expanded, resolved

    # ── Contextual summarization ────────────────────────────────

    def _get_neighbor_context(self, nodes, index):
        w = self.config.summary_context_window
        return nodes[max(0, index - w): min(len(nodes), index + w + 1)]

    def _generate_contextual_summary(self, node, neighbors) -> str:
        context = "\n---\n".join(
            f"[p.{n.metadata.get('page_number', '?')}]: {n.text[:400]}"
            for n in neighbors
        )
        prompt = f"""Summarize the TEXT CHUNK below in 2-3 concise sentences.
Surrounding context is for disambiguation only — summarize the CHUNK.

CONTEXT:
{context}

TEXT CHUNK:
{node.text}

SUMMARY:"""
        try:
            resp = self.llm.complete(prompt)
            return resp.text.strip()
        except Exception as exc:
            logger.warning(f"Chunk summary failed: {exc}")
            return node.text[:500].strip()

    def _build_embedding_text(self, node: TextNode) -> str:
        """Build an enriched semantic string for chunk embeddings."""
        parts = []
        headings = node.metadata.get("headings") or []
        if headings:
            parts.append("Headings: " + " > ".join(str(h) for h in headings))

        summary = node.metadata.get("chunk_summary")
        if summary:
            parts.append(f"Chunk summary: {summary}")

        resolved = node.metadata.get("resolved_entities") or []
        if resolved:
            rendered = ", ".join(
                f"{item.get('abbreviation')} = {item.get('full_form')}"
                for item in resolved
            )
            parts.append(f"Resolved abbreviations: {rendered}")

        source = node.metadata.get("source_document")
        page = node.metadata.get("page_number")
        if source:
            parts.append(f"Source: {source}, page {page}")

        parts.append(f"Text: {node.text}")
        return "\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        plain = re.search(r"\{.*\}", text, re.DOTALL)
        return plain.group(0) if plain else ""