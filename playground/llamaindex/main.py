"""
Entry point — demonstrates the full pipeline with dual retrieval modes.
"""

import logging
import os

from dotenv import load_dotenv

from config import PipelineConfig
from pipeline import PropertyGraphPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-35s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    config = PipelineConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    pipeline = PropertyGraphPipeline(config)

    # ── Build ───────────────────────────────────────────────────
    documents = [
        "data/technical_report.pdf",
        "data/api_documentation.pdf",
    ]
    pipeline.build(documents)
    if pipeline.retrieval is None:
        raise RuntimeError("Retrieval engine not initialized.")

    # ── Simple query (DeepSeek V4 Flash) ────────────────────────
    print("\n" + "=" * 80)
    print("SIMPLE MODE (DeepSeek V4 Flash)")
    print("=" * 80)

    result = pipeline.retrieval.query(
        "What is the relationship between the API gateway and authentication service?",
        mode="simple",
    )
    print(f"\nAnswer:\n{result['answer']}")

    # Show source traceability
    print("\n📍 Source Chunks:")
    for chunk in result.get("local_chunks", [])[:5]:
        print(
            f"  📄 {chunk.get('source_document', '?')}, "
            f"p.{chunk.get('page_number', '?')}, "
            f"bbox={chunk.get('bbox', {})}"
        )

    # ── Detailed query (DeepSeek V4 Pro + agentic loop) ────────
    print("\n" + "=" * 80)
    print("DETAILED MODE (DeepSeek V4 Pro — Agentic)")
    print("=" * 80)

    result = pipeline.retrieval.query(
        "Explain the complete data flow from client request through the caching layer "
        "to the database, including error handling and retry mechanisms.",
        mode="detailed",
    )
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nIterations: {result['iterations']}")
    print(f"Entities explored: {len(result.get('entities', []))}")

    # ── Incremental insert ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("INCREMENTAL INSERT")
    print("=" * 80)

    pipeline.insert_documents(["data/new_document.pdf"])
    print("New document inserted. Communities updated.")

    # ── End-of-day maintenance ──────────────────────────────────
    pipeline.end_of_day_maintenance()
    print("End-of-day rebalance complete.")


if __name__ == "__main__":
    main()
