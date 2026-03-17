"""
pipeline/orchestrator.py
========================
Runs the full BaseballIQ pipeline end-to-end:
    1. Ingest Statcast (Bronze)
    2. Clean → Silver (DuckDB)
    3. Build Gold analytical datasets
    4. LLM enrichment (skipped in DEMO_MODE)

Usage:
    python -m pipeline.orchestrator --start 2024-07-01 --end 2024-07-07
    DEMO_MODE=1 python -m pipeline.orchestrator  # skip ingestion, use existing data
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import duckdb

from pipeline.config import DEMO_MODE, DUCKDB_PATH
from pipeline.ingestion.statcast_ingestion import ingest_date_range
from pipeline.silver.cleaning import create_players_table, load_bronze_to_silver
from pipeline.silver.feature_engineering import run_all as build_gold

logger = logging.getLogger(__name__)


def run_pipeline(start: str, end: str, skip_enrich: bool = False) -> None:
    logger.info("=" * 60)
    logger.info("BaseballIQ Pipeline — %s → %s", start, end)
    logger.info("=" * 60)

    if not DEMO_MODE:
        # Step 1: Bronze ingestion
        logger.info("[1/4] Ingesting Statcast data...")
        ingest_date_range(start, end)
    else:
        logger.info("[1/4] DEMO_MODE — skipping ingestion")

    # Step 2: Bronze → Silver
    logger.info("[2/4] Loading Bronze → Silver (DuckDB)...")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        if not DEMO_MODE:
            load_bronze_to_silver(con)
            create_players_table(con)

        # Step 3: Silver → Gold
        logger.info("[3/4] Building Gold layer...")
        build_gold(con)

    # Step 4: LLM enrichment
    if skip_enrich or DEMO_MODE:
        logger.info("[4/4] Skipping LLM enrichment (use 'make enrich DATE=...' separately)")
    else:
        logger.info("[4/4] Running LLM enrichment for %s...", end)
        from enrichment.insight_writer import InsightWriter
        with duckdb.connect(str(DUCKDB_PATH)) as con:
            writer = InsightWriter(con)
            game_pks = con.execute(
                "SELECT DISTINCT game_pk FROM pitcher_game_summary WHERE game_date = ?",
                [end]
            ).fetchall()
            for (pk,) in game_pks:
                writer.enrich_game(pk, end)
            writer.detect_anomalies(end)

    logger.info("Pipeline complete ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run full BaseballIQ pipeline")
    parser.add_argument("--start", default=str(date.today()), help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=str(date.today()), help="End date YYYY-MM-DD")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip LLM enrichment step")
    args = parser.parse_args()
    run_pipeline(args.start, args.end, args.skip_enrich)
