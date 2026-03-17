"""
pipeline/gold/aggregations.py
==============================
Builds Gold analytical datasets from Silver DuckDB tables.
Calls feature_engineering.py to produce the final Gold Parquet files.

Usage:
    python -m pipeline.gold.aggregations
"""

from __future__ import annotations

import logging

import duckdb

from pipeline.config import DUCKDB_PATH
from pipeline.silver.feature_engineering import run_all

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        run_all(con)
        logger.info("Gold layer complete.")
