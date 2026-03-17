"""
pipeline/silver/cleaning.py
============================
Loads all Bronze Parquet partitions into DuckDB Silver layer.
Applies type casting, deduplication, and null handling.

Usage:
    python -m pipeline.silver.cleaning
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pipeline.config import BRONZE_DIR, DUCKDB_PATH

logger = logging.getLogger(__name__)


def load_bronze_to_silver(con: duckdb.DuckDBPyConnection) -> None:
    """
    Scan all Bronze Parquet partitions and create the Silver `pitches` table.
    Uses DuckDB's hive-partitioned Parquet reader for efficiency.
    """
    bronze_glob = str(BRONZE_DIR / "statcast/**/*.parquet")
    logger.info("Scanning Bronze: %s", bronze_glob)

    con.execute(f"""
    CREATE OR REPLACE TABLE pitches AS
    SELECT
        pitch_id,
        CAST(game_pk      AS INTEGER)   AS game_pk,
        CAST(game_date    AS DATE)       AS game_date,
        CAST(pitcher_id   AS INTEGER)   AS pitcher_id,
        CAST(batter_id    AS INTEGER)   AS batter_id,
        CAST(inning       AS INTEGER)   AS inning,
        inning_topbot,
        pitch_type,
        CAST(release_speed  AS FLOAT)   AS release_speed,
        CAST(release_spin   AS FLOAT)   AS release_spin,
        CAST(pfx_x          AS FLOAT)   AS pfx_x,
        CAST(pfx_z          AS FLOAT)   AS pfx_z,
        CAST(plate_x        AS FLOAT)   AS plate_x,
        CAST(plate_z        AS FLOAT)   AS plate_z,
        CAST(zone           AS INTEGER) AS zone,
        description,
        events,
        stand,
        p_throws,
        CAST(balls          AS INTEGER) AS balls,
        CAST(strikes        AS INTEGER) AS strikes,
        CAST(outs_when_up   AS INTEGER) AS outs_when_up,
        CAST(launch_speed   AS FLOAT)   AS launch_speed,
        CAST(launch_angle   AS FLOAT)   AS launch_angle,
        CAST(estimated_ba   AS FLOAT)   AS estimated_ba,
        CAST(estimated_woba AS FLOAT)   AS estimated_woba,
        CAST(delta_run_exp  AS FLOAT)   AS delta_run_exp,
        CAST(is_barrel      AS BOOLEAN) AS is_barrel,
        home_team,
        away_team,
        CAST(post_home_score AS INTEGER) AS post_home_score,
        CAST(post_away_score AS INTEGER) AS post_away_score,
        NOW()                            AS loaded_at

    FROM read_parquet('{bronze_glob}', hive_partitioning=true)

    -- Deduplicate on pitch_id
    QUALIFY ROW_NUMBER() OVER (PARTITION BY pitch_id ORDER BY game_date) = 1
    """)

    n = con.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    logger.info("Silver pitches table: %d rows", n)

    # Build games lookup table
    con.execute("""
    CREATE OR REPLACE TABLE games AS
    SELECT DISTINCT
        game_pk,
        game_date,
        home_team,
        away_team,
        MAX(post_home_score) AS home_score,
        MAX(post_away_score) AS away_score
    FROM pitches
    GROUP BY game_pk, game_date, home_team, away_team
    """)
    logger.info("Silver games table: %d rows",
                con.execute("SELECT COUNT(*) FROM games").fetchone()[0])


def create_players_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build a players lookup from pitcher/batter IDs found in pitches.
    In production this would join to a player registry API.
    Here we create a minimal table with IDs.
    """
    con.execute("""
    CREATE OR REPLACE TABLE players AS
    SELECT DISTINCT
        pitcher_id AS player_id,
        CONCAT('Pitcher #', pitcher_id) AS full_name,
        p_throws AS hand,
        'SP' AS position
    FROM pitches
    WHERE pitcher_id IS NOT NULL

    UNION ALL

    SELECT DISTINCT
        batter_id AS player_id,
        CONCAT('Batter #', batter_id) AS full_name,
        stand AS hand,
        'POS' AS position
    FROM pitches
    WHERE batter_id IS NOT NULL
    """)
    logger.info("Silver players table created")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        load_bronze_to_silver(con)
        create_players_table(con)
        logger.info("Silver layer complete: %s", DUCKDB_PATH)
