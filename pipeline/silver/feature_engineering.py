"""
pipeline/silver/feature_engineering.py
=======================================
Transforms cleaned Silver pitches table into analytical features
for the Gold layer pitcher_game_summary and batter_game_summary tables.

Runs inside DuckDB — no Pandas required for the heavy lifting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pipeline.config import DUCKDB_PATH, GOLD_DIR

logger = logging.getLogger(__name__)


def build_pitcher_game_summary(con: duckdb.DuckDBPyConnection) -> None:
    """
    Aggregate pitch-level Silver data into pitcher-game Gold summary.

    Key metrics:
        - Velocity and spin averages
        - Whiff rate, CSW rate, zone rate, chase rate
        - xwOBA allowed (proxy for quality of contact)
        - Run expectancy delta (pitcher value metric)
        - Pitch mix entropy (stuff diversity)

    Also computes 30-day rolling context columns via window functions.
    """
    logger.info("Building pitcher_game_summary (Gold)")

    con.execute("""
    CREATE OR REPLACE TABLE pitcher_game_summary AS

    WITH game_agg AS (
        SELECT
            pitcher_id,
            game_pk,
            game_date,

            -- Volume
            COUNT(*)                                                    AS total_pitches,

            -- Velocity & movement
            ROUND(AVG(release_speed), 2)                                AS avg_velo,
            ROUND(STDDEV(release_speed), 2)                             AS velo_stddev,
            ROUND(AVG(release_spin), 0)                                 AS avg_spin,
            ROUND(AVG(ABS(pfx_x)), 2)                                   AS avg_h_break,
            ROUND(AVG(pfx_z), 2)                                        AS avg_v_break,

            -- Outcomes
            ROUND(
                SUM(CASE WHEN description = 'swinging_strike' THEN 1.0 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN description LIKE '%swing%' THEN 1.0 ELSE 0 END), 0),
            4)                                                          AS whiff_rate,

            ROUND(
                SUM(CASE WHEN description IN ('called_strike','swinging_strike') THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
            4)                                                          AS csw_rate,

            ROUND(
                SUM(CASE WHEN zone BETWEEN 1 AND 9 THEN 1.0 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
            4)                                                          AS zone_rate,

            -- Chase: swing on ball outside zone
            ROUND(
                SUM(CASE WHEN zone IN (11,12,13,14) AND description LIKE '%swing%' THEN 1.0 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN zone IN (11,12,13,14) THEN 1.0 ELSE 0 END), 0),
            4)                                                          AS chase_rate,

            -- Contact quality allowed
            ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN estimated_woba END), 4)
                                                                        AS avg_xwoba_allowed,
            ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END), 2)
                                                                        AS avg_exit_velo_allowed,
            ROUND(
                SUM(CASE WHEN is_barrel THEN 1.0 ELSE 0 END)
                / NULLIF(SUM(CASE WHEN launch_speed IS NOT NULL THEN 1.0 ELSE 0 END), 0),
            4)                                                          AS barrel_rate_allowed,

            -- Run value
            ROUND(SUM(delta_run_exp), 3)                                AS total_re_delta,

            -- Pitch mix entropy (stuff diversity: high = more unpredictable)
            ROUND(
                -SUM(pitch_pct * LN(NULLIF(pitch_pct, 0)))
            , 4)                                                        AS stuff_diversity

        FROM pitches
        CROSS JOIN LATERAL (
            -- Subquery to compute per-pitcher-game pitch type %
            SELECT pitch_type, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pitch_pct
            FROM pitches p2
            WHERE p2.pitcher_id = pitches.pitcher_id
              AND p2.game_pk    = pitches.game_pk
            GROUP BY pitch_type
        ) pitch_mix
        WHERE pitcher_id IS NOT NULL
        GROUP BY pitcher_id, game_pk, game_date
    ),

    with_rolling AS (
        SELECT
            *,
            -- 30-day rolling averages using window functions
            ROUND(AVG(avg_velo) OVER (
                PARTITION BY pitcher_id
                ORDER BY game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
            ), 2)                                                       AS rolling_30d_avg_velo,

            ROUND(AVG(whiff_rate) OVER (
                PARTITION BY pitcher_id
                ORDER BY game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
            ), 4)                                                       AS rolling_30d_whiff_rate,

            ROUND(AVG(csw_rate) OVER (
                PARTITION BY pitcher_id
                ORDER BY game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
            ), 4)                                                       AS rolling_30d_csw_rate,

            -- Delta vs rolling baseline (key engineered features)
            ROUND(avg_velo - AVG(avg_velo) OVER (
                PARTITION BY pitcher_id
                ORDER BY game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
            ), 2)                                                       AS velo_vs_30d_avg,

            ROUND(whiff_rate - AVG(whiff_rate) OVER (
                PARTITION BY pitcher_id
                ORDER BY game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING
            ), 4)                                                       AS whiff_rate_delta,

            -- Season averages
            ROUND(AVG(avg_velo)   OVER (PARTITION BY pitcher_id), 2)   AS season_avg_velo,
            ROUND(AVG(csw_rate)   OVER (PARTITION BY pitcher_id), 4)   AS season_avg_csw,
            ROUND(AVG(whiff_rate) OVER (PARTITION BY pitcher_id), 4)   AS season_avg_whiff,

            -- Performance tier label (used for LLM enrichment context)
            CASE
                WHEN csw_rate >= 0.32 THEN 'elite'
                WHEN csw_rate >= 0.29 THEN 'above_avg'
                WHEN csw_rate >= 0.26 THEN 'average'
                WHEN csw_rate >= 0.22 THEN 'below_avg'
                ELSE 'poor'
            END                                                         AS performance_tier

        FROM game_agg
    )

    SELECT * FROM with_rolling
    ORDER BY game_date DESC, pitcher_id;
    """)

    # Export to Gold Parquet
    out = GOLD_DIR / "pitcher_game_summary.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY pitcher_game_summary TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
    logger.info("Gold table written: %s", out)


def build_batter_game_summary(con: duckdb.DuckDBPyConnection) -> None:
    """
    Aggregate batter-level Gold table with contact quality metrics.
    """
    logger.info("Building batter_game_summary (Gold)")

    con.execute("""
    CREATE OR REPLACE TABLE batter_game_summary AS
    SELECT
        batter_id,
        game_pk,
        game_date,

        COUNT(*)                                                        AS pitches_seen,
        COUNT(DISTINCT inning)                                          AS innings_appeared,

        -- Plate discipline
        ROUND(AVG(CASE WHEN description LIKE '%swing%' THEN 1.0 ELSE 0 END), 4)
                                                                        AS swing_rate,
        ROUND(
            SUM(CASE WHEN zone IN (11,12,13,14) AND description LIKE '%swing%' THEN 1.0 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN zone IN (11,12,13,14) THEN 1.0 ELSE 0 END), 0),
        4)                                                              AS o_swing_rate,

        -- Contact quality (batted balls only)
        ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_speed END), 2)
                                                                        AS avg_exit_velo,
        ROUND(AVG(CASE WHEN launch_speed IS NOT NULL THEN launch_angle END), 2)
                                                                        AS avg_launch_angle,
        ROUND(AVG(estimated_woba), 4)                                   AS avg_xwoba,
        ROUND(
            SUM(CASE WHEN is_barrel THEN 1.0 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN launch_speed IS NOT NULL THEN 1.0 ELSE 0 END), 0),
        4)                                                              AS barrel_rate,

        -- Hard hit rate (EV >= 95 mph)
        ROUND(
            SUM(CASE WHEN launch_speed >= 95 THEN 1.0 ELSE 0 END)
            / NULLIF(SUM(CASE WHEN launch_speed IS NOT NULL THEN 1.0 ELSE 0 END), 0),
        4)                                                              AS hard_hit_rate,

        -- Run value created
        ROUND(SUM(delta_run_exp), 3)                                    AS total_re_created

    FROM pitches
    WHERE batter_id IS NOT NULL
    GROUP BY batter_id, game_pk, game_date
    ORDER BY game_date DESC, avg_xwoba DESC;
    """)

    out = GOLD_DIR / "batter_game_summary.parquet"
    con.execute(f"COPY batter_game_summary TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
    logger.info("Gold table written: %s", out)


def run_all(con: duckdb.DuckDBPyConnection) -> None:
    """Build all Gold tables."""
    build_pitcher_game_summary(con)
    build_batter_game_summary(con)
    logger.info("All Gold tables built successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        run_all(con)
