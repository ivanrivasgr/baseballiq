"""
pipeline/silver/plate_appearances.py
=====================================
Collapses the Silver `pitches` table to plate-appearance grain.

One row per (game_pk, at_bat_number): the final pitch of each at-bat carries
the PA outcome in `events`. This is the training grain for everything
downstream in the HR-prop pipeline — `is_hr` here is the source of truth for
the Gold-layer target.

Spray angle: computed from Statcast hit coordinates and sign-adjusted by
batter handedness so that negative = pulled for every batter. Pulled fly
balls are the primary HR channel, so this feeds b_pull_fb_rate in Gold.

Usage:
    python -m pipeline.silver.plate_appearances
"""

from __future__ import annotations

import logging

import duckdb

from pipeline.config import DUCKDB_PATH

logger = logging.getLogger(__name__)

# Statcast hit-coordinate origin (home plate) in the hc_x/hc_y system.
HOME_PLATE_HC_X = 125.42
HOME_PLATE_HC_Y = 198.27


def build_plate_appearances(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build the `plate_appearances` table: one row per PA with its outcome.

    Grain guarantee: exactly one row per (game_pk, at_bat_number) — enforced
    by taking the last pitch (max pitch_number) of each at-bat.
    """
    logger.info("Building plate_appearances (Silver)")

    con.execute(f"""
    CREATE OR REPLACE TABLE plate_appearances AS
    SELECT
        game_pk,
        game_date,
        at_bat_number,
        batter_id,
        pitcher_id,
        stand,
        p_throws,
        inning,
        inning_topbot,
        home_team,
        away_team,
        -- The batter's team: bottom of inning = home team batting
        CASE WHEN inning_topbot = 'Bot' THEN home_team ELSE away_team END AS batter_team,
        CASE WHEN inning_topbot = 'Bot' THEN away_team ELSE home_team END AS pitching_team,
        events,
        CAST(events = 'home_run' AS INTEGER)          AS is_hr,
        CAST(launch_speed IS NOT NULL AS INTEGER)     AS is_batted_ball,
        launch_speed,
        launch_angle,
        estimated_woba,
        is_barrel,
        bb_type,
        -- Spray angle in degrees from straightaway CF; sign-flipped by
        -- handedness so negative = pull side for all batters.
        CASE WHEN hc_x IS NOT NULL AND hc_y IS NOT NULL THEN
            DEGREES(ATAN2(hc_x - {HOME_PLATE_HC_X}, {HOME_PLATE_HC_Y} - hc_y))
            * (CASE WHEN stand = 'L' THEN -1 ELSE 1 END)
        END                                           AS spray_angle_adj
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY game_pk, at_bat_number
                   ORDER BY pitch_number DESC
               ) AS rn
        FROM pitches
        WHERE at_bat_number IS NOT NULL
    )
    WHERE rn = 1
      AND events IS NOT NULL AND events <> ''
      AND batter_id IS NOT NULL;
    """)

    n, n_hr = con.execute(
        "SELECT COUNT(*), SUM(is_hr) FROM plate_appearances"
    ).fetchone()
    logger.info("plate_appearances: %d PAs, %s HRs", n, n_hr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        build_plate_appearances(con)
