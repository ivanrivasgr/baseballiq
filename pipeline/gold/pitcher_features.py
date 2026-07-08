"""
pipeline/gold/pitcher_features.py
==================================
Opposing-pitcher block for the HR-prop Gold layer. Three tables:

    game_starters         — who started each (game_pk, pitching_team)
    pitcher_game_rolling  — rolling HR susceptibility per pitcher-game
    bullpen_team_rolling  — rolling bullpen HR/PA allowed per team-game
                            (the starter only faces a batter ~2-3 times;
                            the bullpen covers the rest of the exposure)

Same leakage rule as batter_features: every window ends at
`INTERVAL 1 DAY PRECEDING`; rates are sums-over-sums, never averaged rates.
"""

from __future__ import annotations

import logging

import duckdb

from pipeline.config import DUCKDB_PATH

logger = logging.getLogger(__name__)


def build_game_starters(con: duckdb.DuckDBPyConnection) -> None:
    """The starter is the pitcher of the team's first PA faced in the game."""
    con.execute("""
    CREATE OR REPLACE TABLE game_starters AS
    SELECT game_pk, game_date, pitching_team, pitcher_id AS starter_id
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY game_pk, pitching_team
                   ORDER BY at_bat_number
               ) AS rn
        FROM plate_appearances
    )
    WHERE rn = 1;
    """)
    n = con.execute("SELECT COUNT(*) FROM game_starters").fetchone()[0]
    logger.info("game_starters: %d team-game rows", n)


def build_pitcher_game_rolling(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building pitcher_game_rolling (Gold)")

    con.execute("""
    CREATE OR REPLACE TABLE pitcher_game_rolling AS

    WITH per_game AS (
        SELECT
            pitcher_id,
            game_pk,
            game_date,
            MODE(p_throws)                                          AS p_throws,
            COUNT(*)                                                AS pa,
            SUM(is_hr)                                              AS hr,
            SUM(is_batted_ball)                                     AS bbe,
            SUM(CASE WHEN is_barrel THEN 1 ELSE 0 END)              AS barrels,
            SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END)   AS fly_balls,
            SUM(launch_speed)                                       AS ev_sum
        FROM plate_appearances
        WHERE pitcher_id IS NOT NULL
        GROUP BY pitcher_id, game_pk, game_date
    ),

    rolled AS (
        SELECT
            *,
            SUM(pa)        OVER w30  AS pa_30d,
            SUM(hr)        OVER w30  AS hr_30d,
            SUM(bbe)       OVER w30  AS bbe_30d,
            SUM(barrels)   OVER w30  AS barrels_30d,
            SUM(fly_balls) OVER w30  AS fly_balls_30d,
            SUM(ev_sum)    OVER w30  AS ev_sum_30d,
            SUM(pa)        OVER w365 AS pa_365d,
            SUM(hr)        OVER w365 AS hr_365d
        FROM per_game
        WINDOW
            w30 AS (PARTITION BY pitcher_id ORDER BY game_date
                    RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING),
            w365 AS (PARTITION BY pitcher_id ORDER BY game_date
                     RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING)
    )

    SELECT
        pitcher_id, game_pk, game_date, p_throws,
        pa_30d                                              AS p_pa_30d,
        ROUND(hr_30d  / NULLIF(pa_30d,  0), 5)              AS p_hr_per_pa_allowed_30d,
        ROUND(hr_365d / NULLIF(pa_365d, 0), 5)              AS p_hr_per_pa_allowed_365d,
        ROUND(barrels_30d   / NULLIF(bbe_30d, 0), 5)        AS p_barrel_rate_allowed_30d,
        ROUND(fly_balls_30d / NULLIF(bbe_30d, 0), 5)        AS p_fb_rate_allowed_30d,
        ROUND(ev_sum_30d / NULLIF(bbe_30d, 0), 2)           AS p_avg_ev_allowed_30d
    FROM rolled
    ORDER BY game_date, pitcher_id;
    """)

    n = con.execute("SELECT COUNT(*) FROM pitcher_game_rolling").fetchone()[0]
    logger.info("pitcher_game_rolling: %d pitcher-game rows", n)


def build_bullpen_team_rolling(con: duckdb.DuckDBPyConnection) -> None:
    """Team bullpen (non-starter) HR/PA allowed, rolling 30 days."""
    logger.info("Building bullpen_team_rolling (Gold)")

    con.execute("""
    CREATE OR REPLACE TABLE bullpen_team_rolling AS

    -- Spine = EVERY team-game, so a game where only the starter pitched
    -- still carries the team's rolling bullpen prior into the join.
    WITH team_games AS (
        SELECT DISTINCT pitching_team, game_pk, game_date
        FROM plate_appearances
    ),

    bullpen_per_game AS (
        SELECT
            pa.pitching_team,
            pa.game_pk,
            pa.game_date,
            COUNT(*)       AS pa,
            SUM(pa.is_hr)  AS hr
        FROM plate_appearances pa
        JOIN game_starters gs
          ON gs.game_pk = pa.game_pk AND gs.pitching_team = pa.pitching_team
        WHERE pa.pitcher_id <> gs.starter_id
        GROUP BY pa.pitching_team, pa.game_pk, pa.game_date
    )

    SELECT
        tg.pitching_team, tg.game_pk, tg.game_date,
        SUM(COALESCE(bp.pa, 0)) OVER w30 AS bp_pa_30d,
        ROUND(
            SUM(COALESCE(bp.hr, 0)) OVER w30
            / NULLIF(SUM(COALESCE(bp.pa, 0)) OVER w30, 0), 5)
                                         AS bp_hr_per_pa_allowed_30d
    FROM team_games tg
    LEFT JOIN bullpen_per_game bp
      ON bp.pitching_team = tg.pitching_team AND bp.game_pk = tg.game_pk
    WINDOW
        w30 AS (PARTITION BY tg.pitching_team ORDER BY tg.game_date
                RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING)
    ORDER BY tg.game_date, tg.pitching_team;
    """)

    n = con.execute("SELECT COUNT(*) FROM bullpen_team_rolling").fetchone()[0]
    logger.info("bullpen_team_rolling: %d team-game rows", n)


def run_all(con: duckdb.DuckDBPyConnection) -> None:
    build_game_starters(con)
    build_pitcher_game_rolling(con)
    build_bullpen_team_rolling(con)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        run_all(con)
