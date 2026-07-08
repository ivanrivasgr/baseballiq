"""
pipeline/gold/context_features.py
==================================
Game-context block for the HR-prop Gold layer: one row per game_pk with
park HR factors, weather (wind-out projection against park orientation),
and day/night.

Degrades gracefully: `schedule` and `game_weather` are LEFT-joined and may
be entirely absent (e.g. fixture tests, partial backfills) — context columns
are NULL rather than dropping games.

Roof handling: domes and retractable-roof parks get wind zeroed (we don't
know historical roof state; wind is structurally suppressed either way) and
temperature defaulted to 72°F indoors when missing.
"""

from __future__ import annotations

import logging

import duckdb

from pipeline.config import DUCKDB_PATH
from pipeline.gold.park_data import register_parks_table

logger = logging.getLogger(__name__)


def _table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0] > 0


def build_game_context(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building game_context (Gold)")

    seasons = [r[0] for r in con.execute(
        "SELECT DISTINCT YEAR(game_date) FROM plate_appearances"
    ).fetchall()]
    register_parks_table(con, seasons or [2024])

    schedule_join = (
        "LEFT JOIN schedule s ON s.game_pk = g.game_pk"
        if _table_exists(con, "schedule")
        else "LEFT JOIN (SELECT NULL::BIGINT AS game_pk, NULL::VARCHAR AS day_night) s ON s.game_pk = g.game_pk"
    )
    weather_join = (
        "LEFT JOIN game_weather w ON w.game_pk = g.game_pk"
        if _table_exists(con, "game_weather")
        else ("LEFT JOIN (SELECT NULL::BIGINT AS game_pk, NULL::FLOAT AS temp_f, "
              "NULL::FLOAT AS wind_mph, NULL::FLOAT AS wind_dir_deg) w ON w.game_pk = g.game_pk")
    )

    con.execute(f"""
    CREATE OR REPLACE TABLE game_context AS

    WITH g AS (
        SELECT DISTINCT game_pk, game_date, home_team
        FROM plate_appearances
    )

    SELECT
        g.game_pk,
        g.game_date,
        g.home_team,
        p.venue_name,
        p.roof,
        p.park_hr_factor,
        p.park_hr_factor_lhb,
        p.park_hr_factor_rhb,
        CASE WHEN s.day_night = 'night' THEN 1
             WHEN s.day_night = 'day'   THEN 0
        END                                                     AS is_night,
        -- indoors default when roofed and no reading
        CASE WHEN p.roof IN ('dome', 'retractable')
             THEN COALESCE(w.temp_f, 72.0)
             ELSE w.temp_f
        END                                                     AS temp_f,
        -- wind blowing out to CF (+) / in from CF (−), mph;
        -- wind_dir is meteorological (direction wind comes FROM), so a wind
        -- FROM (cf_bearing + 180°) blows straight out to center.
        CASE WHEN p.roof IN ('dome', 'retractable') THEN 0.0
             ELSE ROUND(
                 w.wind_mph * COS(RADIANS(w.wind_dir_deg - p.cf_bearing_deg - 180)), 2)
        END                                                     AS wind_out_mph
    FROM g
    {schedule_join}
    {weather_join}
    LEFT JOIN parks p
      ON p.team_abbr = g.home_team AND p.season = YEAR(g.game_date);
    """)

    n = con.execute("SELECT COUNT(*) FROM game_context").fetchone()[0]
    logger.info("game_context: %d games", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        build_game_context(con)
