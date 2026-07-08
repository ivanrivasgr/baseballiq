"""
pipeline/gold/matchup_table.py
===============================
Assembles the HR-prop training/scoring table `batter_matchup_features`:

    one row = one batter in one game
    target  = hr_hit (any PA that game ended in a home run)

Joins, in order:
    batter_game_rolling   — batter form (rolling, strictly prior days)
    game_starters         — opposing starter identity
    pitcher_game_rolling  — that starter's rolling HR susceptibility
    bullpen_team_rolling  — opposing bullpen rolling HR/PA
    game_context          — park factors, weather, day/night

All rolling inputs already obey the `INTERVAL 1 DAY PRECEDING` rule; this
module adds no windows of its own. The only same-game fields it emits are
the target (`hr_hit`) and explicitly-labeled non-feature columns
(`pa_this_game`, `lineup_slot_actual`) which the model layer must exclude.

Usage:
    python -m pipeline.gold.matchup_table
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pipeline.config import DUCKDB_PATH, GOLD_DIR
from pipeline.gold.batter_features import build_batter_game_rolling
from pipeline.gold.context_features import build_game_context
from pipeline.gold.pitcher_features import run_all as build_pitcher_tables

logger = logging.getLogger(__name__)


def build_batter_matchup_features(
    con: duckdb.DuckDBPyConnection, export_path: Path | None = None
) -> None:
    """Build the table; export to parquet only when export_path is given
    (production paths pass it explicitly — tests never write to data/)."""
    logger.info("Building batter_matchup_features (Gold)")

    con.execute("""
    CREATE OR REPLACE TABLE batter_matchup_features AS

    SELECT
        -- keys
        b.batter_id,
        b.game_pk,
        b.game_date,
        b.batter_team,
        b.stand,
        gs.starter_id                                           AS opp_starter_id,

        -- ── TARGET ────────────────────────────────────────────
        CAST(b.hr_this_game > 0 AS INTEGER)                     AS hr_hit,

        -- same-game outcomes: NOT features (analysis / v2 profiles only)
        b.pa_this_game,
        b.lineup_slot_actual,

        -- ── batter form (T-1day) ──────────────────────────────
        b.b_pa_30d,
        b.b_hr_per_pa_7d,
        b.b_hr_per_pa_30d,
        b.b_hr_per_pa_365d,
        b.b_hr_per_pa_shrunk,
        b.b_barrel_rate_30d,
        b.b_hard_hit_30d,
        b.b_fb_rate_30d,
        b.b_pull_fb_rate_30d,
        b.b_avg_ev_30d,
        b.b_max_ev_30d,
        b.b_avg_pa_per_game_30d,
        b.b_avg_slot_30d,

        -- platoon: batter's long-window HR rate vs the starter's hand
        -- (unknown starter hand → NULL, never a silent vs-R default)
        CASE WHEN p.p_throws = 'L' THEN b.b_hr_per_pa_vs_l_365d
             WHEN p.p_throws = 'R' THEN b.b_hr_per_pa_vs_r_365d
        END                                                     AS b_hr_per_pa_vs_hand_365d,

        -- ── opposing starter (T-1day) ─────────────────────────
        p.p_throws                                              AS opp_starter_throws,
        CAST(b.stand = p.p_throws AS INTEGER)                   AS same_handed,
        p.p_pa_30d,
        p.p_hr_per_pa_allowed_30d,
        p.p_hr_per_pa_allowed_365d,
        p.p_barrel_rate_allowed_30d,
        p.p_fb_rate_allowed_30d,
        p.p_avg_ev_allowed_30d,

        -- ── opposing bullpen (T-1day) ─────────────────────────
        bp.bp_hr_per_pa_allowed_30d,

        -- ── game context ──────────────────────────────────────
        CAST(b.batter_team = b.home_team AS INTEGER)            AS is_home,
        CASE WHEN b.stand = 'L' THEN gc.park_hr_factor_lhb
             ELSE gc.park_hr_factor_rhb
        END                                                     AS park_hr_factor_hand,
        gc.park_hr_factor,
        gc.roof,
        gc.is_night,
        gc.temp_f,
        gc.wind_out_mph

    FROM batter_game_rolling b
    LEFT JOIN game_starters gs
      ON gs.game_pk = b.game_pk
     AND gs.pitching_team <> b.batter_team
    LEFT JOIN pitcher_game_rolling p
      ON p.pitcher_id = gs.starter_id AND p.game_pk = b.game_pk
    LEFT JOIN bullpen_team_rolling bp
      ON bp.pitching_team = gs.pitching_team AND bp.game_pk = b.game_pk
    LEFT JOIN game_context gc
      ON gc.game_pk = b.game_pk
    ORDER BY b.game_date, b.game_pk, b.batter_id;
    """)

    n, hr, hr_rate = con.execute("""
        SELECT COUNT(*), SUM(hr_hit), ROUND(AVG(hr_hit), 4)
        FROM batter_matchup_features
    """).fetchone()
    logger.info("batter_matchup_features: %d rows, %s HR games (base rate %s)",
                n, hr, hr_rate)

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY batter_matchup_features TO '{export_path}' "
                    f"(FORMAT PARQUET, COMPRESSION SNAPPY)")
        logger.info("Gold table written: %s", export_path)


DEFAULT_EXPORT_PATH = GOLD_DIR / "batter_matchup_features.parquet"


def run_all(con: duckdb.DuckDBPyConnection, export_path: Path | None = None) -> None:
    """Build the full HR-prop Gold layer in dependency order."""
    build_batter_game_rolling(con)
    build_pitcher_tables(con)
    build_game_context(con)
    build_batter_matchup_features(con, export_path=export_path)
    logger.info("HR-prop Gold layer complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        run_all(con, export_path=DEFAULT_EXPORT_PATH)
