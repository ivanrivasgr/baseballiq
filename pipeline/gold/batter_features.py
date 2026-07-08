"""
pipeline/gold/batter_features.py
=================================
Batter form block for the HR-prop Gold layer.

Builds `batter_game_rolling`: one row per batter-game carrying
    - this game's raw outcome counts (pa, hr, ...) — used ONLY to build the
      target and future windows, never as features for the same game
    - rolling form features over 7/30/365-day windows

LEAKAGE RULE (do not break): every window frame ends at
`INTERVAL 1 DAY PRECEDING`. Rolling stats for a game NEVER include that
game's own PAs, nor any same-day PAs (doubleheader game 1 is excluded from
game 2's features — a small cost for a hard temporal guarantee).

Rates are computed as (sum of numerators) / (sum of denominators) over the
window — never an average of per-game rates, which over-weights 1-PA games.
"""

from __future__ import annotations

import logging

import duckdb

from pipeline.config import DUCKDB_PATH

logger = logging.getLogger(__name__)

# Empirical-Bayes shrinkage weight (in PAs) blending the 30-day HR rate
# toward the batter's 365-day rate; ~100 PAs ≈ 3-4 weeks of everyday play.
K_SHRINK_PA = 100


def build_batter_game_rolling(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building batter_game_rolling (Gold)")

    con.execute(f"""
    CREATE OR REPLACE TABLE batter_game_rolling AS

    WITH lineup AS (
        -- Batting-order slot: order of first PA within the team's game.
        -- In-game outcome — allowed only for building rolling priors.
        SELECT
            batter_id, game_pk,
            ROW_NUMBER() OVER (
                PARTITION BY game_pk, batter_team
                ORDER BY MIN(at_bat_number)
            ) AS lineup_slot_actual
        FROM plate_appearances
        GROUP BY batter_id, game_pk, batter_team
    ),

    per_game AS (
        SELECT
            pa.batter_id,
            pa.game_pk,
            pa.game_date,
            ANY_VALUE(pa.batter_team)                                   AS batter_team,
            ANY_VALUE(pa.home_team)                                     AS home_team,
            ANY_VALUE(pa.away_team)                                     AS away_team,
            MODE(pa.stand)                                              AS stand,
            COUNT(*)                                                    AS pa,
            SUM(pa.is_hr)                                               AS hr,
            SUM(pa.is_batted_ball)                                      AS bbe,
            SUM(CASE WHEN pa.is_barrel THEN 1 ELSE 0 END)               AS barrels,
            SUM(CASE WHEN pa.launch_speed >= 95 THEN 1 ELSE 0 END)      AS hard_hits,
            SUM(CASE WHEN pa.bb_type = 'fly_ball' THEN 1 ELSE 0 END)    AS fly_balls,
            SUM(CASE WHEN pa.bb_type = 'fly_ball'
                      AND pa.spray_angle_adj < -15 THEN 1 ELSE 0 END)   AS pulled_fb,
            SUM(pa.launch_speed)                                        AS ev_sum,
            MAX(pa.launch_speed)                                        AS ev_max,
            -- platoon splits (vs pitcher handedness)
            SUM(CASE WHEN pa.p_throws = 'R' THEN 1 ELSE 0 END)          AS pa_vs_r,
            SUM(CASE WHEN pa.p_throws = 'R' THEN pa.is_hr ELSE 0 END)   AS hr_vs_r,
            SUM(CASE WHEN pa.p_throws = 'L' THEN 1 ELSE 0 END)          AS pa_vs_l,
            SUM(CASE WHEN pa.p_throws = 'L' THEN pa.is_hr ELSE 0 END)   AS hr_vs_l,
            ANY_VALUE(l.lineup_slot_actual)                             AS lineup_slot_actual
        FROM plate_appearances pa
        LEFT JOIN lineup l
          ON l.batter_id = pa.batter_id AND l.game_pk = pa.game_pk
        GROUP BY pa.batter_id, pa.game_pk, pa.game_date
    ),

    rolled AS (
        SELECT
            *,
            -- window sums (all strictly before game_date)
            COUNT(*)        OVER w30  AS games_30d,
            SUM(pa)         OVER w7   AS pa_7d,
            SUM(hr)         OVER w7   AS hr_7d,
            SUM(pa)         OVER w30  AS pa_30d,
            SUM(hr)         OVER w30  AS hr_30d,
            SUM(bbe)        OVER w30  AS bbe_30d,
            SUM(barrels)    OVER w30  AS barrels_30d,
            SUM(hard_hits)  OVER w30  AS hard_hits_30d,
            SUM(fly_balls)  OVER w30  AS fly_balls_30d,
            SUM(pulled_fb)  OVER w30  AS pulled_fb_30d,
            SUM(ev_sum)     OVER w30  AS ev_sum_30d,
            MAX(ev_max)     OVER w30  AS ev_max_30d,
            AVG(lineup_slot_actual) OVER w30 AS avg_slot_30d,
            SUM(pa)         OVER w365 AS pa_365d,
            SUM(hr)         OVER w365 AS hr_365d,
            SUM(pa_vs_r)    OVER w365 AS pa_vs_r_365d,
            SUM(hr_vs_r)    OVER w365 AS hr_vs_r_365d,
            SUM(pa_vs_l)    OVER w365 AS pa_vs_l_365d,
            SUM(hr_vs_l)    OVER w365 AS hr_vs_l_365d
        FROM per_game
        WINDOW
            w7 AS (PARTITION BY batter_id ORDER BY game_date
                   RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING),
            w30 AS (PARTITION BY batter_id ORDER BY game_date
                    RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING),
            w365 AS (PARTITION BY batter_id ORDER BY game_date
                     RANGE BETWEEN INTERVAL 365 DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING)
    )

    SELECT
        batter_id, game_pk, game_date,
        batter_team, home_team, away_team, stand,
        -- current-game outcomes: target material, NOT features
        pa  AS pa_this_game,
        hr  AS hr_this_game,
        lineup_slot_actual,
        -- rolling features (T-1day availability)
        pa_30d                                                  AS b_pa_30d,
        ROUND(hr_7d  / NULLIF(pa_7d,  0), 5)                    AS b_hr_per_pa_7d,
        ROUND(hr_30d / NULLIF(pa_30d, 0), 5)                    AS b_hr_per_pa_30d,
        ROUND(hr_365d / NULLIF(pa_365d, 0), 5)                  AS b_hr_per_pa_365d,
        -- COALESCE the 30d window sums: an empty 30d window (IL return,
        -- callup) must collapse to the 365d rate, not to NULL.
        ROUND(
            (COALESCE(hr_30d, 0) + {K_SHRINK_PA} * (hr_365d / NULLIF(pa_365d, 0)))
            / NULLIF(COALESCE(pa_30d, 0) + {K_SHRINK_PA}, 0), 5) AS b_hr_per_pa_shrunk,
        ROUND(barrels_30d   / NULLIF(bbe_30d, 0), 5)            AS b_barrel_rate_30d,
        ROUND(hard_hits_30d / NULLIF(bbe_30d, 0), 5)            AS b_hard_hit_30d,
        ROUND(fly_balls_30d / NULLIF(bbe_30d, 0), 5)            AS b_fb_rate_30d,
        ROUND(pulled_fb_30d / NULLIF(fly_balls_30d, 0), 5)      AS b_pull_fb_rate_30d,
        ROUND(ev_sum_30d / NULLIF(bbe_30d, 0), 2)               AS b_avg_ev_30d,
        ev_max_30d                                              AS b_max_ev_30d,
        ROUND(pa_30d * 1.0 / NULLIF(games_30d, 0), 3)           AS b_avg_pa_per_game_30d,
        ROUND(avg_slot_30d, 2)                                  AS b_avg_slot_30d,
        ROUND(hr_vs_r_365d / NULLIF(pa_vs_r_365d, 0), 5)        AS b_hr_per_pa_vs_r_365d,
        ROUND(hr_vs_l_365d / NULLIF(pa_vs_l_365d, 0), 5)        AS b_hr_per_pa_vs_l_365d
    FROM rolled
    ORDER BY game_date, batter_id;
    """)

    n = con.execute("SELECT COUNT(*) FROM batter_game_rolling").fetchone()[0]
    logger.info("batter_game_rolling: %d batter-game rows", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        build_batter_game_rolling(con)
