"""
pipeline/gold/matchup_board.py
===============================
Slate matchup board: one row per hitter vs the opposing probable starter,
for every game on a slate date — the Kasper-style grid, built entirely from
our own Silver tables ($0 data cost).

Column map vs the reference screen:

    reference        ours                        source
    ─────────────    ─────────────────────────   ─────────────────────────
    PIT              pit_365d (pitches seen)     silver.pitches
    BIP              bip_365d (balls in play)    silver.plate_appearances
    ISO              iso_365d                    PA events (TB−1B over AB)
    XWOBA            xwoba_365d (approximated)   estimated_woba + BB/HBP wts
    XWOBAC           xwobacon_365d               estimated_woba on BBE
    SWS%             sws_365d (swstr/pitch)      pitch descriptions
    PBRL%            pbrl_365d (pulled barrels)  barrel & spray < −15°
    BRL%             brl_365d                    barrels / BBE
    SWSP%            swsp_365d (sweet spot)      launch angle 8–32° / BBE
    FB%              fb_365d                     bb_type
    HH%              hh_365d                     EV ≥ 95 / BBE
    LA               la_365d                     mean launch angle
    FORM             form_pct + form_arrow       30d vs 365d xwOBAcon
    KHR (theirs)     hr_prob (OURS, better)      calibrated model P(HR),
                                                 joined when scores exist

Their MATCH/CEIL/ZONE composites are proprietary blends of the same public
inputs; our calibrated probability + Voss Edge tier is the principled
replacement, so we don't fake them.

Temporal rule: the board for slate_date uses ONLY data with
game_date < slate_date (single as-of cutoff — same guarantee as the
1-DAY-PRECEDING windows in the training tables).

Lineups aren't posted morning-of, so the hitter list per team is a roster
proxy: batters who took PAs for that team in the trailing window, ordered
by their recent average lineup slot.

Usage:
    python -m pipeline.gold.matchup_board --date 2025-07-04
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from pipeline.config import DUCKDB_PATH, GOLD_DIR

logger = logging.getLogger(__name__)

ROSTER_LOOKBACK_DAYS = 14      # how recent a PA keeps a batter on the proxy roster
ROSTER_MAX_HITTERS = 13        # per team
FORM_STEADY_BAND = 0.05        # ±5% of baseline counts as "steady"

# wOBA weights for the xwOBA approximation (BB/HBP aren't batted balls, so
# Statcast's own xwOBA credits them at their real wOBA weights)
W_BB, W_HBP = 0.69, 0.72

_AB_EXCLUDED = ("walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
                "catcher_interf", "sac_fly_double_play", "sac_bunt_double_play")
_SWSTR = ("swinging_strike", "swinging_strike_blocked")


def _batter_history_cte(alias: str, days: int) -> str:
    """Per-batter aggregates over [slate_date − days, slate_date)."""
    ab_excluded = ", ".join(f"'{e}'" for e in _AB_EXCLUDED)
    return f"""
    {alias} AS (
        SELECT
            batter_id,
            COUNT(*)                                                    AS pa,
            SUM(is_hr)                                                  AS hr,
            SUM(is_batted_ball)                                         AS bip,
            SUM(CASE WHEN events NOT IN ({ab_excluded}) THEN 1 ELSE 0 END) AS ab,
            SUM(CASE WHEN events = 'single' THEN 1
                     WHEN events = 'double' THEN 2
                     WHEN events = 'triple' THEN 3
                     WHEN events = 'home_run' THEN 4 ELSE 0 END)        AS total_bases,
            SUM(CASE WHEN events IN ('single','double','triple','home_run')
                     THEN 1 ELSE 0 END)                                 AS hits,
            -- unintentional walks only: canonical wOBA excludes IBB from
            -- both numerator and denominator
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END)            AS bb,
            SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END)    AS hbp,
            SUM(CASE WHEN events IN ('sac_fly','sac_fly_double_play')
                     THEN 1 ELSE 0 END)                                 AS sf,
            SUM(CASE WHEN is_batted_ball = 1 THEN estimated_woba END)   AS xwoba_bbe_sum,
            SUM(CASE WHEN is_barrel THEN 1 ELSE 0 END)                  AS barrels,
            SUM(CASE WHEN is_barrel AND spray_angle_adj < -15
                     THEN 1 ELSE 0 END)                                 AS pulled_barrels,
            SUM(CASE WHEN launch_angle BETWEEN 8 AND 32
                      AND is_batted_ball = 1 THEN 1 ELSE 0 END)         AS sweet_spot,
            SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END)       AS fly_balls,
            SUM(CASE WHEN launch_speed >= 95 THEN 1 ELSE 0 END)         AS hard_hits,
            AVG(CASE WHEN is_batted_ball = 1 THEN launch_angle END)     AS la_avg,
            -- switch-hitters collapse to their modal side here; these are
            -- blended (not platoon-split) stats, unlike the training table
            MODE(stand)                                                 AS stand
        FROM plate_appearances
        WHERE game_date <  DATE '{{slate_date}}'
          AND game_date >= DATE '{{slate_date}}' - INTERVAL {days} DAYS
        GROUP BY batter_id
    )"""


def build_matchup_board(
    con: duckdb.DuckDBPyConnection,
    slate_date: str,
    scores_path: Path | None = None,
    export_path: Path | None = None,
):
    """
    Build the `matchup_board` table for one slate date. Returns the frame.
    `scores_path`: optional parquet of calibrated model output
    (batter_id, game_pk, hr_probability) to fill the hr_prob column.
    """
    from datetime import date as _date
    _date.fromisoformat(slate_date)   # validates before SQL interpolation

    swstr = ", ".join(f"'{d}'" for d in _SWSTR)
    has_schedule = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='schedule'"
    ).fetchone()[0] > 0
    if not has_schedule:
        raise RuntimeError("Silver `schedule` table required for the slate board")

    has_names = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='player_names'"
    ).fetchone()[0] > 0

    # slot lookup degrades gracefully when the Gold batter features haven't
    # been built (standalone `make board` against a fresh silver DB)
    has_bgr = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='batter_game_rolling'"
    ).fetchone()[0] > 0
    slot_join = (
        """LEFT JOIN (
                SELECT batter_id, ARG_MAX(b_avg_slot_30d, game_date) AS b_avg_slot_30d
                FROM batter_game_rolling
                WHERE game_date < DATE '{slate_date}'
                GROUP BY batter_id
            ) b ON b.batter_id = r.batter_id"""
        if has_bgr else
        "LEFT JOIN (SELECT NULL::BIGINT AS batter_id, NULL::DOUBLE AS b_avg_slot_30d) b "
        "ON b.batter_id = r.batter_id"
    )
    names_join = (
        "LEFT JOIN player_names nm ON nm.player_id = r.batter_id"
        if has_names else
        "LEFT JOIN (SELECT NULL::BIGINT AS player_id, NULL::VARCHAR AS full_name) nm "
        "ON nm.player_id = r.batter_id"
    )
    starter_names_join = (
        "LEFT JOIN player_names pnm ON pnm.player_id = g.opp_starter_id"
        if has_names else
        "LEFT JOIN (SELECT NULL::BIGINT AS player_id, NULL::VARCHAR AS full_name) pnm "
        "ON pnm.player_id = g.opp_starter_id"
    )

    if scores_path is not None:
        scores_join = (f"LEFT JOIN read_parquet('{scores_path}') sc "
                       "ON sc.batter_id = r.batter_id AND sc.game_pk = g.game_pk")
        hr_prob_expr = "ROUND(sc.hr_probability, 4)"
    else:
        scores_join = ""
        hr_prob_expr = "NULL::DOUBLE"

    sql = f"""
    CREATE OR REPLACE TABLE matchup_board AS

    WITH games AS (
        -- both sides of every scheduled game: the batting team + who they face
        SELECT game_pk, game_date, venue_name, day_night,
               home_team_abbr AS batter_team, away_team_abbr AS opp_team,
               away_probable_pitcher_id AS opp_starter_id, TRUE AS is_home
        FROM schedule WHERE game_date = DATE '{{slate_date}}'
        UNION ALL
        SELECT game_pk, game_date, venue_name, day_night,
               away_team_abbr, home_team_abbr,
               home_probable_pitcher_id, FALSE
        FROM schedule WHERE game_date = DATE '{{slate_date}}'
    ),

    -- roster proxy: recent PAs for the team, ordered by recent lineup slot.
    -- A traded batter is kept ONLY on his most-recent team (team_rank = 1).
    recent AS (
        SELECT * FROM (
            SELECT batter_id, batter_team,
                   COUNT(*) AS pa_recent,
                   MAX(game_date) AS last_seen,
                   ROW_NUMBER() OVER (
                       PARTITION BY batter_id
                       ORDER BY MAX(game_date) DESC, COUNT(*) DESC
                   ) AS team_rank
            FROM plate_appearances
            WHERE game_date <  DATE '{{slate_date}}'
              AND game_date >= DATE '{{slate_date}}' - INTERVAL {ROSTER_LOOKBACK_DAYS} DAYS
            GROUP BY batter_id, batter_team
        ) WHERE team_rank = 1
    ),
    roster AS (
        SELECT * FROM (
            SELECT r.*,
                   b.b_avg_slot_30d,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.batter_team
                       ORDER BY COALESCE(b.b_avg_slot_30d, 99), r.pa_recent DESC
                   ) AS roster_rank
            FROM recent r
            {slot_join}
        ) WHERE roster_rank <= {ROSTER_MAX_HITTERS}
    ),

    {_batter_history_cte("h365", 365)},
    {_batter_history_cte("h30", 30)},

    pitch365 AS (
        SELECT batter_id,
               COUNT(*) AS pit,
               SUM(CASE WHEN description IN ({swstr}) THEN 1 ELSE 0 END) AS swstr
        FROM pitches
        WHERE game_date <  DATE '{{slate_date}}'
          AND game_date >= DATE '{{slate_date}}' - INTERVAL 365 DAYS
        GROUP BY batter_id
    ),

    starter_hand AS (
        SELECT pitcher_id, MODE(p_throws) AS p_throws
        FROM plate_appearances
        WHERE game_date < DATE '{{slate_date}}'
        GROUP BY pitcher_id
    )

    SELECT
        g.game_pk, g.game_date, g.venue_name, g.day_night, g.is_home,
        r.batter_id,
        COALESCE(nm.full_name, 'Batter #' || r.batter_id)   AS batter_name,
        h365.stand,
        r.batter_team, g.opp_team,
        g.opp_starter_id,
        COALESCE(pnm.full_name,
                 CASE WHEN g.opp_starter_id IS NULL THEN 'TBD'
                      ELSE 'Pitcher #' || g.opp_starter_id END) AS opp_starter_name,
        sh.p_throws                                          AS opp_starter_throws,
        ROUND(r.b_avg_slot_30d, 1)                           AS recent_slot,

        {hr_prob_expr}                                       AS hr_prob,

        p365.pit                                             AS pit_365d,
        h365.bip                                             AS bip_365d,
        ROUND((h365.total_bases - h365.hits) * 1.0 / NULLIF(h365.ab, 0), 3)  AS iso_365d,
        -- wOBA denominator = AB + uBB + HBP + SF (sac flies count in the
        -- denominator even though they aren't at-bats)
        ROUND((h365.xwoba_bbe_sum + {W_BB} * h365.bb + {W_HBP} * h365.hbp)
              / NULLIF(h365.ab + h365.bb + h365.hbp + h365.sf, 0), 3)  AS xwoba_365d,
        ROUND(h365.xwoba_bbe_sum / NULLIF(h365.bip, 0), 3)   AS xwobacon_365d,
        ROUND(p365.swstr * 1.0 / NULLIF(p365.pit, 0), 3)     AS sws_365d,
        ROUND(h365.pulled_barrels * 1.0 / NULLIF(h365.bip, 0), 3) AS pbrl_365d,
        ROUND(h365.barrels * 1.0 / NULLIF(h365.bip, 0), 3)   AS brl_365d,
        ROUND(h365.sweet_spot * 1.0 / NULLIF(h365.bip, 0), 3) AS swsp_365d,
        ROUND(h365.fly_balls * 1.0 / NULLIF(h365.bip, 0), 3) AS fb_365d,
        ROUND(h365.hard_hits * 1.0 / NULLIF(h365.bip, 0), 3) AS hh_365d,
        ROUND(h365.la_avg, 1)                                AS la_365d,

        -- FORM: recent (30d) contact quality vs own 365d baseline,
        -- mapped so 50% = steady. NULL when either window is empty.
        ROUND(100.0 * (h30.xwoba_bbe_sum / NULLIF(h30.bip, 0))
              / NULLIF((h30.xwoba_bbe_sum / NULLIF(h30.bip, 0))
                       + (h365.xwoba_bbe_sum / NULLIF(h365.bip, 0)), 0), 0) AS form_pct,
        CASE
            WHEN COALESCE(h30.bip, 0) = 0 OR COALESCE(h365.bip, 0) = 0 THEN NULL
            WHEN (h30.xwoba_bbe_sum / NULLIF(h30.bip, 0))
                 > (1 + {FORM_STEADY_BAND}) * (h365.xwoba_bbe_sum / NULLIF(h365.bip, 0))
                 THEN '↑'
            WHEN (h30.xwoba_bbe_sum / NULLIF(h30.bip, 0))
                 < (1 - {FORM_STEADY_BAND}) * (h365.xwoba_bbe_sum / NULLIF(h365.bip, 0))
                 THEN '↓'
            ELSE '→'
        END                                                  AS form_arrow

    FROM games g
    JOIN roster r        ON r.batter_team = g.batter_team
    LEFT JOIN h365       ON h365.batter_id = r.batter_id
    LEFT JOIN h30        ON h30.batter_id = r.batter_id
    LEFT JOIN pitch365 p365 ON p365.batter_id = r.batter_id
    LEFT JOIN starter_hand sh ON sh.pitcher_id = g.opp_starter_id
    {names_join}
    {starter_names_join}
    {scores_join}
    ORDER BY g.game_pk, r.roster_rank;
    """.replace("{slate_date}", slate_date)

    con.execute(sql)
    df = con.execute("SELECT * FROM matchup_board").fetchdf()
    logger.info("matchup_board: %d hitter rows across %d games for %s",
                len(df), df["game_pk"].nunique() if len(df) else 0, slate_date)

    # Statcast and the Stats API can disagree on team abbreviations; a
    # mismatch makes the roster join return zero hitters SILENTLY. Surface it.
    scheduled = {t for row in con.execute(
        "SELECT home_team_abbr, away_team_abbr FROM schedule WHERE game_date = ?",
        [slate_date]).fetchall() for t in row if t}
    covered = set(df["batter_team"].unique()) if len(df) else set()
    uncovered = scheduled - covered
    if uncovered:
        logger.warning(
            "No roster rows for scheduled team(s) %s — check team-abbreviation "
            "mismatch between Statcast and the MLB Stats API, or missing recent PAs",
            sorted(uncovered))

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY matchup_board TO '{export_path}' "
                    f"(FORMAT PARQUET, COMPRESSION SNAPPY)")
        logger.info("Board written: %s", export_path)
    return df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Slate date YYYY-MM-DD")
    parser.add_argument("--scores", default=None,
                        help="Optional parquet with calibrated hr_probability per batter/game")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out = Path(args.output) if args.output else GOLD_DIR / f"matchup_board_{args.date}.parquet"
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        build_matchup_board(
            con, args.date,
            scores_path=Path(args.scores) if args.scores else None,
            export_path=out,
        )
