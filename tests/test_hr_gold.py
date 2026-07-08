"""
tests/test_hr_gold.py
======================
Layer-2 tests for the HR-prop Gold layer, against a synthetic two-game
scenario in in-memory DuckDB (fully offline).

Scenario: NYY hosts BOS on 2024-06-01 (game 1) and 2024-06-02 (game 2).
    - Batter 10 (NYY, LHB): game 1 → 4 PAs, 1 HR (a pulled barreled fly
      ball off the RHP starter); game 2 → 3 PAs, 0 HR vs a LHP starter.
    - Batter 20 (BOS, RHB): 2 PAs in game 1 (one HR off a NYY *reliever*),
      1 PA in game 2.
    - Game 1 starters: BOS pitcher 90 (R), NYY pitcher 80 (R).
      Relievers: BOS 92 (L), NYY 81 (R, allows the bullpen HR).
    - Game 2 starters: BOS pitcher 91 (L), NYY pitcher 82 (R).

The assertions pin down the invariants that matter:
    1. rolling windows exclude the current game (the leakage rule)
    2. rates are sums-over-sums
    3. platoon feature follows the opposing starter's hand
    4. bullpen rates exclude the starter
    5. park factor resolves by batter handedness
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.gold.matchup_table import run_all as build_hr_gold

D1, D2 = "2024-06-01", "2024-06-02"


def _pa(game_pk, date, ab, batter, pitcher, stand, p_throws, batter_team,
        events="field_out", is_hr=0, ls=None, la=None, barrel=False,
        bb_type=None, spray=None):
    home, away = "NYY", "BOS"
    return {
        "game_pk": game_pk, "game_date": date, "at_bat_number": ab,
        "batter_id": batter, "pitcher_id": pitcher,
        "stand": stand, "p_throws": p_throws,
        "inning": 1, "inning_topbot": "Bot" if batter_team == home else "Top",
        "home_team": home, "away_team": away,
        "batter_team": batter_team,
        "pitching_team": away if batter_team == home else home,
        "events": events, "is_hr": is_hr,
        "is_batted_ball": 1 if ls is not None else 0,
        "launch_speed": ls, "launch_angle": la, "is_barrel": barrel,
        "bb_type": bb_type, "spray_angle_adj": spray,
    }


@pytest.fixture
def con():
    rows = [
        # ── Game 1 (pk 1001) ─────────────────────────────────────────────
        # NYY batting (batter 10, LHB) vs BOS staff
        _pa(1001, D1, 1, 10, 90, "L", "R", "NYY", events="home_run", is_hr=1,
            ls=106.0, la=27.0, barrel=True, bb_type="fly_ball", spray=-20.0),
        _pa(1001, D1, 3, 10, 90, "L", "R", "NYY",
            ls=90.0, la=5.0, bb_type="ground_ball", spray=-10.0),
        _pa(1001, D1, 5, 10, 92, "L", "L", "NYY",
            ls=85.0, la=12.0, bb_type="line_drive", spray=5.0),
        _pa(1001, D1, 7, 10, 92, "L", "L", "NYY", events="strikeout"),
        # BOS batting (batter 20, RHB) vs NYY staff: starter 80, reliever 81
        _pa(1001, D1, 2, 20, 80, "R", "R", "BOS",
            ls=92.0, la=10.0, bb_type="line_drive", spray=-5.0),
        _pa(1001, D1, 6, 20, 81, "R", "R", "BOS", events="home_run", is_hr=1,
            ls=103.0, la=30.0, barrel=True, bb_type="fly_ball", spray=-15.5),

        # ── Game 2 (pk 1002) ─────────────────────────────────────────────
        # NYY batting vs BOS starter 91 (LHP)
        _pa(1002, D2, 1, 10, 91, "L", "L", "NYY",
            ls=88.0, la=20.0, bb_type="fly_ball", spray=10.0),
        _pa(1002, D2, 3, 10, 91, "L", "L", "NYY", events="strikeout"),
        _pa(1002, D2, 5, 10, 91, "L", "L", "NYY",
            ls=95.0, la=8.0, bb_type="ground_ball", spray=-12.0),
        # BOS batting vs NYY starter 82
        _pa(1002, D2, 2, 20, 82, "R", "R", "BOS",
            ls=90.0, la=15.0, bb_type="line_drive", spray=0.0),
    ]
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE plate_appearances AS SELECT * FROM df")
    build_hr_gold(con)
    yield con
    con.close()


def _row(con, sql):
    df = con.execute(sql).fetchdf()
    assert len(df) == 1, f"expected 1 row, got {len(df)}"
    return df.iloc[0]


class TestLeakageBoundary:

    def test_first_game_has_no_rolling_features(self, con):
        """No prior history → rolling features must be NULL, never 0."""
        r = _row(con, """
            SELECT b_hr_per_pa_30d, b_barrel_rate_30d, b_avg_ev_30d
            FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1001
        """)
        assert pd.isna(r.b_hr_per_pa_30d)
        assert pd.isna(r.b_barrel_rate_30d)
        assert pd.isna(r.b_avg_ev_30d)

    def test_rolling_window_is_strictly_prior(self, con):
        """Game-2 features reflect ONLY game 1 — never game 2 itself."""
        r = _row(con, """
            SELECT b_hr_per_pa_30d, b_pa_30d, b_barrel_rate_30d,
                   b_fb_rate_30d, b_pull_fb_rate_30d, b_avg_ev_30d,
                   b_max_ev_30d, b_avg_pa_per_game_30d
            FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1002
        """)
        assert r.b_pa_30d == 4
        assert r.b_hr_per_pa_30d == pytest.approx(0.25)         # 1 HR / 4 PA, game 1 only
        assert r.b_barrel_rate_30d == pytest.approx(1 / 3, abs=1e-4)
        assert r.b_fb_rate_30d == pytest.approx(1 / 3, abs=1e-4)
        assert r.b_pull_fb_rate_30d == pytest.approx(1.0)       # the HR was pulled
        assert r.b_avg_ev_30d == pytest.approx((106 + 90 + 85) / 3, abs=0.01)
        assert r.b_max_ev_30d == pytest.approx(106.0)
        assert r.b_avg_pa_per_game_30d == pytest.approx(4.0)

    def test_target_is_this_games_outcome(self, con):
        targets = dict(con.execute("""
            SELECT game_pk, hr_hit FROM batter_matchup_features WHERE batter_id = 10
        """).fetchall())
        assert targets == {1001: 1, 1002: 0}


class TestMatchupAssembly:

    def test_grain_one_row_per_batter_game(self, con):
        total, distinct = con.execute("""
            SELECT COUNT(*),
                   COUNT(DISTINCT batter_id::VARCHAR || '_' || game_pk::VARCHAR)
            FROM batter_matchup_features
        """).fetchone()
        assert total == 4
        assert total == distinct

    def test_opposing_starter_identity(self, con):
        r = _row(con, """
            SELECT opp_starter_id, opp_starter_throws, same_handed
            FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1002
        """)
        assert r.opp_starter_id == 91          # BOS game-2 starter, not reliever
        assert r.opp_starter_throws == "L"
        assert r.same_handed == 1              # LHB vs LHP

    def test_platoon_feature_follows_starter_hand(self, con):
        """Game-2 starter is LHP → feature = batter's HR rate vs L (0/2)."""
        r = _row(con, """
            SELECT b_hr_per_pa_vs_hand_365d
            FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1002
        """)
        assert r.b_hr_per_pa_vs_hand_365d == pytest.approx(0.0)

    def test_bullpen_rate_excludes_starter(self, con):
        """NYY bullpen in game 1 = reliever 81 only: 1 PA, 1 HR → 1.0."""
        r = _row(con, """
            SELECT bp_hr_per_pa_allowed_30d
            FROM batter_matchup_features
            WHERE batter_id = 20 AND game_pk = 1002
        """)
        assert r.bp_hr_per_pa_allowed_30d == pytest.approx(1.0)

    def test_starter_rolling_features_prior_games_only(self, con):
        """Starter 91 never pitched before game 2 → rolling NULL."""
        r = _row(con, """
            SELECT p_hr_per_pa_allowed_30d
            FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1002
        """)
        assert pd.isna(r.p_hr_per_pa_allowed_30d)


class TestShrinkageOnReturn:

    def test_shrunk_rate_collapses_to_long_rate_after_gap(self):
        """
        A batter returning after a >30-day gap has an empty 30d window.
        The shrunk rate must collapse to the 365d rate — not go NULL.
        (Regression: pipeline-reviewer gate finding #1.)
        """
        rows = [
            _pa(2001, "2024-04-01", 1, 30, 90, "R", "R", "NYY",
                events="home_run", is_hr=1, ls=104.0, la=28.0,
                barrel=True, bb_type="fly_ball", spray=-18.0),
            _pa(2001, "2024-04-01", 3, 30, 90, "R", "R", "NYY", events="strikeout"),
            # 61 days later — outside every short window
            _pa(2002, "2024-06-01", 1, 30, 91, "R", "L", "NYY", events="strikeout"),
        ]
        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE plate_appearances AS SELECT * FROM df")
        build_hr_gold(con)
        r = _row(con, """
            SELECT b_hr_per_pa_30d, b_hr_per_pa_365d, b_hr_per_pa_shrunk
            FROM batter_matchup_features
            WHERE batter_id = 30 AND game_pk = 2002
        """)
        con.close()
        assert pd.isna(r.b_hr_per_pa_30d)                       # no recent data: NULL is correct
        assert r.b_hr_per_pa_365d == pytest.approx(0.5)         # 1 HR / 2 PA
        assert r.b_hr_per_pa_shrunk == pytest.approx(0.5)       # collapses to 365d rate


class TestGameContext:

    def test_park_factor_by_handedness(self, con):
        """LHB at Yankee Stadium gets the LHB factor; RHB the RHB factor."""
        lhb = _row(con, """
            SELECT park_hr_factor_hand FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1001
        """)
        rhb = _row(con, """
            SELECT park_hr_factor_hand FROM batter_matchup_features
            WHERE batter_id = 20 AND game_pk = 1001
        """)
        assert lhb.park_hr_factor_hand > rhb.park_hr_factor_hand

    def test_is_home_flag(self, con):
        flags = dict(con.execute("""
            SELECT batter_id, is_home FROM batter_matchup_features WHERE game_pk = 1001
        """).fetchall())
        assert flags == {10: 1, 20: 0}

    def test_missing_schedule_and_weather_degrade_to_null(self, con):
        """No schedule/game_weather tables in this fixture → NULLs, not errors."""
        r = _row(con, """
            SELECT is_night, temp_f FROM batter_matchup_features
            WHERE batter_id = 10 AND game_pk = 1001
        """)
        assert pd.isna(r.is_night)
        assert pd.isna(r.temp_f)
