"""
tests/test_matchup_board.py
============================
Layer-6 tests: the slate matchup board. Synthetic Silver tables, fully
offline. Pins down: the as-of-slate-date cutoff (no same-day or future
data), the Statcast column math (ISO, SwS%, sweet spot), the FORM arrow,
roster proxy + probable-starter wiring, and the optional model-score join.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.gold.matchup_board import build_matchup_board

SLATE = "2024-06-10"


def _pa(date, ab, batter, team, events, ls=None, la=None, xwoba=None,
        barrel=False, bb_type=None, spray=None, pitcher=90, p_throws="R"):
    home, away = "NYY", "BOS"
    return {
        "game_pk": int(date.replace("-", "")), "game_date": date,
        "at_bat_number": ab, "batter_id": batter, "pitcher_id": pitcher,
        "stand": "R", "p_throws": p_throws,
        "inning": 1, "inning_topbot": "Bot" if team == home else "Top",
        "home_team": home, "away_team": away,
        "batter_team": team,
        "pitching_team": away if team == home else home,
        "events": events, "is_hr": int(events == "home_run"),
        "is_batted_ball": 1 if ls is not None else 0,
        "launch_speed": ls, "launch_angle": la, "estimated_woba": xwoba,
        "is_barrel": barrel, "bb_type": bb_type, "spray_angle_adj": spray,
    }


def _pitch(date, batter, description):
    return {"game_date": date, "batter_id": batter, "description": description}


@pytest.fixture
def con():
    """
    Batter 10 (NYY): 2024-06-01 — HR (barrel, pulled, sweet spot), single,
    two strikeouts → AB=4, TB=5, hits=2 → ISO=0.75. Ten pitches seen,
    3 swinging strikes → SwS% = 0.30.
    A PA and pitches ON the slate date must be excluded by the cutoff.

    Batter 20 (BOS): weak contact 300 days ago (xwOBAcon .200), strong five
    days ago (.800) → FORM arrow ↑.
    """
    pas = [
        _pa("2024-06-01", 1, 10, "NYY", "home_run", ls=106.0, la=27.0,
            xwoba=1.95, barrel=True, bb_type="fly_ball", spray=-20.0),
        _pa("2024-06-01", 3, 10, "NYY", "single", ls=90.0, la=12.0,
            xwoba=0.55, bb_type="line_drive", spray=0.0),
        _pa("2024-06-01", 5, 10, "NYY", "strikeout"),
        _pa("2024-06-01", 7, 10, "NYY", "strikeout"),
        # slate-day PA — must NOT appear in any board number
        _pa(SLATE, 1, 10, "NYY", "home_run", ls=110.0, la=30.0,
            xwoba=2.0, barrel=True, bb_type="fly_ball", spray=-20.0),
        # batter 20: old cold, recent hot
        _pa("2023-08-15", 2, 20, "BOS", "field_out", ls=82.0, la=45.0,
            xwoba=0.20, bb_type="popup", spray=5.0),
        _pa("2024-06-05", 2, 20, "BOS", "double", ls=101.0, la=18.0,
            xwoba=0.80, bb_type="line_drive", spray=-10.0),
        # batter 30 (NYY): single + sac fly + walk → xwOBA denominator must
        # include the sac fly (regression: gate finding #1)
        _pa("2024-06-02", 1, 30, "NYY", "single", ls=95.0, la=10.0,
            xwoba=0.90, bb_type="line_drive", spray=0.0),
        _pa("2024-06-02", 3, 30, "NYY", "sac_fly", ls=97.0, la=35.0,
            xwoba=0.90, bb_type="fly_ball", spray=5.0),
        _pa("2024-06-02", 5, 30, "NYY", "walk"),
        # batter 40 (BOS): old BIP baseline, recent PAs but ZERO recent BIP
        # → FORM arrow must be NULL, not '→' (regression: gate finding #2)
        _pa("2023-08-20", 4, 40, "BOS", "field_out", ls=90.0, la=20.0,
            xwoba=0.50, bb_type="fly_ball", spray=0.0),
        _pa("2024-06-05", 4, 40, "BOS", "strikeout"),
        # batter 50: traded — BOS on 06-01, NYY on 06-05 → must appear only
        # on NYY's side (regression: gate finding #3)
        _pa("2024-06-01", 9, 50, "BOS", "single", ls=91.0, la=9.0,
            xwoba=0.60, bb_type="ground_ball", spray=-5.0),
        _pa("2024-06-05", 9, 50, "NYY", "single", ls=92.0, la=11.0,
            xwoba=0.62, bb_type="line_drive", spray=-3.0),
    ]
    pitches = (
        [_pitch("2024-06-01", 10, "swinging_strike")] * 3
        + [_pitch("2024-06-01", 10, "foul")] * 3
        + [_pitch("2024-06-01", 10, "ball")] * 4
        + [_pitch(SLATE, 10, "swinging_strike")] * 5      # slate-day: excluded
        + [_pitch("2024-06-05", 20, "hit_into_play")]
    )
    schedule = [{
        "game_pk": 5001, "game_date": SLATE, "game_datetime_utc": f"{SLATE}T17:05:00Z",
        "day_night": "day", "game_type": "R", "status": "Scheduled",
        "venue_id": 3313, "venue_name": "Yankee Stadium",
        "home_team_id": 147, "home_team_name": "New York Yankees", "home_team_abbr": "NYY",
        "away_team_id": 111, "away_team_name": "Boston Red Sox", "away_team_abbr": "BOS",
        "home_probable_pitcher_id": 82, "home_probable_pitcher_name": "Home Starter",
        "away_probable_pitcher_id": 91, "away_probable_pitcher_name": "Away Starter",
    }]
    names = [
        {"player_id": 10, "full_name": "Aaron Judge", "bats": "R", "throws": "R"},
        {"player_id": 20, "full_name": "Rafael Devers", "bats": "L", "throws": "R"},
        {"player_id": 91, "full_name": "Away Starter", "bats": "L", "throws": "L"},
    ]

    pa_df = pd.DataFrame(pas)
    pa_df["game_date"] = pd.to_datetime(pa_df["game_date"]).dt.date
    pitch_df = pd.DataFrame(pitches)
    pitch_df["game_date"] = pd.to_datetime(pitch_df["game_date"]).dt.date
    sched_df = pd.DataFrame(schedule)
    sched_df["game_date"] = pd.to_datetime(sched_df["game_date"]).dt.date
    names_df = pd.DataFrame(names)

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE plate_appearances AS SELECT * FROM pa_df")
    con.execute("CREATE TABLE pitches AS SELECT * FROM pitch_df")
    con.execute("CREATE TABLE schedule AS SELECT * FROM sched_df")
    con.execute("CREATE TABLE player_names AS SELECT * FROM names_df")
    # minimal batter_game_rolling for the recent-slot lookup
    con.execute("""
        CREATE TABLE batter_game_rolling AS
        SELECT 10 AS batter_id, DATE '2024-06-01' AS game_date, 2.0 AS b_avg_slot_30d
        UNION ALL SELECT 20, DATE '2024-06-05', 3.0
    """)
    yield con
    con.close()


def _judge(df):
    rows = df[df["batter_id"] == 10]
    assert len(rows) == 1
    return rows.iloc[0]


class TestAsOfCutoff:

    def test_slate_day_data_excluded(self, con):
        """PAs and pitches ON the slate date never reach the board."""
        judge = _judge(build_matchup_board(con, SLATE))
        assert judge["pit_365d"] == 10            # not 15
        assert judge["bip_365d"] == 2             # not 3
        assert judge["iso_365d"] == pytest.approx(0.75)   # slate-day HR excluded


class TestColumnMath:

    def test_iso_sws_sweetspot(self, con):
        judge = _judge(build_matchup_board(con, SLATE))
        assert judge["iso_365d"] == pytest.approx((5 - 2) / 4)   # TB−H over AB
        assert judge["sws_365d"] == pytest.approx(0.30)          # 3 swstr / 10
        assert judge["swsp_365d"] == pytest.approx(1.0)          # LA 27 & 12 both 8–32
        assert judge["brl_365d"] == pytest.approx(0.5)
        assert judge["pbrl_365d"] == pytest.approx(0.5)          # the pulled HR
        assert judge["fb_365d"] == pytest.approx(0.5)
        assert judge["hh_365d"] == pytest.approx(0.5)            # 106 ≥ 95, 90 < 95
        assert judge["xwobacon_365d"] == pytest.approx((1.95 + 0.55) / 2, abs=1e-3)

    def test_form_arrow_hot_batter(self, con):
        df = build_matchup_board(con, SLATE)
        devers = df[df["batter_id"] == 20].iloc[0]
        assert devers["form_arrow"] == "↑"        # .800 recent vs .500 baseline
        assert devers["form_pct"] > 50


class TestMatchupWiring:

    def test_roster_faces_opposing_probable(self, con):
        df = build_matchup_board(con, SLATE)
        judge = _judge(df)
        assert judge["batter_name"] == "Aaron Judge"
        assert judge["opp_starter_id"] == 91      # NYY hitters face BOS probable
        assert judge["opp_starter_name"] == "Away Starter"
        assert judge["opp_starter_throws"] is None or judge["opp_starter_throws"] == "L"
        devers = df[df["batter_id"] == 20].iloc[0]
        assert devers["opp_starter_id"] == 82

    def test_scores_join_optional(self, con, tmp_path):
        df = build_matchup_board(con, SLATE)
        assert df["hr_prob"].isna().all()         # no scores file → NULL, not error

        scores = pd.DataFrame([{"batter_id": 10, "game_pk": 5001,
                                "hr_probability": 0.123}])
        scores_path = tmp_path / "scores.parquet"
        scores.to_parquet(scores_path, index=False)
        df2 = build_matchup_board(con, SLATE, scores_path=scores_path)
        assert _judge(df2)["hr_prob"] == pytest.approx(0.123)

    def test_bad_slate_date_rejected(self, con):
        with pytest.raises(ValueError):
            build_matchup_board(con, "2024-06-10'; DROP TABLE pitches;--")

    def test_missing_batter_game_rolling_tolerated(self, con):
        """Standalone board build against a fresh silver DB: slot is NULL,
        board still renders (regression: gate finding #4)."""
        con.execute("DROP TABLE batter_game_rolling")
        df = build_matchup_board(con, SLATE)
        assert len(df) > 0
        assert df["recent_slot"].isna().all()


class TestGateRegressions:

    def test_xwoba_denominator_includes_sac_fly(self, con):
        """single(.9) + sac_fly(.9) + walk → (0.9+0.9+0.69)/(1 AB+1 BB+1 SF)."""
        df = build_matchup_board(con, SLATE)
        row = df[df["batter_id"] == 30].iloc[0]
        assert row["xwoba_365d"] == pytest.approx((0.9 + 0.9 + 0.69) / 3, abs=1e-3)

    def test_form_arrow_null_without_recent_bip(self, con):
        df = build_matchup_board(con, SLATE)
        row = df[df["batter_id"] == 40].iloc[0]
        assert pd.isna(row["form_arrow"])
        assert pd.isna(row["form_pct"])

    def test_traded_player_only_on_latest_team(self, con):
        df = build_matchup_board(con, SLATE)
        rows = df[df["batter_id"] == 50]
        assert len(rows) == 1
        assert rows.iloc[0]["batter_team"] == "NYY"
        assert rows.iloc[0]["opp_starter_id"] == 91   # faces BOS's probable
