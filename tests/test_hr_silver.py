"""
tests/test_hr_silver.py
========================
Layer-1 tests for the HR-prop pipeline: plate-appearance grain and schedule
parsing. Fully offline — in-memory DuckDB and recorded JSON fixtures only.

Run:
    pytest tests/test_hr_silver.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline.ingestion.schedule_ingestion import parse_schedule_json
from pipeline.silver.plate_appearances import build_plate_appearances

FIXTURES = Path(__file__).parent / "fixtures"


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _pitch_row(game_pk, at_bat, pitch_no, batter, pitcher, stand, events=None, **kw):
    """One synthetic Statcast pitch row with sane defaults."""
    row = {
        "game_pk": game_pk, "game_date": "2024-07-01",
        "at_bat_number": at_bat, "pitch_number": pitch_no,
        "batter_id": batter, "pitcher_id": pitcher,
        "stand": stand, "p_throws": "R",
        "inning": 1, "inning_topbot": "Top",
        "home_team": "PHI", "away_team": "NYM",
        "events": events,
        "launch_speed": None, "launch_angle": None,
        "estimated_woba": None,
        "is_barrel": False, "bb_type": None,
        "hc_x": None, "hc_y": None,
    }
    row.update(kw)
    return row


@pytest.fixture
def con_with_pitches():
    """In-memory DuckDB with a tiny synthetic `pitches` table (3 PAs)."""
    rows = [
        # PA 1: three pitches, ends in a home run (RHB, barreled fly ball)
        _pitch_row(700001, 1, 1, 2001, 1001, "R"),
        _pitch_row(700001, 1, 2, 2001, 1001, "R"),
        _pitch_row(700001, 1, 3, 2001, 1001, "R", events="home_run",
                   launch_speed=105.3, launch_angle=28.0, is_barrel=True,
                   bb_type="fly_ball", hc_x=95.0, hc_y=40.0),
        # PA 2: two pitches, strikeout
        _pitch_row(700001, 2, 1, 2002, 1001, "L"),
        _pitch_row(700001, 2, 2, 2002, 1001, "L", events="strikeout"),
        # PA 3: single pitch, LHB pulls a ground out to the right side
        _pitch_row(700001, 3, 1, 2003, 1001, "L", events="field_out",
                   launch_speed=88.0, launch_angle=-5.0,
                   bb_type="ground_ball", hc_x=160.0, hc_y=140.0,
                   inning_topbot="Bot", home_team="PHI", away_team="NYM"),
    ]
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE pitches AS SELECT * FROM df")
    yield con
    con.close()


# ── Tests: plate_appearances ────────────────────────────────────────────────────

class TestPlateAppearances:

    def test_one_row_per_pa(self, con_with_pitches):
        """Grain: exactly one row per (game_pk, at_bat_number)."""
        build_plate_appearances(con_with_pitches)
        total, distinct = con_with_pitches.execute("""
            SELECT COUNT(*),
                   COUNT(DISTINCT game_pk::VARCHAR || '_' || at_bat_number::VARCHAR)
            FROM plate_appearances
        """).fetchone()
        assert total == 3
        assert total == distinct

    def test_is_hr_flag(self, con_with_pitches):
        """Only the home-run PA is flagged; outcome comes from the final pitch."""
        build_plate_appearances(con_with_pitches)
        flags = dict(con_with_pitches.execute(
            "SELECT batter_id, is_hr FROM plate_appearances"
        ).fetchall())
        assert flags == {2001: 1, 2002: 0, 2003: 0}

    def test_outcome_from_last_pitch(self, con_with_pitches):
        """The PA row must carry the final pitch's batted-ball data."""
        build_plate_appearances(con_with_pitches)
        ev, barrel = con_with_pitches.execute(
            "SELECT launch_speed, is_barrel FROM plate_appearances WHERE batter_id = 2001"
        ).fetchone()
        assert ev == pytest.approx(105.3, abs=0.01)
        assert barrel is True

    def test_batter_team_mapping(self, con_with_pitches):
        """Top of inning → away team bats; bottom → home team bats."""
        build_plate_appearances(con_with_pitches)
        teams = dict(con_with_pitches.execute(
            "SELECT batter_id, batter_team FROM plate_appearances"
        ).fetchall())
        assert teams[2001] == "NYM"   # Top 1st: away batting
        assert teams[2003] == "PHI"   # Bot 1st: home batting

    def test_spray_angle_pull_sign_convention(self, con_with_pitches):
        """Negative spray_angle_adj = pulled, for BOTH handednesses."""
        build_plate_appearances(con_with_pitches)
        # RHB HR to left field (hc_x < 125.42): pulled → negative
        rhb = con_with_pitches.execute(
            "SELECT spray_angle_adj FROM plate_appearances WHERE batter_id = 2001"
        ).fetchone()[0]
        # LHB grounder to right side (hc_x > 125.42): pulled → negative
        lhb = con_with_pitches.execute(
            "SELECT spray_angle_adj FROM plate_appearances WHERE batter_id = 2003"
        ).fetchone()[0]
        assert rhb < 0
        assert lhb < 0

    def test_non_batted_pa_has_null_spray(self, con_with_pitches):
        build_plate_appearances(con_with_pitches)
        spray, batted = con_with_pitches.execute(
            "SELECT spray_angle_adj, is_batted_ball FROM plate_appearances WHERE batter_id = 2002"
        ).fetchone()
        assert spray is None
        assert batted == 0


# ── Tests: schedule parsing ─────────────────────────────────────────────────────

class TestScheduleParsing:

    @pytest.fixture
    def schedule_df(self):
        payload = json.loads((FIXTURES / "mlb_schedule_sample.json").read_text())
        return parse_schedule_json(payload)

    def test_one_row_per_game(self, schedule_df):
        assert len(schedule_df) == 2
        assert schedule_df["game_pk"].is_unique

    def test_probable_pitchers_parsed(self, schedule_df):
        game = schedule_df[schedule_df["game_pk"] == 745804].iloc[0]
        assert game["home_probable_pitcher_id"] == 554430
        assert game["home_probable_pitcher_name"] == "Zack Wheeler"
        assert game["away_probable_pitcher_id"] == 656849

    def test_missing_probable_pitcher_is_null_not_dropped(self, schedule_df):
        """A TBD starter must not drop the game from the slate."""
        game = schedule_df[schedule_df["game_pk"] == 745805].iloc[0]
        assert pd.isna(game["home_probable_pitcher_id"])
        assert game["away_probable_pitcher_id"] == 678394

    def test_context_fields(self, schedule_df):
        by_pk = schedule_df.set_index("game_pk")
        assert by_pk.loc[745804, "day_night"] == "night"
        assert by_pk.loc[745805, "day_night"] == "day"
        assert by_pk.loc[745804, "venue_name"] == "Citizens Bank Park"
        assert by_pk.loc[745804, "home_team_abbr"] == "PHI"

    def test_empty_payload(self):
        df = parse_schedule_json({"dates": []})
        assert df.empty
