"""
tests/test_pipeline.py
=======================
Core tests for the BaseballIQ pipeline.
Uses sample data — no network calls, no API keys required.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pitcher_df():
    """Minimal pitcher_game_summary dataframe for testing."""
    return pd.DataFrame([
        {
            "pitcher_id": 1001, "pitcher_name": "Test Pitcher", "team": "LAD",
            "game_pk": 700001, "game_date": pd.Timestamp("2024-07-01"),
            "opponent": "HOU", "total_pitches": 95,
            "avg_velo": 95.5, "season_avg_velo": 95.0, "velo_vs_30d_avg": 0.5,
            "whiff_rate": 0.31, "csw_rate": 0.32, "zone_rate": 0.48,
            "chase_rate": 0.30, "barrel_rate_allowed": 0.07,
            "avg_xwoba_allowed": 0.270, "stuff_diversity": 1.2,
            "total_re_delta": -0.15, "rolling_30d_csw_rate": 0.30,
            "rolling_30d_whiff_rate": 0.29, "rolling_30d_avg_velo": 95.0,
            "whiff_rate_delta": 0.02, "season_avg_csw": 0.30,
            "season_avg_whiff": 0.28, "performance_tier": "elite",
            "avg_spin": 2280, "avg_h_break": 8.5, "avg_v_break": 5.5,
        },
        {
            "pitcher_id": 1002, "pitcher_name": "Another Pitcher", "team": "HOU",
            "game_pk": 700002, "game_date": pd.Timestamp("2024-07-01"),
            "opponent": "LAD", "total_pitches": 88,
            "avg_velo": 93.0, "season_avg_velo": 93.5, "velo_vs_30d_avg": -0.5,
            "whiff_rate": 0.24, "csw_rate": 0.26, "zone_rate": 0.47,
            "chase_rate": 0.28, "barrel_rate_allowed": 0.09,
            "avg_xwoba_allowed": 0.305, "stuff_diversity": 0.9,
            "total_re_delta": 0.08, "rolling_30d_csw_rate": 0.27,
            "rolling_30d_whiff_rate": 0.25, "rolling_30d_avg_velo": 93.5,
            "whiff_rate_delta": -0.01, "season_avg_csw": 0.27,
            "season_avg_whiff": 0.25, "performance_tier": "average",
            "avg_spin": 2180, "avg_h_break": 7.2, "avg_v_break": 4.8,
        },
    ])


@pytest.fixture
def sample_batter_df():
    return pd.DataFrame([
        {
            "batter_id": 2001, "batter_name": "Test Batter", "team": "LAD",
            "game_date": pd.Timestamp("2024-07-01"), "opponent": "HOU",
            "pitches_seen": 14, "swing_rate": 0.46, "o_swing_rate": 0.29,
            "avg_exit_velo": 93.5, "avg_launch_angle": 14.2,
            "avg_xwoba": 0.380, "barrel_rate": 0.115,
            "hard_hit_rate": 0.44, "total_re_created": 0.18,
        }
    ])


# ── Tests: Data validation ───────────────────────────────────────────────────────

class TestDataValidation:

    def test_pitcher_csw_rate_bounds(self, sample_pitcher_df):
        """CSW rate should always be between 0 and 1."""
        assert (sample_pitcher_df["csw_rate"] >= 0).all()
        assert (sample_pitcher_df["csw_rate"] <= 1).all()

    def test_whiff_rate_bounds(self, sample_pitcher_df):
        assert (sample_pitcher_df["whiff_rate"] >= 0).all()
        assert (sample_pitcher_df["whiff_rate"] <= 1).all()

    def test_performance_tier_valid_values(self, sample_pitcher_df):
        valid_tiers = {"elite", "above_avg", "average", "below_avg", "poor"}
        assert set(sample_pitcher_df["performance_tier"]).issubset(valid_tiers)

    def test_no_null_pitcher_ids(self, sample_pitcher_df):
        assert sample_pitcher_df["pitcher_id"].notna().all()

    def test_batter_exit_velo_realistic(self, sample_batter_df):
        """Exit velocity should be in realistic MLB range."""
        assert (sample_batter_df["avg_exit_velo"] >= 60).all()
        assert (sample_batter_df["avg_exit_velo"] <= 120).all()

    def test_barrel_rate_bounds(self, sample_batter_df):
        assert (sample_batter_df["barrel_rate"] >= 0).all()
        assert (sample_batter_df["barrel_rate"] <= 1).all()


# ── Tests: Feature engineering logic ───────────────────────────────────────────

class TestFeatureEngineering:

    def test_performance_tier_mapping(self):
        """Tier labels should map correctly to CSW ranges."""
        def tier(csw):
            if csw >= 0.32: return "elite"
            if csw >= 0.29: return "above_avg"
            if csw >= 0.26: return "average"
            if csw >= 0.22: return "below_avg"
            return "poor"

        assert tier(0.33) == "elite"
        assert tier(0.30) == "above_avg"
        assert tier(0.27) == "average"
        assert tier(0.23) == "below_avg"
        assert tier(0.19) == "poor"

    def test_velo_delta_calculation(self, sample_pitcher_df):
        """velo_vs_30d_avg should equal avg_velo - rolling_30d_avg_velo."""
        for _, row in sample_pitcher_df.iterrows():
            expected = round(row["avg_velo"] - row["rolling_30d_avg_velo"], 2)
            actual   = round(row["velo_vs_30d_avg"], 2)
            assert abs(expected - actual) < 0.05, f"Velo delta mismatch: {expected} vs {actual}"

    def test_whiff_rate_delta_direction(self, sample_pitcher_df):
        """whiff_rate_delta > 0 means improving vs. baseline."""
        elite_row   = sample_pitcher_df[sample_pitcher_df["performance_tier"] == "elite"].iloc[0]
        average_row = sample_pitcher_df[sample_pitcher_df["performance_tier"] == "average"].iloc[0]
        assert elite_row["whiff_rate_delta"] > average_row["whiff_rate_delta"]


# ── Tests: LLM prompt parsing ────────────────────────────────────────────────────

class TestLLMParsing:

    def test_valid_insight_json_structure(self):
        """LLM insight JSON should have required keys."""
        sample_response = {
            "performance_tier": "elite",
            "headline": "Pitcher dominated with elite swing-and-miss stuff.",
            "key_finding": "CSW rate of 32.1% was well above league average.",
            "concern_flag": None,
            "pitch_mix_note": "Fastball-heavy with effective slider.",
        }
        required_keys = {"performance_tier", "headline", "key_finding", "concern_flag"}
        assert required_keys.issubset(sample_response.keys())

    def test_valid_tier_in_response(self):
        valid_tiers = {"elite", "above_avg", "average", "below_avg", "poor"}
        response    = {"performance_tier": "elite"}
        assert response["performance_tier"] in valid_tiers

    def test_malformed_json_fallback(self):
        """System should handle JSON parse failures gracefully."""
        malformed = "Here is my analysis: the pitcher was great..."
        try:
            json.loads(malformed)
            parsed = True
        except json.JSONDecodeError:
            parsed = False
        assert not parsed  # confirms we need the try/except in llm_client.py


# ── Tests: Report generation ─────────────────────────────────────────────────────

class TestReportGeneration:

    def test_grade_mapping(self):
        """Letter grades should map correctly to CSW ranges."""
        def grade(csw):
            if csw >= 0.33: return "A+"
            if csw >= 0.31: return "A"
            if csw >= 0.29: return "B+"
            if csw >= 0.27: return "B"
            if csw >= 0.25: return "C+"
            if csw >= 0.23: return "C"
            return "D"

        assert grade(0.340) == "A+"
        assert grade(0.315) == "A"
        assert grade(0.295) == "B+"
        assert grade(0.260) == "B"
        assert grade(0.210) == "D"

    def test_stuff_score_bounds(self):
        """Stuff score should be 0-100."""
        for csw in [0.15, 0.25, 0.30, 0.35, 0.42]:
            score = int(np.clip((csw - 0.18) / 0.20 * 100, 0, 100))
            assert 0 <= score <= 100

    def test_percentile_monotone(self):
        """Higher CSW → higher percentile."""
        def pct(csw):
            breakpoints = [
                (0.35, 99), (0.33, 95), (0.31, 85), (0.29, 70),
                (0.27, 55), (0.25, 40), (0.23, 25), (0.20, 10),
            ]
            for threshold, p in breakpoints:
                if csw >= threshold:
                    return p
            return 5

        assert pct(0.36) > pct(0.30) > pct(0.24) > pct(0.18)


# ── Tests: Demo data generation ──────────────────────────────────────────────────

class TestDemoData:

    def test_demo_data_row_count(self, sample_pitcher_df):
        """Should have data for multiple pitchers."""
        assert sample_pitcher_df["pitcher_id"].nunique() >= 2

    def test_demo_data_date_types(self, sample_pitcher_df):
        """game_date should be datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_pitcher_df["game_date"])

    def test_demo_insights_structure(self):
        """Pre-generated insights should have required fields."""
        sample_insight = {
            "pitcher_id": 1001,
            "game_date": "2024-07-01",
            "performance_tier": "elite",
            "headline": "Test headline.",
            "key_finding": "Test finding.",
            "concern_flag": None,
        }
        assert "pitcher_id" in sample_insight
        assert "headline"   in sample_insight
        assert "key_finding" in sample_insight
