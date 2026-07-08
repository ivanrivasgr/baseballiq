"""
tests/test_odds.py
===================
Layer-4 tests: odds math round-trips, devig, The Odds API adapter (against
recorded fixtures + a fake HTTP session — never live quota), edge math,
parlays, and Voss Edge tier assignment. Fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from betting.edge import build_prop_edges, market_probabilities, normalize_name
from betting.odds import (
    american_to_decimal,
    american_to_implied,
    decimal_to_american,
    devig_one_sided,
    devig_two_way,
    overround,
)
from betting.odds_loader import (
    PROP_COLUMNS,
    QuotaFloorReached,
    TheOddsAPISource,
    parse_event_odds_json,
)
from betting.parlay import Leg, combine
from betting.tiers import TierInputs, assign_tier, load_rubric

FIXTURES = Path(__file__).parent / "fixtures"
EVENT_PAYLOAD = json.loads((FIXTURES / "the_odds_api_event_sample.json").read_text())


# ── Odds math ───────────────────────────────────────────────────────────────────

class TestOddsMath:

    @pytest.mark.parametrize("american", [320, -450, 150, -120, 100, 10000])
    def test_american_decimal_round_trip(self, american):
        assert decimal_to_american(american_to_decimal(american)) == pytest.approx(
            american, abs=1e-9)

    def test_even_money_boundary_canonicalizes(self):
        """-100 and +100 are the same price (decimal 2.0 → +100 canonical)."""
        assert american_to_decimal(-100) == american_to_decimal(100) == 2.0
        assert decimal_to_american(2.0) == 100

    @pytest.mark.parametrize("bad", [0, 50, -50, 99, -99])
    def test_invalid_american_rejected(self, bad):
        with pytest.raises(ValueError):
            american_to_decimal(bad)

    def test_hand_checked_two_way_devig(self):
        """-120 / +100 two-way: fair probs must sum to 1, favorite ≈ .5217."""
        p_yes = american_to_implied(-120)   # 0.545454...
        p_no = american_to_implied(100)     # 0.5
        fair_yes, fair_no = devig_two_way(p_yes, p_no)
        assert fair_yes + fair_no == pytest.approx(1.0)
        assert fair_yes == pytest.approx(0.545454 / 1.045454, abs=1e-4)
        assert overround(p_yes, p_no) == pytest.approx(0.045454, abs=1e-4)

    def test_one_sided_haircut_reduces_probability(self):
        implied = american_to_implied(450)
        assert devig_one_sided(implied) < implied


# ── The Odds API adapter ────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload, remaining):
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {"x-requests-remaining": str(remaining)}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """Routes The Odds API URLs to fixtures; counts real 'HTTP' calls."""

    def __init__(self, remaining=500):
        self.calls = 0
        self.remaining = remaining

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        self.remaining -= 1
        if url.endswith("/events"):
            payload = [{"id": EVENT_PAYLOAD["id"],
                        "commence_time": EVENT_PAYLOAD["commence_time"],
                        "home_team": EVENT_PAYLOAD["home_team"],
                        "away_team": EVENT_PAYLOAD["away_team"]}]
        else:
            payload = EVENT_PAYLOAD
        return FakeResponse(payload, self.remaining)


class TestTheOddsAPISource:

    def _source(self, tmp_path, session):
        return TheOddsAPISource(api_key="test-key", cache_dir=tmp_path / "odds",
                                quota_floor=25, session=session)

    def test_fetch_parses_hr_market_only(self, tmp_path):
        session = FakeSession()
        df = self._source(tmp_path, session).fetch_hr_props("2025-07-04")
        assert list(df.columns) == PROP_COLUMNS
        assert len(df) == 3                      # 2 Schwarber quotes + 1 Soto
        assert set(df["side"]) == {"Yes", "No"}  # totals market ignored
        assert session.calls == 2                # events + 1 event odds

    def test_second_fetch_is_fully_cached(self, tmp_path):
        session = FakeSession()
        source = self._source(tmp_path, session)
        source.fetch_hr_props("2025-07-04")
        calls_after_first = session.calls
        source.fetch_hr_props("2025-07-04")
        assert session.calls == calls_after_first   # zero new HTTP calls

    def test_raw_response_cached_before_parse(self, tmp_path):
        session = FakeSession()
        self._source(tmp_path, session).fetch_hr_props("2025-07-04")
        cached = list((tmp_path / "odds" / "date=2025-07-04").glob("*.json"))
        assert len(cached) == 2                  # events + event odds, raw JSON

    def test_quota_floor_hard_stops(self, tmp_path):
        session = FakeSession(remaining=26)      # events call → remaining 25
        source = self._source(tmp_path, session)
        with pytest.raises(QuotaFloorReached):
            source.fetch_hr_props("2025-07-04")
        assert session.calls == 1                # stopped before spending more

    def test_parse_is_pure_and_fixture_shaped(self):
        rows = parse_event_odds_json(EVENT_PAYLOAD)
        assert len(rows) == 3
        schwarber = [r for r in rows if r["player_name"] == "Kyle Schwarber"]
        assert {r["side"] for r in schwarber} == {"Yes", "No"}
        assert schwarber[0]["book"] == "draftkings"


# ── Market probabilities + edges ────────────────────────────────────────────────

def _props_df():
    rows = parse_event_odds_json(EVENT_PAYLOAD)
    df = pd.DataFrame(rows)
    df["slate_date"] = "2025-07-04"
    df["snapshot_ts"] = "2025-07-04T13:05:00+00:00"
    return df


class TestEdges:

    def test_two_way_devig_when_both_sides_quoted(self):
        market = market_probabilities(_props_df())
        schwarber = market[market["player_name"] == "Kyle Schwarber"].iloc[0]
        implied_yes = american_to_implied(320)
        implied_no = american_to_implied(-450)
        expected = implied_yes / (implied_yes + implied_no)
        assert schwarber["devig_quality"] == "two_way"
        assert schwarber["market_p_devig"] == pytest.approx(expected, abs=1e-4)

    def test_assumed_devig_when_one_sided(self):
        market = market_probabilities(_props_df())
        soto = market[market["player_name"] == "Juan Soto"].iloc[0]
        assert soto["devig_quality"] == "assumed"
        assert soto["market_p_devig"] < american_to_implied(450)

    def test_edge_ev_kelly_hand_check(self):
        market = market_probabilities(_props_df())
        scored = pd.DataFrame([
            {"batter_name": "Kyle Schwarber", "hr_probability": 0.28,
             "batter_id": 656941, "game_pk": 999},
            {"batter_name": "Juan Soto", "hr_probability": 0.15,
             "batter_id": 665742, "game_pk": 999},
        ])
        edges = build_prop_edges(scored, market)
        row = edges[edges["player_name"] == "Kyle Schwarber"].iloc[0]
        assert row["edge"] == pytest.approx(0.28 - row["market_p_devig"], abs=1e-5)
        # +320 → decimal 4.2, b = 3.2: EV = .28*3.2 − .72 = 0.176
        assert row["ev_per_unit"] == pytest.approx(0.176, abs=1e-4)
        assert row["kelly_frac"] == pytest.approx(0.176 / 3.2, abs=1e-4)
        assert "snapshot_ts" in edges.columns    # forward-archive ready

    def test_uncalibrated_frame_rejected(self):
        with pytest.raises(ValueError, match="hr_probability"):
            build_prop_edges(pd.DataFrame({"batter_name": ["X"]}),
                             market_probabilities(_props_df()))

    def test_name_normalization(self):
        assert normalize_name("Ronald Acuña Jr.") == "ronald acuna"
        assert normalize_name("J.D.  Martinez") == "jd martinez"


# ── Parlays ─────────────────────────────────────────────────────────────────────

class TestParlay:

    def test_product_math(self):
        legs = [Leg("A", 0.25, 4.2, game_pk=1),
                Leg("B", 0.20, 5.5, game_pk=2)]
        parlay = combine(legs)
        assert parlay.fair_p == pytest.approx(0.05)
        assert parlay.decimal_odds == pytest.approx(23.1)
        assert parlay.ev_per_unit == pytest.approx(0.05 * 22.1 - 0.95, abs=1e-4)
        assert parlay.correlated is False

    def test_same_game_flagged_correlated(self):
        parlay = combine([Leg("A", 0.25, 4.2, game_pk=1),
                          Leg("B", 0.20, 5.5, game_pk=1)])
        assert parlay.correlated is True
        assert parlay.notes

    def test_single_leg_rejected(self):
        with pytest.raises(ValueError):
            combine([Leg("A", 0.25, 4.2)])


# ── Voss Edge tiers ─────────────────────────────────────────────────────────────

class TestVossEdgeTiers:

    def test_rubric_loads_and_is_draft(self):
        rubric = load_rubric()
        assert "draft" in str(rubric["version"])
        assert set(rubric["tiers"]) <= {"L2", "L3", "L4", "L5"}

    def test_high_edge_two_way_assigns_high_tier(self):
        result = assign_tier(TierInputs(
            edge=0.055, model_p=0.28, devig_quality="two_way",
            data_completeness=1.0))
        assert result.tier == "L4"
        assert result.caps_applied == []
        assert result.inputs["edge"] == 0.055    # audit trail preserved

    def test_assumed_devig_caps_tier(self):
        result = assign_tier(TierInputs(
            edge=0.09, model_p=0.20, devig_quality="assumed",
            data_completeness=1.0))
        assert result.uncapped_tier == "L5"
        assert result.tier == "L3"
        assert "assumed_devig" in result.caps_applied

    def test_correlated_parlay_caps_tier(self):
        result = assign_tier(TierInputs(
            edge=0.09, model_p=0.20, devig_quality="two_way",
            data_completeness=1.0, correlated=True))
        assert result.tier == "L2"

    def test_low_completeness_caps_tier(self):
        result = assign_tier(TierInputs(
            edge=0.06, model_p=0.10, devig_quality="two_way",
            data_completeness=0.5))
        assert result.tier == "L2"
        assert "low_data_completeness" in result.caps_applied

    def test_no_edge_is_l1(self):
        result = assign_tier(TierInputs(
            edge=0.005, model_p=0.05, devig_quality="two_way",
            data_completeness=1.0))
        assert result.tier == "L1"
