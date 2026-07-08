"""
tests/test_narrator.py
=======================
Layer-5 tests: narrator payload construction, JSON schema validation, the
no-invented-numbers check, top-N cost gating, and prop-card rendering.
Fully offline — a fake Anthropic client, zero API calls.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from enrichment.llm_narrator import (
    build_card_payload,
    build_prompt,
    narrate_cards,
    narrative_numbers_ok,
    parse_card_response,
)
from reports.prop_card import build_card_inputs, render_text_card


def _card_input(name="Kyle Schwarber", edge=0.055, model_p=0.28,
                market_p=0.2254, line=320):
    return {
        "edge_row": {
            "player_name": name, "batter_team": "PHI", "slate_date": "2025-07-04",
            "book": "draftkings", "line_american": line,
            "model_p": model_p, "market_p_devig": market_p, "edge": edge,
            "ev_per_unit": 0.176, "kelly_frac": 0.055,
            "batter_id": 656941, "game_pk": 999, "devig_quality": "two_way",
        },
        "tier": {"tier": "L4", "uncapped_tier": "L4", "caps_applied": [],
                 "rubric_version": "0.1-draft"},
        "drivers": [
            {"feature": "b_barrel_rate_30d", "shap": 0.021, "value": 0.14},
            {"feature": "p_hr_per_pa_allowed_30d", "shap": 0.015, "value": 0.045},
            {"feature": "wind_out_mph", "shap": 0.008, "value": 9.0},
        ],
    }


VALID_CARD = {
    "headline": "Schwarber's barrels meet a homer-prone starter.",
    "matchup_story": "The model projects 28.0% against a market price of 22.5%.",
    "risk_note": None,
}


class FakeMessage:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class FakeClient:
    """Counts calls; returns canned JSON (optionally per-call)."""

    def __init__(self, responses=None):
        self.calls = 0
        self._responses = responses
        self.messages = self

    def create(self, **kwargs):
        idx = self.calls
        self.calls += 1
        if self._responses is not None:
            text = self._responses[idx]
        else:
            text = json.dumps(VALID_CARD)
        return FakeMessage(text)


class TestPayload:

    def test_all_numbers_formatted_by_us(self):
        payload = build_card_payload(**{k: _card_input()[k] for k in
                                        ("edge_row", "tier", "drivers")})
        assert payload["model_p_pct"] == "28.0%"
        assert payload["market_p_pct"] == "22.5%"
        assert payload["line_american"] == "+320"
        assert payload["edge_pp"] == "+5.5 percentage points"
        assert "b_barrel_rate_30d" in payload["drivers_block"]

    def test_prompt_contains_injected_numbers_and_rule(self):
        ci = _card_input()
        prompt = build_prompt(build_card_payload(ci["edge_row"], ci["tier"], ci["drivers"]))
        assert "28.0%" in prompt and "+320" in prompt
        assert "do not compute" in prompt.lower()


class TestResponseValidation:

    def test_valid_json_parses(self):
        card = parse_card_response(json.dumps(VALID_CARD))
        assert card["headline"].startswith("Schwarber")

    def test_fenced_json_parses(self):
        card = parse_card_response(f"```json\n{json.dumps(VALID_CARD)}\n```")
        assert set(card) == {"headline", "matchup_story", "risk_note"}

    def test_missing_keys_rejected(self):
        with pytest.raises(ValueError, match="missing keys"):
            parse_card_response(json.dumps({"headline": "x"}))

    def test_invented_number_detected(self):
        ci = _card_input()
        payload = build_card_payload(ci["edge_row"], ci["tier"], ci["drivers"])
        ok_card = {"headline": "x", "matchup_story": "Model says 28.0%.",
                   "risk_note": None}
        bad_card = {"headline": "x", "matchup_story": "He hit 46 homers in 2021.",
                    "risk_note": None}
        assert narrative_numbers_ok(payload, ok_card)
        assert not narrative_numbers_ok(payload, bad_card)


class TestNarrateCards:

    def test_top_n_gating_caps_api_calls(self):
        inputs = [_card_input(name=f"B{i}", edge=0.02 + i / 100) for i in range(5)]
        client = FakeClient()
        cards = narrate_cards(inputs, top_n=2, client=client)
        assert client.calls == 2                     # cost control
        # highest-edge rows were the ones narrated
        narrated = {c["edge_row"]["player_name"] for c in cards}
        assert narrated == {"B4", "B3"}

    def test_invalid_json_card_dropped_not_fatal(self):
        inputs = [_card_input(name="Good", edge=0.06),
                  _card_input(name="Bad", edge=0.05)]
        client = FakeClient(responses=[json.dumps(VALID_CARD), "not json at all"])
        cards = narrate_cards(inputs, top_n=2, client=client)
        assert len(cards) == 1

    def test_number_inventing_card_dropped(self):
        invented = dict(VALID_CARD, matchup_story="He is hitting .315 with 46 bombs.")
        client = FakeClient(responses=[json.dumps(invented)])
        cards = narrate_cards([_card_input()], top_n=1, client=client)
        assert cards == []


class TestPropCard:

    def test_build_card_inputs_assigns_tiers(self):
        edges = pd.DataFrame([_card_input()["edge_row"]])
        cards = build_card_inputs(edges)
        assert cards[0]["tier"]["tier"] == "L4"
        assert cards[0]["tier"]["rubric_version"].endswith("draft")

    def test_render_text_card_smoke(self):
        card = _card_input()
        card["narrative"] = VALID_CARD
        text = render_text_card(card)
        assert "HR PROP CARD — Kyle Schwarber" in text
        assert "+320" in text and "L4" in text
        assert "28.0%" in text
