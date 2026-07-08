"""
enrichment/llm_narrator.py
===========================
LLM narrator for HR prop cards.

THE ABSOLUTE RULE (docs/HR_PROP_PLAN.md §0.2): the LLM writes prose FROM
computed numbers and NEVER computes, sums, compares, or invents a number.
Every figure the narrative may mention is string-formatted by US before the
prompt is built; the response is schema-validated JSON, and
`narrative_numbers_ok()` verifies no numeric token appears in the prose
that wasn't in the injected payload.

Cost controls (docs/HR_PROP_PLAN.md §10):
    - narrate only the top-N edge plays per slate (default 10)
    - Haiku model (short structured cards ≈ pennies per slate)
    - optional Batch API path (50% discount) for non-urgent runs
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pipeline.config import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

NARRATOR_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 500
DEFAULT_TOP_N = 10

REQUIRED_KEYS = {"headline", "matchup_story", "risk_note"}

PROP_CARD_PROMPT = """\
You are a sports betting analyst writing a home-run prop scouting card for an
internal system. All numbers below were computed by a validated model
pipeline. Use ONLY these numbers — do not compute, derive, or estimate any
other number, total, or comparison. If a number is not listed, do not
mention one.

Batter: {batter_name} ({batter_team}) vs {opp_starter_desc}
Date: {slate_date}

Model probability of a home run: {model_p_pct}
Sportsbook line: {line_american} at {book} (market implied, vig removed: {market_p_pct})
Edge: {edge_pp}
Voss Edge tier: {tier}{caps_note}

Top model drivers (SHAP):
{drivers_block}

Respond with JSON only, exactly these keys:
- "headline": one sentence, max 15 words
- "matchup_story": 2-3 sentences on why the model likes/dislikes this spot,
  referencing only the drivers and numbers above
- "risk_note": one sentence on the main risk, or null if the tier has no caps
"""


# ── Payload construction (all formatting happens HERE, not in the LLM) ─────────

def build_card_payload(edge_row: dict[str, Any], tier: dict[str, Any],
                       drivers: list[dict[str, Any]]) -> dict[str, str]:
    """Pre-format every number the narrative is allowed to contain."""
    caps = tier.get("caps_applied") or []
    drivers_block = "\n".join(
        f"  - {d['feature']}: value {d['value']}, impact {d['shap']:+.3f}"
        for d in drivers
    ) or "  (drivers unavailable)"

    return {
        "batter_name": str(edge_row.get("player_name", "Unknown")),
        "batter_team": str(edge_row.get("batter_team", "")),
        "opp_starter_desc": str(edge_row.get("opp_starter_desc", "opposing starter")),
        "slate_date": str(edge_row.get("slate_date", "")),
        "model_p_pct": f"{float(edge_row['model_p']) * 100:.1f}%",
        "market_p_pct": f"{float(edge_row['market_p_devig']) * 100:.1f}%",
        "line_american": f"{int(edge_row['line_american']):+d}",
        "book": str(edge_row.get("book", "")),
        "edge_pp": f"{float(edge_row['edge']) * 100:+.1f} percentage points",
        "tier": str(tier.get("tier", "L1")),
        "caps_note": f" (capped: {', '.join(caps)})" if caps else "",
        "drivers_block": drivers_block,
    }


def build_prompt(payload: dict[str, str]) -> str:
    return PROP_CARD_PROMPT.format(**payload)


# ── Response validation ─────────────────────────────────────────────────────────

def parse_card_response(text: str) -> dict[str, Any]:
    """Parse + schema-validate the narrator's JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    card = json.loads(text)
    missing = REQUIRED_KEYS - set(card)
    if missing:
        raise ValueError(f"Narrator response missing keys: {missing}")
    return {k: card[k] for k in REQUIRED_KEYS}


_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def narrative_numbers_ok(payload: dict[str, str], card: dict[str, Any]) -> bool:
    """
    Every numeric token in the narrative must literally appear in the
    injected payload — the mechanical check that the LLM computed nothing.
    """
    allowed = set(_NUM_TOKEN.findall(" ".join(str(v) for v in payload.values())))
    prose = " ".join(str(v) for v in card.values() if v)
    return all(tok in allowed for tok in _NUM_TOKEN.findall(prose))


# ── Narration ───────────────────────────────────────────────────────────────────

def narrate_cards(
    card_inputs: list[dict[str, Any]],
    top_n: int = DEFAULT_TOP_N,
    client: Any | None = None,
    use_batch: bool = False,
) -> list[dict[str, Any]]:
    """
    card_inputs: dicts with keys edge_row, tier, drivers (see prop_card.py).
    Narrates the top_n by edge; every other row stays numbers-only.
    `client` is injectable for tests; defaults to a real Anthropic client.
    """
    ranked = sorted(card_inputs, key=lambda c: -float(c["edge_row"]["edge"]))[:top_n]
    if not ranked:
        return []

    if client is None:
        import anthropic
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set — cannot narrate")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    payloads = [build_card_payload(c["edge_row"], c["tier"], c["drivers"])
                for c in ranked]

    if use_batch:
        texts = _run_batch(client, payloads)
    else:
        texts = []
        for payload in payloads:
            resp = client.messages.create(
                model=NARRATOR_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": build_prompt(payload)}],
            )
            texts.append(resp.content[0].text)
            time.sleep(0.3)

    cards = []
    for card_input, payload, text in zip(ranked, payloads, texts):
        try:
            card = parse_card_response(text)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Narrator response invalid for %s: %s",
                           payload["batter_name"], exc)
            continue
        if not narrative_numbers_ok(payload, card):
            logger.warning("Narrator invented a number for %s — card dropped",
                           payload["batter_name"])
            continue
        cards.append({**card_input, "payload": payload, "narrative": card})
    return cards


def _run_batch(client: Any, payloads: list[dict[str, str]],
               poll_s: int = 15, timeout_s: int = 1800) -> list[str]:
    """Anthropic Batch API path (50% discount) for non-urgent runs."""
    batch = client.messages.batches.create(requests=[
        {
            "custom_id": f"card-{i}",
            "params": {
                "model": NARRATOR_MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": build_prompt(p)}],
            },
        }
        for i, p in enumerate(payloads)
    ])
    deadline = time.time() + timeout_s
    while batch.processing_status != "ended":
        if time.time() > deadline:
            raise TimeoutError("Narration batch did not finish in time")
        time.sleep(poll_s)
        batch = client.messages.batches.retrieve(batch.id)

    texts = [""] * len(payloads)
    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id.split("-")[1])
        if result.result.type == "succeeded":
            texts[idx] = result.result.message.content[0].text
    return texts
