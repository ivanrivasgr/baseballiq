"""
reports/prop_card.py
=====================
Assembles HR prop cards: model probability + market line + edge + Voss Edge
tier + SHAP drivers + (for the top-N plays) the LLM narrative.

Card assembly is pure data plumbing — every number was computed upstream
(model, betting layer, tier rubric); the narrative was validated to contain
no numbers beyond the injected payload.

Usage:
    python -m reports.prop_card --edges data/gold/prop_edges.parquet [--narrate]
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from betting.tiers import TierInputs, assign_tier, load_rubric

logger = logging.getLogger(__name__)


def build_card_inputs(edges: pd.DataFrame,
                      drivers_by_key: dict[Any, list[dict]] | None = None,
                      rubric: dict | None = None) -> list[dict[str, Any]]:
    """
    One card input per edge row: tier assignment (with audit inputs) plus
    SHAP drivers keyed by (batter_id, game_pk) when provided.
    """
    rubric = rubric or load_rubric()
    drivers_by_key = drivers_by_key or {}
    inputs = []
    for _, row in edges.iterrows():
        completeness = float(row["data_completeness"]) if "data_completeness" in row else 1.0
        tier = assign_tier(TierInputs(
            edge=float(row["edge"]),
            model_p=float(row["model_p"]),
            devig_quality=str(row.get("devig_quality", "assumed")),
            data_completeness=completeness,
        ), rubric)
        inputs.append({
            "edge_row": row.to_dict(),
            "tier": {"tier": tier.tier, "uncapped_tier": tier.uncapped_tier,
                     "caps_applied": tier.caps_applied,
                     "rubric_version": tier.rubric_version},
            "drivers": drivers_by_key.get((row.get("batter_id"), row.get("game_pk")), []),
        })
    return inputs


def render_text_card(card: dict[str, Any]) -> str:
    """ASCII card in the repo's scouting-report idiom."""
    row, tier = card["edge_row"], card["tier"]
    narrative = card.get("narrative") or {}
    lines = [
        "═" * 55,
        f"  HR PROP CARD — {row.get('player_name', '?')}",
        "═" * 55,
        f"  Model P(HR):      {float(row['model_p']):.1%}",
        f"  Line:             {int(row['line_american']):+d} @ {row.get('book', '?')}"
        f"  (market {float(row['market_p_devig']):.1%}, {row.get('devig_quality')})",
        f"  Edge:             {float(row['edge']):+.1%}",
        f"  EV / unit:        {float(row.get('ev_per_unit', 0)):+.3f}",
        f"  Kelly fraction:   {float(row.get('kelly_frac', 0)):.3f}",
        f"  Voss Edge tier:   {tier['tier']}"
        + (f"  (uncapped {tier['uncapped_tier']}; caps: {', '.join(tier['caps_applied'])})"
           if tier["caps_applied"] else ""),
    ]
    if card.get("drivers"):
        lines.append("  Top drivers:")
        lines += [f"    - {d['feature']}: {d['shap']:+.3f}" for d in card["drivers"]]
    if narrative:
        lines += [
            "  " + "─" * 51,
            f"  {narrative.get('headline', '')}",
            f"  {narrative.get('matchup_story', '')}",
        ]
        if narrative.get("risk_note"):
            lines.append(f"  Risk: {narrative['risk_note']}")
    lines.append("═" * 55)
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="data/gold/prop_edges.parquet")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--narrate", action="store_true",
                        help="Generate LLM narratives for the top-N (costs API tokens)")
    parser.add_argument("--batch", action="store_true",
                        help="Use the Anthropic Batch API (50%% discount, slower)")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()

    edges = pd.read_parquet(args.edges).head(args.top)
    cards = build_card_inputs(edges)

    if args.narrate:
        from enrichment.llm_narrator import narrate_cards
        cards = narrate_cards(cards, top_n=args.top, use_batch=args.batch)

    if args.as_json:
        print(json.dumps([{k: v for k, v in c.items() if k != "edge_row"} |
                          {"edge_row": {kk: str(vv) for kk, vv in c["edge_row"].items()}}
                          for c in cards], indent=2, default=str))
    else:
        for card in cards:
            print(render_text_card(card))
