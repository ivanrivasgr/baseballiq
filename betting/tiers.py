"""
betting/tiers.py
=================
Voss Edge L1–L5 confidence tier hooks.

The rubric is CONFIGURATION, not code: thresholds live in
config/voss_edge.yaml so cutoffs can be retuned from backtest results
without touching the pipeline. Every assignment returns the full set of
rubric inputs alongside the tier, so any pick's tier is auditable and the
narrator can explain WHY something is (say) L4.

Current yaml is a marked DRAFT — final numbers come from the holdout
flat-stake ROI backtest per the build plan.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parents[1] / "config" / "voss_edge.yaml"

TIER_ORDER = ["L1", "L2", "L3", "L4", "L5"]


@dataclass(frozen=True)
class TierInputs:
    edge: float
    model_p: float
    devig_quality: str            # 'two_way' | 'assumed'
    data_completeness: float      # fraction of model features non-null
    correlated: bool = False      # parlay-only flag


@dataclass(frozen=True)
class TierAssignment:
    tier: str
    uncapped_tier: str
    caps_applied: list[str]
    inputs: dict
    rubric_version: str


def load_rubric(path: Path | None = None) -> dict:
    with open(path or DEFAULT_RUBRIC_PATH) as f:
        return yaml.safe_load(f)


def _cap(tier: str, cap_tier: str) -> str:
    return TIER_ORDER[min(TIER_ORDER.index(tier), TIER_ORDER.index(cap_tier))]


def assign_tier(inputs: TierInputs, rubric: dict | None = None) -> TierAssignment:
    rubric = rubric or load_rubric()

    tier = "L1"
    for name in reversed(TIER_ORDER[1:]):          # L5 → L2
        cond = rubric["tiers"].get(name)
        if cond is None:
            continue
        if (inputs.edge >= cond.get("min_edge", 0.0)
                and inputs.model_p >= cond.get("min_model_p", 0.0)):
            tier = name
            break

    uncapped = tier
    caps = rubric.get("caps", {})
    caps_applied = []

    if inputs.devig_quality != "two_way":
        capped = _cap(tier, caps.get("assumed_devig_max_tier", "L3"))
        if capped != tier:
            caps_applied.append("assumed_devig")
        tier = capped

    if inputs.correlated:
        capped = _cap(tier, caps.get("correlated_max_tier", "L2"))
        if capped != tier:
            caps_applied.append("correlated_parlay")
        tier = capped

    if inputs.data_completeness < caps.get("min_data_completeness", 0.0):
        capped = _cap(tier, caps.get("low_completeness_max_tier", "L2"))
        if capped != tier:
            caps_applied.append("low_data_completeness")
        tier = capped

    return TierAssignment(
        tier=tier,
        uncapped_tier=uncapped,
        caps_applied=caps_applied,
        inputs=asdict(inputs),
        rubric_version=str(rubric.get("version", "unknown")),
    )
