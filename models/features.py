"""
models/features.py
===================
The single source of truth for HR-prop model features.

Every feature the model may consume is registered here with an availability
tag — WHEN the value is knowable relative to the game being predicted:

    T-1DAY           computed entirely from prior days (rolling Statcast)
    MORNING_FORECAST knowable morning-of as a forecast (weather, park, slate)
    LINEUP_RELEASE   knowable only once lineups post (~2-4h before first pitch)
    IN_GAME          an outcome of the game itself — NEVER a feature

Training and inference must request features through `feature_cols()`;
`validate_features()` refuses any column that is unregistered or banned.
This is leakage guardrail #4 from docs/HR_PROP_PLAN.md — the allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET = "hr_hit"

# Availability tags
T_1DAY = "T-1DAY"
MORNING_FORECAST = "MORNING_FORECAST"
LINEUP_RELEASE = "LINEUP_RELEASE"
IN_GAME = "IN_GAME"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    availability: str


REGISTRY: list[FeatureSpec] = [
    # ── batter form (rolling, strictly prior days) ─────────────────────
    FeatureSpec("b_pa_30d",                    T_1DAY),
    FeatureSpec("b_hr_per_pa_7d",              T_1DAY),
    FeatureSpec("b_hr_per_pa_30d",             T_1DAY),
    FeatureSpec("b_hr_per_pa_365d",            T_1DAY),
    FeatureSpec("b_hr_per_pa_shrunk",          T_1DAY),
    FeatureSpec("b_barrel_rate_30d",           T_1DAY),
    FeatureSpec("b_hard_hit_30d",              T_1DAY),
    FeatureSpec("b_fb_rate_30d",               T_1DAY),
    FeatureSpec("b_pull_fb_rate_30d",          T_1DAY),
    FeatureSpec("b_avg_ev_30d",                T_1DAY),
    FeatureSpec("b_max_ev_30d",                T_1DAY),
    FeatureSpec("b_avg_pa_per_game_30d",       T_1DAY),
    FeatureSpec("b_avg_slot_30d",              T_1DAY),
    FeatureSpec("b_hr_per_pa_vs_hand_365d",    T_1DAY),
    # ── opposing starter ───────────────────────────────────────────────
    FeatureSpec("same_handed",                 T_1DAY),
    FeatureSpec("p_pa_30d",                    T_1DAY),
    FeatureSpec("p_hr_per_pa_allowed_30d",     T_1DAY),
    FeatureSpec("p_hr_per_pa_allowed_365d",    T_1DAY),
    FeatureSpec("p_barrel_rate_allowed_30d",   T_1DAY),
    FeatureSpec("p_fb_rate_allowed_30d",       T_1DAY),
    FeatureSpec("p_avg_ev_allowed_30d",        T_1DAY),
    # ── opposing bullpen ───────────────────────────────────────────────
    FeatureSpec("bp_hr_per_pa_allowed_30d",    T_1DAY),
    # ── game context ───────────────────────────────────────────────────
    FeatureSpec("is_home",                     T_1DAY),
    FeatureSpec("park_hr_factor_hand",         T_1DAY),
    FeatureSpec("park_hr_factor",              T_1DAY),
    FeatureSpec("is_night",                    MORNING_FORECAST),
    FeatureSpec("temp_f",                      MORNING_FORECAST),
    FeatureSpec("wind_out_mph",                MORNING_FORECAST),
    # ── lineup-release profile only (v2) ───────────────────────────────
    FeatureSpec("lineup_slot_actual",          LINEUP_RELEASE),
]

# Same-game outcomes present in the Gold table for target/analysis purposes.
# These may NEVER appear in any prediction profile.
BANNED_AS_FEATURES: frozenset[str] = frozenset({
    TARGET,
    "pa_this_game",
    "hr_this_game",
})

# Legacy dashboard columns computed over full-partition windows (include
# future games). Also banned — leakage guardrail #2.
BANNED_AS_FEATURES = BANNED_AS_FEATURES | frozenset({
    "season_avg_velo", "season_avg_csw", "season_avg_whiff",
})

PROFILES: dict[str, frozenset[str]] = {
    # Morning of slate: the v1 production profile (user decision).
    "morning_of": frozenset({T_1DAY, MORNING_FORECAST}),
    # After lineups post: adds real lineup slot (v2).
    "lineup_release": frozenset({T_1DAY, MORNING_FORECAST, LINEUP_RELEASE}),
}


def feature_cols(profile: str = "morning_of") -> list[str]:
    """Feature names available under the given prediction-time profile."""
    allowed = PROFILES[profile]
    return [f.name for f in REGISTRY if f.availability in allowed]


def validate_features(cols: list[str]) -> None:
    """
    Refuse unregistered or banned columns. Called by training and inference —
    a new Gold column cannot leak into the model by accident; it must be
    registered here first, with an availability tag.
    """
    registered = {f.name for f in REGISTRY}
    banned = [c for c in cols if c in BANNED_AS_FEATURES]
    if banned:
        raise ValueError(f"Banned (in-game/leaky) columns requested as features: {banned}")
    unknown = [c for c in cols if c not in registered]
    if unknown:
        raise ValueError(f"Unregistered feature columns (add to models/features.py): {unknown}")
    if IN_GAME in {f.availability for f in REGISTRY if f.name in cols}:
        raise ValueError("IN_GAME-tagged columns can never be features")
