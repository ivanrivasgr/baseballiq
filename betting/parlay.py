"""
betting/parlay.py
==================
Multi-leg parlay math on top of prop edges.

v1 treats legs as independent for the fair-probability product and instead
LABELS correlation honestly:

    correlated=True when any two legs share a game_pk (same-game parlay) —
    HR outcomes in one game share weather, pitcher blowup, extra innings.
    v1 does not model the joint distribution; the Voss Edge tier layer caps
    a correlated parlay's tier instead (config: correlated_max_tier).

An honest flag beats fake precision — modeling the correlation is a v2 item.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Leg:
    player_name: str
    model_p: float            # calibrated P(HR)
    decimal_odds: float       # book's Yes price
    game_pk: int | None = None
    book: str | None = None


@dataclass(frozen=True)
class Parlay:
    legs: tuple[Leg, ...]
    fair_p: float             # ∏ model_p (independence assumption)
    decimal_odds: float       # ∏ leg decimals (book parlay price)
    implied_p: float          # 1 / decimal_odds
    ev_per_unit: float
    correlated: bool
    notes: list[str] = field(default_factory=list)


def combine(legs: list[Leg]) -> Parlay:
    if len(legs) < 2:
        raise ValueError("A parlay needs at least 2 legs")
    for leg in legs:
        if not (0.0 < leg.model_p < 1.0):
            raise ValueError(f"Leg {leg.player_name}: model_p out of (0,1)")
        if leg.decimal_odds <= 1.0:
            raise ValueError(f"Leg {leg.player_name}: invalid decimal odds")

    fair_p = 1.0
    decimal = 1.0
    for leg in legs:
        fair_p *= leg.model_p
        decimal *= leg.decimal_odds

    game_pks = [leg.game_pk for leg in legs if leg.game_pk is not None]
    correlated = len(game_pks) != len(set(game_pks))

    notes = []
    if correlated:
        notes.append(
            "Same-game legs detected: fair_p assumes independence and is "
            "optimistic-or-pessimistic depending on shared conditions; tier is capped."
        )

    ev = fair_p * (decimal - 1.0) - (1.0 - fair_p)
    return Parlay(
        legs=tuple(legs),
        fair_p=round(fair_p, 6),
        decimal_odds=round(decimal, 4),
        implied_p=round(1.0 / decimal, 6),
        ev_per_unit=round(ev, 6),
        correlated=correlated,
        notes=notes,
    )
