"""
betting/odds.py
================
Pure odds mathematics. No I/O, no state — every function is unit-testable
and round-trip safe.

Conventions:
    american: +150 / -120 style (never in (-100, 100) except ±100)
    decimal:  European decimal odds, > 1.0
    implied:  probability in (0, 1) INCLUDING the book's vig
    fair:     probability with vig removed (devigged)

Devig: HR props are two-way markets (Yes/No, i.e. Over/Under 0.5). When both
sides are quoted we remove the overround multiplicatively (v1 default; the
interface leaves room for power/Shin later). When only one side is quoted we
apply an assumed-overround haircut and the caller must mark the result as
lower quality (`devig_quality='assumed'`) — the tier layer caps confidence
on those.
"""

from __future__ import annotations

DEFAULT_ASSUMED_OVERROUND = 0.05  # one-sided fallback haircut


# ── Conversions ─────────────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    if -100 < american < 100:
        raise ValueError(f"Invalid american odds: {american}")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def decimal_to_implied(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal}")
    return 1.0 / decimal


def american_to_implied(american: float) -> float:
    return decimal_to_implied(american_to_decimal(american))


# ── Devig ───────────────────────────────────────────────────────────────────────

def devig_two_way(implied_yes: float, implied_no: float) -> tuple[float, float]:
    """
    Multiplicative devig of a two-way market. Returns (fair_yes, fair_no),
    which sum to exactly 1.
    """
    total = implied_yes + implied_no
    if total <= 0:
        raise ValueError("Implied probabilities must be positive")
    return implied_yes / total, implied_no / total


def devig_one_sided(
    implied_yes: float, assumed_overround: float = DEFAULT_ASSUMED_OVERROUND
) -> float:
    """
    Fallback when only the Yes side is quoted: shave an assumed overround
    share. Callers MUST flag results as devig_quality='assumed'.
    """
    return implied_yes / (1.0 + assumed_overround)


def overround(implied_yes: float, implied_no: float) -> float:
    """Book margin of a two-way market (0 = fair, 0.05 = 5% vig)."""
    return implied_yes + implied_no - 1.0
