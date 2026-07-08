"""
betting/edge.py
================
Model probability vs market probability → edge, EV, Kelly.

Rules enforced here:
    - the model side must be the CALIBRATED `hr_probability` from
      models/predict_hr.py — this module never sees raw scores
    - the market side is devigged: two-way when both sides are quoted
      (devig_quality='two_way'), assumed-overround haircut when only Yes is
      posted (devig_quality='assumed' — the tier layer caps those)
    - every output row carries `snapshot_ts` so the forward-collected
      archive is backtest-ready from day one
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from betting.odds import (
    american_to_decimal,
    american_to_implied,
    devig_one_sided,
    devig_two_way,
)

logger = logging.getLogger(__name__)

EDGE_COLUMNS = [
    "slate_date", "snapshot_ts", "book", "player_name", "batter_id", "game_pk",
    "line_american", "decimal_odds", "implied_p",
    "model_p", "market_p_devig", "devig_quality",
    "edge", "ev_per_unit", "kelly_frac",
]


def normalize_name(name: str) -> str:
    """Join key for sportsbook player names vs our roster names."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def market_probabilities(props: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse quote rows (Yes/No per player/book) into one market row per
    (slate_date, book, player) with a devigged HR probability.
    """
    out = []
    keys = ["slate_date", "book", "player_name"]
    for (slate_date, book, player), grp in props.groupby(keys):
        yes = grp[grp["side"] == "Yes"]
        no = grp[grp["side"] == "No"]
        if yes.empty:
            continue  # can't price a HR prop without the Yes side
        yes_row = yes.iloc[0]
        implied_yes = american_to_implied(float(yes_row["american"]))

        if not no.empty:
            implied_no = american_to_implied(float(no.iloc[0]["american"]))
            market_p, _ = devig_two_way(implied_yes, implied_no)
            quality = "two_way"
        else:
            market_p = devig_one_sided(implied_yes)
            quality = "assumed"

        out.append({
            "slate_date": slate_date,
            "snapshot_ts": yes_row.get("snapshot_ts"),
            "book": book,
            "player_name": player,
            "player_key": normalize_name(player),
            "event_id": yes_row.get("event_id"),
            "line_american": float(yes_row["american"]),
            "decimal_odds": round(american_to_decimal(float(yes_row["american"])), 4),
            "implied_p": round(implied_yes, 5),
            "market_p_devig": round(market_p, 5),
            "devig_quality": quality,
        })
    return pd.DataFrame(out)


def build_prop_edges(scored: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Join calibrated model output to devigged market rows and compute value.

    `scored` needs: hr_probability (+ batter_name for the join; batter_id /
    game_pk carried through when present). `market` comes from
    market_probabilities().
    """
    if "hr_probability" not in scored.columns:
        raise ValueError("scored frame must carry calibrated `hr_probability`")
    if "batter_name" not in scored.columns:
        raise ValueError("scored frame needs `batter_name` to match sportsbook lines")

    s = scored.copy()
    s["player_key"] = s["batter_name"].map(normalize_name)

    merged = market.merge(
        s[[c for c in ("player_key", "hr_probability", "batter_id", "game_pk")
           if c in s.columns]],
        on="player_key", how="inner",
    )
    unmatched = len(market) - len(merged)
    if unmatched:
        logger.warning("%d market rows had no model match (name mismatch or "
                       "batter filtered out)", unmatched)

    merged["model_p"] = merged["hr_probability"]
    merged["edge"] = (merged["model_p"] - merged["market_p_devig"]).round(5)
    b = merged["decimal_odds"] - 1.0
    merged["ev_per_unit"] = (merged["model_p"] * b - (1 - merged["model_p"])).round(5)
    merged["kelly_frac"] = ((merged["model_p"] * b - (1 - merged["model_p"])) / b) \
        .clip(lower=0).round(5)

    cols = [c for c in EDGE_COLUMNS if c in merged.columns]
    return merged[cols].sort_values("edge", ascending=False).reset_index(drop=True)
