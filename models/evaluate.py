"""
models/evaluate.py
===================
Probability-quality evaluation for the HR-prop classifier.

At a ~7% base rate, accuracy is meaningless. What matters for pricing
against sportsbook lines:

    - log loss / Brier score  (primary)
    - reliability table       (predicted-probability bins vs observed HR
                               rate — the go/no-go artifact for calibration)
    - AUC                     (secondary, rank quality only)

`flat_stake_roi` is the business-KPI backtest hook: given model
probabilities and (devigged) market probabilities with decimal odds, it
simulates flat-stake betting above an edge threshold. The betting layer
supplies real market data; this function only does the accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def probability_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Core probability-quality metrics."""
    out = {
        "n": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 5),
        "mean_predicted": round(float(np.mean(p)), 5),
        "log_loss": round(float(log_loss(y_true, p, labels=[0, 1])), 5),
        "brier": round(float(brier_score_loss(y_true, p)), 6),
    }
    # AUC undefined for single-class slices
    if len(np.unique(y_true)) == 2:
        out["auc"] = round(float(roc_auc_score(y_true, p)), 4)
    return out


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Quantile-binned calibration table: each row is a bin of predictions with
    its mean predicted probability vs the observed HR rate. Well calibrated
    ⇒ the two columns track each other.
    """
    df = pd.DataFrame({"y": y_true, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    table = (
        df.groupby("bin")
          .agg(n=("y", "size"),
               mean_predicted=("p", "mean"),
               observed_rate=("y", "mean"))
          .reset_index()
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table.round(5)


def max_calibration_gap(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Largest |predicted − observed| across reliability bins."""
    table = reliability_table(y_true, p, n_bins)
    return float(table["gap"].abs().max())


def flat_stake_roi(
    model_p: np.ndarray,
    market_p: np.ndarray,
    decimal_odds: np.ndarray,
    outcomes: np.ndarray,
    edge_threshold: float = 0.02,
    stake: float = 1.0,
) -> dict:
    """
    Flat-stake backtest: bet `stake` on every leg where
    model_p − market_p > edge_threshold; settle at decimal_odds.
    """
    picks = (model_p - market_p) > edge_threshold
    n = int(picks.sum())
    if n == 0:
        return {"edge_threshold": edge_threshold, "bets": 0, "roi": None,
                "profit": 0.0, "hit_rate": None}
    profit = float(np.where(outcomes[picks] == 1,
                            stake * (decimal_odds[picks] - 1.0),
                            -stake).sum())
    return {
        "edge_threshold": edge_threshold,
        "bets": n,
        "profit": round(profit, 2),
        "roi": round(profit / (n * stake), 4),
        "hit_rate": round(float(outcomes[picks].mean()), 4),
    }
