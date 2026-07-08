"""
models/predict_hr.py
=====================
Inference wrapper for the HR-prop model.

Output per batter-matchup row:
    hr_probability     — CALIBRATED P(HR this game); the only probability
                         the betting layer is allowed to consume
    hr_probability_raw — uncalibrated model score (diagnostics only)
    top_drivers        — top-3 signed SHAP drivers, e.g.
                         [{"feature": "b_barrel_rate_30d", "shap": +0.021,
                           "value": 0.14}, ...]
                         passed verbatim to the prop card / narrator

Usage:
    python -m models.predict_hr --artifact models/artifacts/hr_prop_v1.pkl \
        --gold-path data/gold/batter_matchup_features.parquet --date 2025-07-04
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.features import validate_features
from models.train_hr import PROB_CLIP, load_artifact

logger = logging.getLogger(__name__)

TOP_K_DRIVERS = 3


def score_frame(df: pd.DataFrame, artifact: dict, with_shap: bool = True) -> pd.DataFrame:
    """
    Score a batter-matchup frame with a trained artifact.
    Returns the id/context columns plus probabilities and SHAP drivers.
    """
    cols = artifact["feature_cols"]
    validate_features(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing model features: {missing}")

    X = df[cols]
    model, calibrator = artifact["model"], artifact["calibrator"]

    p_raw = model.predict_proba(X)[:, 1]
    p_cal = np.clip(calibrator.predict(p_raw), *PROB_CLIP)

    out = df[[c for c in ("batter_id", "game_pk", "game_date", "batter_team",
                          "opp_starter_id") if c in df.columns]].copy()
    out["hr_probability"] = np.round(p_cal, 5)
    out["hr_probability_raw"] = np.round(p_raw, 5)

    if with_shap:
        out["top_drivers"] = _top_shap_drivers(model, X)
    return out


def _top_shap_drivers(model, X: pd.DataFrame) -> list[list[dict]]:
    import shap
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    drivers = []
    feature_names = list(X.columns)
    for i in range(len(X)):
        row = values[i]
        top_idx = np.argsort(-np.abs(row))[:TOP_K_DRIVERS]
        drivers.append([
            {
                "feature": feature_names[j],
                "shap": round(float(row[j]), 5),
                "value": None if pd.isna(X.iloc[i, j]) else round(float(X.iloc[i, j]), 5),
            }
            for j in top_idx
        ])
    return drivers


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="models/artifacts/hr_prop_v1.pkl")
    parser.add_argument("--gold-path", default="data/gold/batter_matchup_features.parquet")
    parser.add_argument("--date", default=None, help="Score only this game_date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Optional parquet output path")
    args = parser.parse_args()

    artifact = load_artifact(Path(args.artifact))
    df = pd.read_parquet(args.gold_path)
    if args.date:
        df = df[df["game_date"].astype(str) == args.date]
    if df.empty:
        raise SystemExit("No rows to score")

    scored = score_frame(df, artifact)
    scored = scored.sort_values("hr_probability", ascending=False)
    print(scored.head(20).to_string(index=False))
    if args.output:
        scored.to_parquet(args.output, index=False)
        logger.info("Scored %d rows → %s", len(scored), args.output)
