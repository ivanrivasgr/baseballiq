"""
models/train_hr.py
===================
Train the HR-prop classifier: P(batter hits a home run in a game).

Pipeline:
    1. Load Gold `batter_matchup_features` (one row = batter × game)
    2. Date-grouped time-series CV — folds split on UNIQUE GAME DATES so a
       single day can never straddle train/validation, and every fold
       trains strictly on earlier dates. No shuffling exists anywhere.
    3. Fit the final model on the earliest ~80% of dates, an isotonic
       calibrator on the next ~10%, and report holdout metrics on the
       final ~10% — calibration is part of the artifact, not an option.
    4. SHAP summary for explainability.
    5. Save artifact: model + calibrator + feature list + metadata.

The existing models/train.py (pitcher CSW regressor) is left untouched —
the dashboard still uses it. This module is the HR-prop counterpart.

Class imbalance note: HR base rate is ~6-9%. We deliberately do NOT use
scale_pos_weight or resampling — the deliverable is an honest probability
to price against sportsbook lines, not a balanced classifier.

Usage:
    python -m models.train_hr --output models/artifacts/hr_prop_v1.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from models.evaluate import max_calibration_gap, probability_metrics, reliability_table
from models.features import TARGET, feature_cols, validate_features
from pipeline.config import GOLD_DIR

logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "n_estimators":     600,
    "max_depth":        4,
    "learning_rate":    0.03,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}

MIN_PRIOR_PA_30D = 10          # require some rolling history
PROB_CLIP = (1e-6, 1 - 1e-6)   # keep calibrated probs off exact 0/1


# ─── Data ───────────────────────────────────────────────────────────────────────

def load_training_data(gold_path: Path | None = None) -> pd.DataFrame:
    path = gold_path or (GOLD_DIR / "batter_matchup_features.parquet")
    df = pd.read_parquet(path)
    return prepare_training_frame(df)


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to modelable rows and enforce temporal ordering."""
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df["b_pa_30d"].fillna(0) >= MIN_PRIOR_PA_30D]
    df = df[df["opp_starter_id"].notna()]
    df = df.sort_values("game_date").reset_index(drop=True)
    logger.info("Training frame: %d rows (from %d), base rate %.4f",
                len(df), n0, df[TARGET].mean() if len(df) else float("nan"))
    return df


# ─── Splits (date-grouped — the leakage-safe way) ───────────────────────────────

def date_grouped_time_series_folds(
    game_dates: pd.Series, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    TimeSeriesSplit over UNIQUE sorted dates, mapped back to row masks.
    Guarantees: every validation date is strictly later than every training
    date in its fold, and no calendar date appears on both sides.
    """
    from sklearn.model_selection import TimeSeriesSplit

    dates = np.sort(pd.unique(game_dates))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_d_idx, val_d_idx in tscv.split(dates):
        train_dates, val_dates = dates[train_d_idx], dates[val_d_idx]
        assert train_dates.max() < val_dates.min(), "fold date overlap"
        folds.append((
            game_dates.isin(train_dates).to_numpy(),
            game_dates.isin(val_dates).to_numpy(),
        ))
    return folds


def time_series_cv(df: pd.DataFrame, cols: list[str], n_splits: int = 5) -> list[dict]:
    validate_features(cols)   # allowlist enforced for ANY caller, not just the CLI
    X, y = df[cols], df[TARGET]
    results = []
    for fold, (train_mask, val_mask) in enumerate(
            date_grouped_time_series_folds(df["game_date"], n_splits), 1):
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X[train_mask], y[train_mask], verbose=False)
        p = model.predict_proba(X[val_mask])[:, 1]

        metrics = probability_metrics(y[val_mask].to_numpy(), p)
        metrics["fold"] = fold
        metrics["train_dates"] = (str(df.loc[train_mask, "game_date"].min()),
                                  str(df.loc[train_mask, "game_date"].max()))
        metrics["val_dates"] = (str(df.loc[val_mask, "game_date"].min()),
                                str(df.loc[val_mask, "game_date"].max()))
        results.append(metrics)
        logger.info("Fold %d | logloss %.5f | brier %.6f | auc %s | val %s → %s",
                    fold, metrics["log_loss"], metrics["brier"],
                    metrics.get("auc"), *metrics["val_dates"])
    return results


# ─── Final model + mandatory calibration ────────────────────────────────────────

def train_with_calibration(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Chronological three-way split on unique dates:
        earliest 80%  → model fit
        next    10%   → isotonic calibrator fit
        final   10%   → holdout report (raw vs calibrated)
    Returns dict with model, calibrator, and holdout evidence.
    """
    validate_features(cols)   # allowlist enforced for ANY caller, not just the CLI
    dates = np.sort(pd.unique(df["game_date"]))
    i_train = int(len(dates) * 0.8)
    i_cal = int(len(dates) * 0.9)
    train_dates, cal_dates, test_dates = dates[:i_train], dates[i_train:i_cal], dates[i_cal:]
    if len(cal_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Not enough distinct dates for train/cal/test split")

    parts = {
        name: df[df["game_date"].isin(d)]
        for name, d in [("train", train_dates), ("cal", cal_dates), ("test", test_dates)]
    }
    logger.info("Split sizes — train %d | cal %d | test %d rows",
                *(len(parts[k]) for k in ("train", "cal", "test")))

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(parts["train"][cols], parts["train"][TARGET], verbose=False)

    p_cal_raw = model.predict_proba(parts["cal"][cols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_cal_raw, parts["cal"][TARGET])

    y_test = parts["test"][TARGET].to_numpy()
    p_test_raw = model.predict_proba(parts["test"][cols])[:, 1]
    p_test_cal = np.clip(calibrator.predict(p_test_raw), *PROB_CLIP)

    holdout = {
        "raw": probability_metrics(y_test, p_test_raw),
        "calibrated": probability_metrics(y_test, p_test_cal),
        "reliability_calibrated": reliability_table(y_test, p_test_cal).to_dict("records"),
        "max_calibration_gap": round(max_calibration_gap(y_test, p_test_cal), 5),
        "date_ranges": {
            "train": (str(train_dates[0]), str(train_dates[-1])),
            "cal": (str(cal_dates[0]), str(cal_dates[-1])),
            "test": (str(test_dates[0]), str(test_dates[-1])),
        },
    }
    logger.info("Holdout — raw brier %.6f → calibrated brier %.6f | max cal gap %.4f",
                holdout["raw"]["brier"], holdout["calibrated"]["brier"],
                holdout["max_calibration_gap"])
    return {"model": model, "calibrator": calibrator, "holdout": holdout}


def shap_summary(model: xgb.XGBClassifier, X: pd.DataFrame) -> dict:
    import shap
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return dict(zip(X.columns, np.abs(values).mean(axis=0).round(6).tolist()))


# ─── Artifact ───────────────────────────────────────────────────────────────────

def save_artifact(path: Path, model, calibrator, cols: list[str], metadata: dict) -> None:
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": cols,
        "target": TARGET,
        "profile": "morning_of",
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Artifact saved: %s", path)


def load_artifact(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-path", default=None, help="Path to batter_matchup_features.parquet")
    parser.add_argument("--output", default="models/artifacts/hr_prop_v1.pkl")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    cols = feature_cols("morning_of")
    validate_features(cols)

    df = load_training_data(Path(args.gold_path) if args.gold_path else None)

    cv_results = [] if args.skip_cv else time_series_cv(df, cols)

    fitted = train_with_calibration(df, cols)

    shap_vals = shap_summary(fitted["model"], df[cols].tail(2000))
    top = sorted(shap_vals.items(), key=lambda kv: -kv[1])[:5]
    logger.info("Top SHAP features: %s", top)

    metadata = {
        "cv_results": cv_results,
        "holdout": fitted["holdout"],
        "shap_summary": shap_vals,
        "n_training_rows": len(df),
        "date_range": (str(df["game_date"].min()), str(df["game_date"].max())),
        "xgb_params": XGB_PARAMS,
    }
    save_artifact(Path(args.output), fitted["model"], fitted["calibrator"], cols, metadata)
