"""
models/train.py
===============
Train an XGBoost model to predict pitcher CSW rate (called strikes + whiffs / pitches).

CSW rate is the single best predictor of pitcher value in a given game start —
preferred by modern front offices over ERA, WHIP, or even strikeout rate.

Pipeline:
    1. Load Gold pitcher_game_summary
    2. Build feature matrix with lag/rolling features
    3. Time-series cross-validation (no lookahead)
    4. Train final XGBoost model
    5. Evaluate with RMSE, MAE, R²
    6. Generate SHAP explainability values
    7. Save model artifact + feature importance

Usage:
    python -m models.train --output models/artifacts/pitcher_effectiveness_v1.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from pipeline.config import GOLD_DIR

logger = logging.getLogger(__name__)

# ─── Feature Configuration ──────────────────────────────────────────────────────

FEATURE_COLS = [
    # Rolling performance signals
    "rolling_30d_avg_velo",
    "rolling_30d_whiff_rate",
    "rolling_30d_csw_rate",

    # Delta features (recent vs baseline)
    "velo_vs_30d_avg",
    "whiff_rate_delta",

    # Pitch characteristics
    "avg_spin",
    "avg_h_break",
    "avg_v_break",
    "stuff_diversity",      # pitch mix entropy

    # Command & zone metrics
    "zone_rate",
    "chase_rate",

    # Contact quality allowed
    "barrel_rate_allowed",
    "avg_xwoba_allowed",

    # Contextual
    "total_pitches",        # pitch count = proxy for workload
    "home_away",            # 1=home, 0=away (encoded)
]

TARGET = "csw_rate"

XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "objective":        "reg:squarederror",
}


# ─── Data Loading ───────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Load Gold pitcher_game_summary and prepare for modeling.

    Key rules:
        - Drop rows with null target or null rolling features
          (insufficient history — typically first 10 games of season)
        - Sort by game_date to preserve temporal order for CV
        - Encode categoricals
    """
    path = GOLD_DIR / "pitcher_game_summary.parquet"
    df   = pd.read_parquet(path)

    logger.info("Loaded %d rows from Gold layer", len(df))

    # Require at least 30 days of history (rolling features populated)
    df = df.dropna(subset=[
        "rolling_30d_csw_rate",
        "rolling_30d_whiff_rate",
        "rolling_30d_avg_velo",
        TARGET,
    ])

    # Encode home/away from game context (requires join in production)
    # Simulated here as binary flag
    if "home_away" not in df.columns:
        df["home_away"] = np.random.randint(0, 2, len(df))   # placeholder

    df = df.sort_values("game_date").reset_index(drop=True)
    logger.info("Training data after filtering: %d rows", len(df))
    return df


# ─── Training ───────────────────────────────────────────────────────────────────

def time_series_cv(df: pd.DataFrame) -> list[dict]:
    """
    5-fold time-series cross-validation.

    CRITICAL: Never allow future games to appear in training fold.
    TimeSeriesSplit preserves temporal ordering — no shuffling.
    """
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET]

    tscv    = TimeSeriesSplit(n_splits=5)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**XGBOOST_PARAMS, early_stopping_rounds=30)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        metrics = {
            "fold":  fold,
            "n_train": len(X_train),
            "n_val":   len(X_val),
            "rmse":  round(np.sqrt(mean_squared_error(y_val, preds)), 5),
            "mae":   round(mean_absolute_error(y_val, preds), 5),
            "r2":    round(r2_score(y_val, preds), 4),
            "date_range_val": (
                df.iloc[val_idx]["game_date"].min().isoformat(),
                df.iloc[val_idx]["game_date"].max().isoformat(),
            ),
        }
        results.append(metrics)
        logger.info(
            "Fold %d | RMSE: %.5f | MAE: %.5f | R²: %.4f | Val dates: %s → %s",
            fold, metrics["rmse"], metrics["mae"], metrics["r2"],
            *metrics["date_range_val"],
        )

    avg_rmse = np.mean([r["rmse"] for r in results])
    avg_r2   = np.mean([r["r2"]   for r in results])
    logger.info("CV Summary — avg RMSE: %.5f | avg R²: %.4f", avg_rmse, avg_r2)
    return results


def train_final_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    """Train on full dataset for production artifact."""
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df[TARGET]

    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)
    logger.info("Final model trained on %d samples", len(X))
    return model


# ─── SHAP Explainability ────────────────────────────────────────────────────────

def compute_shap(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for model explainability.

    In production:
        - These get stored per-prediction and surfaced in scouting reports
        - Top-3 SHAP features become the "why" narrative in the AI report
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def feature_importance_df(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Return sorted feature importance as a DataFrame."""
    importance = model.get_booster().get_fscore()
    df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    return df


# ─── Save / Load Artifact ───────────────────────────────────────────────────────

def save_model(model: xgb.XGBRegressor, path: Path, metadata: dict) -> None:
    """Save model + metadata bundle."""
    artifact = {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "target":       TARGET,
        "metadata":     metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Model artifact saved: %s", path)


def load_model(path: Path) -> tuple[xgb.XGBRegressor, list[str]]:
    """Load model artifact for inference."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["feature_cols"]


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="models/artifacts/pitcher_effectiveness_v1.pkl",
        help="Path to save model artifact",
    )
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()

    df = load_training_data()

    cv_results = []
    if not args.skip_cv:
        cv_results = time_series_cv(df)

    model = train_final_model(df)

    # Feature importance
    fi = feature_importance_df(model)
    logger.info("Top 5 features:\n%s", fi.head().to_string())

    # SHAP on a sample (full run is slow)
    X_sample     = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median()).tail(500)
    shap_values  = compute_shap(model, X_sample)
    shap_summary = dict(zip(FEATURE_COLS, np.abs(shap_values).mean(axis=0).tolist()))
    logger.info("SHAP mean absolute values: %s", shap_summary)

    metadata = {
        "cv_results":    cv_results,
        "shap_summary":  shap_summary,
        "feature_importance": fi.to_dict("records"),
        "n_training_rows": len(df),
        "date_range": (df["game_date"].min().isoformat(), df["game_date"].max().isoformat()),
    }

    save_model(model, Path(args.output), metadata)
