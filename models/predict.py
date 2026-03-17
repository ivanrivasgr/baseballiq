"""
models/predict.py
=================
Inference wrapper for the pitcher effectiveness XGBoost model.
Used by the scouting report engine and dashboard.

Usage:
    from models.predict import PitcherPredictor
    predictor = PitcherPredictor()
    result = predictor.predict_single(feature_dict)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

from models.train import FEATURE_COLS, load_model
from pipeline.config import MODEL_PATH

logger = logging.getLogger(__name__)


class PitcherPredictor:
    """Load model artifact and run inference with SHAP explanations."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: make train"
            )
        self.model, self.feature_cols = load_model(MODEL_PATH)
        self._explainer = None  # lazy-load SHAP

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict CSW rate for one pitcher-game feature vector.

        Args:
            features: dict with keys matching FEATURE_COLS

        Returns:
            dict with predicted_csw_rate, stuff_score, top_shap_features
        """
        X = pd.DataFrame([features])[self.feature_cols].fillna(0)

        predicted_csw  = float(self.model.predict(X)[0])
        shap_values    = self.explainer.shap_values(X)[0]

        shap_pairs = sorted(
            zip(self.feature_cols, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_shap = [
            {
                "feature":      feat,
                "contribution": round(float(val), 5),
                "direction":    "↑ positive" if val > 0 else "↓ negative",
            }
            for feat, val in shap_pairs[:3]
        ]

        stuff_score = int(np.clip((predicted_csw - 0.18) / 0.20 * 100, 0, 100))

        return {
            "predicted_csw_rate": round(predicted_csw, 4),
            "stuff_score":        stuff_score,
            "top_shap_features":  top_shap,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a DataFrame of feature rows."""
        X      = df[self.feature_cols].fillna(0)
        preds  = self.model.predict(X)
        df     = df.copy()
        df["predicted_csw_rate"] = preds.round(4)
        df["stuff_score"]        = np.clip((preds - 0.18) / 0.20 * 100, 0, 100).astype(int)
        return df
