"""
tests/test_leakage.py
======================
THE guardrail test file for the HR-prop pipeline (docs/HR_PROP_PLAN.md §3c).

Four layers of defense, all offline:

    1. SQL source scan  — every rolling window frame in the Gold/Silver
       feature SQL must end at INTERVAL 1 DAY PRECEDING, and no window
       aggregate may be a bare partition-wide AVG/SUM (the pattern that
       includes future games).
    2. Feature registry — banned in-game columns are refused; the
       morning-of profile excludes lineup-release-only features.
    3. Recompute check  — rolling features rebuilt independently from
       plate_appearances restricted to game_date < row date must equal the
       Gold values.
    4. Fold integrity   — date-grouped TimeSeriesSplit folds never share a
       calendar date, and every validation date follows all training dates.
       Plus a train/calibrate/score smoke test on synthetic data.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

# Files whose SQL produces MODEL FEATURES — held to the strict window rule.
FEATURE_SQL_FILES = [
    REPO / "pipeline/gold/batter_features.py",
    REPO / "pipeline/gold/pitcher_features.py",
]


# ── 1. SQL source scan ──────────────────────────────────────────────────────────

class TestSqlWindowScan:

    def test_every_range_frame_ends_one_day_preceding(self):
        """Any RANGE frame in feature SQL must exclude the current day."""
        # capture the UPPER bound (after AND) of each RANGE frame
        pattern = re.compile(
            r"RANGE BETWEEN\s+INTERVAL\s+\d+\s+DAYS?\s+PRECEDING\s+AND\s+([^)\n]+)",
            re.IGNORECASE,
        )
        for path in FEATURE_SQL_FILES:
            src = path.read_text()
            frames = pattern.findall(src)
            assert frames, f"{path.name}: expected rolling windows, found none"
            for upper_bound in frames:
                assert re.search(r"INTERVAL\s+1\s+DAY\s+PRECEDING",
                                 upper_bound, re.IGNORECASE), (
                    f"{path.name}: window frame upper bound is not "
                    f"INTERVAL 1 DAY PRECEDING → leaks same-day data: "
                    f"...AND {upper_bound.strip()}"
                )

    def test_no_bare_partition_wide_aggregates(self):
        """
        AVG/SUM/MAX/MIN/COUNT OVER (PARTITION BY ... ) with no ORDER BY
        spans the whole partition INCLUDING FUTURE GAMES. Banned in feature
        SQL (this is the repo's legacy season_avg_* leak pattern).
        """
        agg_over = re.compile(
            r"\b(AVG|SUM|MAX|MIN|COUNT)\s*\(\s*[\w.*]+\s*\)\s+OVER\s*\(([^)]*)\)",
            re.IGNORECASE | re.DOTALL,
        )
        for path in FEATURE_SQL_FILES:
            src = path.read_text()
            for match in agg_over.finditer(src):
                window_body = match.group(2)
                assert "ORDER BY" in window_body.upper(), (
                    f"{path.name}: bare partition-wide aggregate (includes "
                    f"future games): {match.group(0)[:100]}"
                )

    def test_named_windows_all_bounded(self):
        """Named WINDOW definitions (w7/w30/w365...) must carry the bound."""
        window_def = re.compile(
            r"w\d+\s+AS\s+\(PARTITION BY.*?\)", re.IGNORECASE | re.DOTALL
        )
        for path in FEATURE_SQL_FILES:
            for defn in window_def.findall(path.read_text()):
                assert "1 DAY PRECEDING" in defn.upper(), (
                    f"{path.name}: named window without the 1-day-preceding "
                    f"bound:\n{defn}"
                )


# ── 2. Feature registry ─────────────────────────────────────────────────────────

class TestFeatureRegistry:

    def test_banned_columns_are_refused(self):
        from models.features import validate_features
        for banned in ("hr_hit", "pa_this_game", "hr_this_game", "season_avg_csw"):
            with pytest.raises(ValueError, match="[Bb]anned"):
                validate_features(["b_barrel_rate_30d", banned])

    def test_unregistered_columns_are_refused(self):
        from models.features import validate_features
        with pytest.raises(ValueError, match="[Uu]nregistered"):
            validate_features(["b_barrel_rate_30d", "some_new_gold_column"])

    def test_morning_of_profile_excludes_lineup_release(self):
        from models.features import feature_cols
        morning = feature_cols("morning_of")
        assert "lineup_slot_actual" not in morning
        assert "b_avg_slot_30d" in morning          # the imputed replacement
        assert "b_barrel_rate_30d" in morning

    def test_registry_matches_gold_schema(self):
        """Every registered feature must exist in the Gold matchup table."""
        from models.features import feature_cols
        from tests.test_hr_gold import _pa
        from pipeline.gold.matchup_table import run_all as build_hr_gold

        rows = [_pa(1001, "2024-06-01", 1, 10, 90, "L", "R", "NYY",
                    ls=90.0, la=10.0, bb_type="line_drive", spray=0.0)]
        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE plate_appearances AS SELECT * FROM df")
        build_hr_gold(con)
        gold_cols = {r[0] for r in con.execute(
            "DESCRIBE batter_matchup_features").fetchall()}
        con.close()
        missing = [c for c in feature_cols("lineup_release") if c not in gold_cols]
        assert not missing, f"Registered features absent from Gold: {missing}"


# ── 3. Independent recompute ────────────────────────────────────────────────────

class TestRecompute:

    def test_rolling_hr_rate_recomputed_from_prior_days_only(self):
        """
        Rebuild b_hr_per_pa_30d for every row using ONLY
        plate_appearances strictly before the row's game_date, and require
        exact agreement with the Gold value.
        """
        from tests.test_hr_gold import _pa
        from pipeline.gold.matchup_table import run_all as build_hr_gold

        rng = np.random.default_rng(7)
        rows, ab = [], 1
        for day in pd.date_range("2024-06-01", periods=12):
            for pa_i in range(4):
                is_hr = int(rng.random() < 0.10)
                rows.append(_pa(
                    9000 + day.day, day.date().isoformat(), ab, 10, 90,
                    "R", "R", "NYY",
                    events="home_run" if is_hr else "field_out",
                    is_hr=is_hr,
                    ls=100.0 if is_hr else 88.0, la=25.0,
                    barrel=bool(is_hr), bb_type="fly_ball" if is_hr else "ground_ball",
                    spray=-18.0 if is_hr else 3.0,
                ))
                ab += 1
        df = pd.DataFrame(rows)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE plate_appearances AS SELECT * FROM df")
        build_hr_gold(con)

        mismatches = con.execute("""
            SELECT m.game_pk, m.b_hr_per_pa_30d, recomputed.rate
            FROM batter_matchup_features m
            LEFT JOIN LATERAL (
                SELECT ROUND(SUM(pa.is_hr) * 1.0 / NULLIF(COUNT(*), 0), 5) AS rate
                FROM plate_appearances pa
                WHERE pa.batter_id = m.batter_id
                  AND pa.game_date <  m.game_date
                  AND pa.game_date >= m.game_date - INTERVAL 30 DAYS
            ) recomputed ON TRUE
            WHERE m.b_hr_per_pa_30d IS DISTINCT FROM recomputed.rate
        """).fetchall()
        con.close()
        assert not mismatches, f"Gold vs independent recompute mismatch: {mismatches}"


# ── 4. Fold integrity + training smoke test ─────────────────────────────────────

def _synthetic_gold(n_days: int = 60, rows_per_day: int = 12) -> pd.DataFrame:
    """Synthetic batter_matchup_features-shaped frame for model tests."""
    from models.features import feature_cols

    rng = np.random.default_rng(42)
    cols = feature_cols("morning_of")
    dates = pd.date_range("2024-04-01", periods=n_days).date
    n = n_days * rows_per_day

    df = pd.DataFrame(rng.normal(0.1, 0.05, size=(n, len(cols))), columns=cols)
    df["game_date"] = np.repeat(dates, rows_per_day)
    # target loosely driven by two features, base rate ~8%
    signal = 2.0 * df["b_barrel_rate_30d"] + 1.0 * df["p_hr_per_pa_allowed_30d"]
    logit = -2.7 + (signal - signal.mean()) / (signal.std() + 1e-9) * 0.5
    df["hr_hit"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


class TestFoldsAndTraining:

    def test_date_grouped_folds_never_share_dates(self):
        from models.train_hr import date_grouped_time_series_folds

        df = _synthetic_gold()
        folds = date_grouped_time_series_folds(df["game_date"], n_splits=5)
        assert len(folds) == 5
        for train_mask, val_mask in folds:
            train_dates = set(df.loc[train_mask, "game_date"])
            val_dates = set(df.loc[val_mask, "game_date"])
            assert not train_dates & val_dates
            assert max(train_dates) < min(val_dates)

    def test_train_calibrate_score_smoke(self):
        """End-to-end: train → calibrate → holdout report → score frame."""
        from models.features import feature_cols
        from models.predict_hr import score_frame
        from models.train_hr import train_with_calibration

        df = _synthetic_gold()
        cols = feature_cols("morning_of")
        fitted = train_with_calibration(df, cols)

        assert fitted["calibrator"] is not None
        holdout = fitted["holdout"]
        assert {"raw", "calibrated", "reliability_calibrated",
                "max_calibration_gap", "date_ranges"} <= set(holdout)
        # holdout strictly after calibration, which is strictly after train
        assert holdout["date_ranges"]["train"][1] < holdout["date_ranges"]["cal"][0]
        assert holdout["date_ranges"]["cal"][1] < holdout["date_ranges"]["test"][0]

        artifact = {"model": fitted["model"], "calibrator": fitted["calibrator"],
                    "feature_cols": cols}
        scored = score_frame(df.tail(50), artifact, with_shap=True)
        assert scored["hr_probability"].between(0, 1).all()
        assert scored["top_drivers"].map(len).eq(3).all()
