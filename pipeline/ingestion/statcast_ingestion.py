"""
pipeline/ingestion/statcast_ingestion.py
========================================
Bronze-layer ingestion of MLB Statcast data via pybaseball.

Pulls pitch-by-pitch event data for a date range and writes
immutable Parquet files partitioned by game_date.

Usage:
    python -m pipeline.ingestion.statcast_ingestion \
        --start 2024-07-01 \
        --end   2024-07-07
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pybaseball as pb

from pipeline.config import BRONZE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

# Statcast columns we care about — keeps Parquet files lean
KEEP_COLS = [
    "game_pk", "game_date", "inning", "inning_topbot",
    "pitcher", "batter",
    "pitch_type", "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
    "description", "events", "stand", "p_throws",
    "balls", "strikes", "outs_when_up",
    "launch_speed", "launch_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "delta_run_exp", "barrel",
    "home_team", "away_team",
    "post_home_score", "post_away_score",
]


def ingest_date_range(start: str, end: str) -> None:
    """
    Ingest all Statcast pitches for a date range.

    Writes one Parquet file per day under:
        data/bronze/statcast/game_date=YYYY-MM-DD/part-000.parquet

    Args:
        start: ISO date string (e.g. '2024-07-01')
        end:   ISO date string (inclusive, e.g. '2024-07-07')
    """
    pb.cache.enable()          # cache raw downloads to ~/.pybaseball/
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)

    current = start_dt
    while current <= end_dt:
        _ingest_single_day(current.isoformat())
        current += timedelta(days=1)

    logger.info("Ingestion complete: %s → %s", start, end)


def _ingest_single_day(game_date: str) -> Path:
    """Pull one day of Statcast data and write to Bronze Parquet."""
    out_dir  = BRONZE_DIR / f"statcast/game_date={game_date}"
    out_path = out_dir / "part-000.parquet"

    if out_path.exists():
        logger.info("Skip (already exists): %s", out_path)
        return out_path

    logger.info("Fetching Statcast: %s", game_date)
    try:
        raw: pd.DataFrame = pb.statcast(
            start_dt=game_date,
            end_dt=game_date,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("No data for %s: %s", game_date, exc)
        return out_path

    if raw.empty:
        logger.info("No games on %s", game_date)
        return out_path

    # Keep only columns that exist in this pull (API occasionally varies)
    available = [c for c in KEEP_COLS if c in raw.columns]
    df = raw[available].copy()

    # Rename to our canonical schema
    df = df.rename(columns={
        "pitcher":                               "pitcher_id",
        "batter":                                "batter_id",
        "release_spin_rate":                     "release_spin",
        "estimated_ba_using_speedangle":         "estimated_ba",
        "estimated_woba_using_speedangle":       "estimated_woba",
        "barrel":                                "is_barrel",
    })

    # Stable, deterministic pitch_id
    df = df.reset_index(drop=True)
    df["pitch_id"] = (
        df["game_pk"].astype(str) + "_"
        + df["game_date"].astype(str) + "_"
        + df.index.astype(str)
    )

    # Cast types for Parquet efficiency
    df["game_date"]   = pd.to_datetime(df["game_date"]).dt.date
    df["is_barrel"]   = df["is_barrel"].fillna(0).astype(bool)
    df["pitcher_id"]  = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")
    df["batter_id"]   = pd.to_numeric(df["batter_id"],  errors="coerce").astype("Int64")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Wrote %d rows → %s", len(df), out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest Statcast data to Bronze layer")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()
    ingest_date_range(args.start, args.end)
