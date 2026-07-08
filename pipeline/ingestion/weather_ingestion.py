"""
pipeline/ingestion/weather_ingestion.py
========================================
Bronze-layer ingestion of hourly stadium weather from the Open-Meteo archive
API (free, no key).

Cost-conscious fetch pattern: ONE request per stadium per season (hourly
series for the whole date range) instead of one per game — a full 4-season
backfill is ~120 requests total. Files are written per stadium/season and
skipped if present, so the job is resumable.

    data/bronze/weather/team=PHI/season=2024/part-000.parquet

`build_game_weather` then joins the Silver `schedule` table (first-pitch
timestamp, home team) to the hourly series to produce one weather row per
game. The wind-out projection against park orientation happens later, in the
Gold context features, where the parks table lives.

Usage:
    python -m pipeline.ingestion.weather_ingestion --start 2022-04-01 --end 2022-11-15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pipeline.config import BRONZE_DIR
from pipeline.gold.park_data import get_parks_df

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,wind_speed_10m,wind_direction_10m"
REQUEST_TIMEOUT_S = 60


def fetch_hourly_json(lat: float, lon: float, start: str, end: str) -> dict[str, Any]:
    """One archive request: hourly weather for a coordinate over a date range."""
    resp = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": HOURLY_VARS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def parse_hourly_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten an Open-Meteo hourly payload. Pure function — no network."""
    hourly = payload.get("hourly", {})
    df = pd.DataFrame({
        "hour_utc":     hourly.get("time", []),
        "temp_f":       hourly.get("temperature_2m", []),
        "wind_mph":     hourly.get("wind_speed_10m", []),
        "wind_dir_deg": hourly.get("wind_direction_10m", []),
    })
    if not df.empty:
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    return df


def ingest_weather_range(start: str, end: str) -> None:
    """
    Fetch hourly weather for every stadium over [start, end].
    One file per (team, season-of-start-date); skip-if-exists for resume.
    """
    season = int(start[:4])
    parks = get_parks_df([season]).drop_duplicates(subset=["team_abbr"])

    for _, park in parks.iterrows():
        out_dir = BRONZE_DIR / f"weather/team={park.team_abbr}/season={season}"
        out_path = out_dir / "part-000.parquet"
        if out_path.exists():
            logger.info("Skip (already exists): %s", out_path)
            continue

        logger.info("Fetching weather: %s (%s) %s → %s",
                    park.team_abbr, park.venue_name, start, end)
        df = parse_hourly_json(fetch_hourly_json(park.lat, park.lon, start, end))
        if df.empty:
            logger.warning("No weather data for %s", park.team_abbr)
            continue
        df["team_abbr"] = park.team_abbr
        df["season"] = season

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        logger.info("Wrote %d hours → %s", len(df), out_path)


def build_game_weather(con) -> None:
    """
    Silver `game_weather`: one row per game — the hourly observation at the
    scheduled first-pitch hour at the home stadium. Requires the Silver
    `schedule` table and Bronze weather partitions.
    """
    glob = str(BRONZE_DIR / "weather/**/*.parquet")
    con.execute(f"""
    CREATE OR REPLACE TABLE game_weather AS
    SELECT
        s.game_pk,
        w.temp_f,
        w.wind_mph,
        w.wind_dir_deg
    FROM schedule s
    JOIN read_parquet('{glob}', hive_partitioning=false, union_by_name=true) w
      ON w.team_abbr = s.home_team_abbr
     AND w.hour_utc = DATE_TRUNC('hour', CAST(s.game_datetime_utc AS TIMESTAMPTZ))
    """)
    n = con.execute("SELECT COUNT(*) FROM game_weather").fetchone()[0]
    logger.info("Silver game_weather table: %d games", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest stadium weather to Bronze layer")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (same season)")
    args = parser.parse_args()
    ingest_weather_range(args.start, args.end)
