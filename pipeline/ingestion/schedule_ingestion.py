"""
pipeline/ingestion/schedule_ingestion.py
=========================================
Bronze-layer ingestion of MLB schedule data via the free MLB Stats API.

Per game we capture what the HR-prop pipeline needs at morning-of-slate time:
    - probable starting pitchers (the "opposing starter" join key for Gold)
    - venue (park factor + stadium orientation joins)
    - scheduled first pitch time + day/night flag
    - home/away teams

Writes one Parquet file per day under:
    data/bronze/schedule/game_date=YYYY-MM-DD/part-000.parquet

The fetch and parse steps are separate functions so tests can run against
recorded JSON fixtures with zero network calls.

Usage:
    python -m pipeline.ingestion.schedule_ingestion --start 2024-07-01 --end 2024-07-07
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pipeline.config import BRONZE_DIR

logger = logging.getLogger(__name__)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
HYDRATE = "probablePitcher,team"
REQUEST_TIMEOUT_S = 30
PEOPLE_BATCH_SIZE = 100

SCHEDULE_COLS = [
    "game_pk", "game_date", "game_datetime_utc", "day_night", "game_type",
    "status",
    "venue_id", "venue_name",
    "home_team_id", "home_team_name", "home_team_abbr",
    "away_team_id", "away_team_name", "away_team_abbr",
    "home_probable_pitcher_id", "home_probable_pitcher_name",
    "away_probable_pitcher_id", "away_probable_pitcher_name",
]


def fetch_schedule_json(game_date: str) -> dict[str, Any]:
    """Pull one day of schedule JSON from the MLB Stats API (free, no key)."""
    resp = requests.get(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": game_date, "hydrate": HYDRATE},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def parse_schedule_json(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Flatten a Stats API schedule payload into one row per game.

    Pure function — no network. Probable pitchers may be absent (announced
    late or TBD); those fields are left null rather than dropped so the slate
    still lists every game.
    """
    rows: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_pp = home.get("probablePitcher", {}) or {}
            away_pp = away.get("probablePitcher", {}) or {}
            rows.append({
                "game_pk":            game.get("gamePk"),
                "game_date":          day.get("date"),
                "game_datetime_utc":  game.get("gameDate"),
                "day_night":          game.get("dayNight"),
                "game_type":          game.get("gameType"),
                "status":             (game.get("status", {}) or {}).get("detailedState"),
                "venue_id":           (game.get("venue", {}) or {}).get("id"),
                "venue_name":         (game.get("venue", {}) or {}).get("name"),
                "home_team_id":       (home.get("team", {}) or {}).get("id"),
                "home_team_name":     (home.get("team", {}) or {}).get("name"),
                "home_team_abbr":     (home.get("team", {}) or {}).get("abbreviation"),
                "away_team_id":       (away.get("team", {}) or {}).get("id"),
                "away_team_name":     (away.get("team", {}) or {}).get("name"),
                "away_team_abbr":     (away.get("team", {}) or {}).get("abbreviation"),
                "home_probable_pitcher_id":   home_pp.get("id"),
                "home_probable_pitcher_name": home_pp.get("fullName"),
                "away_probable_pitcher_id":   away_pp.get("id"),
                "away_probable_pitcher_name": away_pp.get("fullName"),
            })

    df = pd.DataFrame(rows, columns=SCHEDULE_COLS)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        for col in ("game_pk", "venue_id", "home_team_id", "away_team_id",
                    "home_probable_pitcher_id", "away_probable_pitcher_id"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def ingest_schedule_range(start: str, end: str) -> None:
    """Ingest schedule for a date range (inclusive), one Parquet per day."""
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    current = start_dt
    while current <= end_dt:
        _ingest_single_day(current.isoformat())
        current += timedelta(days=1)
    logger.info("Schedule ingestion complete: %s → %s", start, end)


def _ingest_single_day(game_date: str) -> Path:
    out_dir = BRONZE_DIR / f"schedule/game_date={game_date}"
    out_path = out_dir / "part-000.parquet"

    if out_path.exists():
        logger.info("Skip (already exists): %s", out_path)
        return out_path

    logger.info("Fetching schedule: %s", game_date)
    df = parse_schedule_json(fetch_schedule_json(game_date))
    if df.empty:
        logger.info("No games on %s", game_date)
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    logger.info("Wrote %d games → %s", len(df), out_path)
    return out_path


def fetch_people_json(person_ids: list[int]) -> dict[str, Any]:
    """One Stats API people call for up to PEOPLE_BATCH_SIZE player ids."""
    resp = requests.get(
        MLB_PEOPLE_URL,
        params={"personIds": ",".join(str(i) for i in person_ids)},
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def parse_people_json(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten a people payload to (player_id, full_name, bats, throws)."""
    rows = [{
        "player_id": p.get("id"),
        "full_name": p.get("fullName"),
        "bats": (p.get("batSide", {}) or {}).get("code"),
        "throws": (p.get("pitchHand", {}) or {}).get("code"),
    } for p in payload.get("people", [])]
    return pd.DataFrame(rows, columns=["player_id", "full_name", "bats", "throws"])


def ingest_player_names(con) -> None:
    """
    Resolve real names for every batter/pitcher id in Silver via the free
    Stats API people endpoint (batched, ~100 ids per request), and build the
    Silver `player_names` table. Skips ids already resolved, so incremental
    daily runs only fetch new call-ups.
    """
    out_path = BRONZE_DIR / "players" / "names.parquet"
    known: set[int] = set()
    frames: list[pd.DataFrame] = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        frames.append(existing)
        known = set(existing["player_id"].astype(int))

    ids = [int(r[0]) for r in con.execute("""
        SELECT DISTINCT batter_id FROM pitches WHERE batter_id IS NOT NULL
        UNION SELECT DISTINCT pitcher_id FROM pitches WHERE pitcher_id IS NOT NULL
    """).fetchall()]
    missing = sorted(set(ids) - known)
    logger.info("Resolving %d new player names (%d already cached)",
                len(missing), len(known))

    for i in range(0, len(missing), PEOPLE_BATCH_SIZE):
        batch = missing[i:i + PEOPLE_BATCH_SIZE]
        frames.append(parse_people_json(fetch_people_json(batch)))

    if frames:
        names = pd.concat(frames, ignore_index=True).drop_duplicates("player_id")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        names.to_parquet(out_path, index=False)
        con.execute("CREATE OR REPLACE TABLE player_names AS SELECT * FROM names")
        logger.info("player_names table: %d players", len(names))


def load_schedule_to_silver(con) -> None:
    """Load all Bronze schedule partitions into the Silver `schedule` table."""
    glob = str(BRONZE_DIR / "schedule/**/*.parquet")
    con.execute(f"""
    CREATE OR REPLACE TABLE schedule AS
    SELECT * FROM read_parquet('{glob}', hive_partitioning=false, union_by_name=true)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY game_date) = 1
    """)
    n = con.execute("SELECT COUNT(*) FROM schedule").fetchone()[0]
    logger.info("Silver schedule table: %d games", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest MLB schedule to Bronze layer")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    args = parser.parse_args()
    ingest_schedule_range(args.start, args.end)
