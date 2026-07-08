"""
betting/odds_loader.py
=======================
Odds sources behind one interface. v1 ships two:

    TheOddsAPISource — live HR prop lines from The Odds API (paid tier for
                       player props). Aggressively quota-conscious:
        * every raw response is cached to data/bronze/odds/date=*/ BEFORE
          parsing — a snapshot is never paid for twice, and the growing
          archive doubles as the forward-collected backtest dataset
        * only the `batter_home_runs` market, optional bookmaker filter
        * remaining quota read from response headers; requests hard-stop
          at a configurable floor (QuotaFloorReached)

    CSVOddsSource    — offline source for tests/manual entry:
                       columns: slate_date, player_name, book, side, american

Fetch and parse are separate so tests run on recorded fixtures with zero
network calls.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pipeline.config import BRONZE_DIR

logger = logging.getLogger(__name__)

PROP_COLUMNS = [
    "snapshot_ts", "slate_date", "event_id", "commence_time",
    "home_team", "away_team", "book", "player_name", "side", "american",
]


class QuotaFloorReached(RuntimeError):
    """Raised instead of spending quota below the configured floor."""


class OddsSource(ABC):
    @abstractmethod
    def fetch_hr_props(self, slate_date: str) -> pd.DataFrame:
        """Return HR prop lines for a slate date with PROP_COLUMNS schema."""


# ── The Odds API ────────────────────────────────────────────────────────────────

class TheOddsAPISource(OddsSource):
    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "baseball_mlb"
    MARKET = "batter_home_runs"

    def __init__(
        self,
        api_key: str | None = None,
        bookmakers: list[str] | None = None,
        regions: str = "us",
        quota_floor: int = 25,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.bookmakers = bookmakers
        self.regions = regions
        self.quota_floor = quota_floor
        self.cache_dir = cache_dir or (BRONZE_DIR / "odds")
        self._session = session or requests.Session()
        self.remaining_quota: int | None = None

    # -- public ------------------------------------------------------------

    def fetch_hr_props(self, slate_date: str) -> pd.DataFrame:
        events = self._get_json(
            slate_date, "events",
            f"{self.BASE_URL}/sports/{self.SPORT}/events",
            {"apiKey": self.api_key,
             "commenceTimeFrom": f"{slate_date}T00:00:00Z",
             "commenceTimeTo": f"{slate_date}T23:59:59Z"},
        )
        rows: list[dict[str, Any]] = []
        for event in events:
            params = {"apiKey": self.api_key, "markets": self.MARKET,
                      "regions": self.regions, "oddsFormat": "american"}
            if self.bookmakers:
                params["bookmakers"] = ",".join(self.bookmakers)
            payload = self._get_json(
                slate_date, f"odds_{event['id']}",
                f"{self.BASE_URL}/sports/{self.SPORT}/events/{event['id']}/odds",
                params,
            )
            rows.extend(parse_event_odds_json(payload))

        df = pd.DataFrame(rows, columns=[c for c in PROP_COLUMNS
                                         if c not in ("snapshot_ts", "slate_date")])
        df.insert(0, "slate_date", slate_date)
        df.insert(0, "snapshot_ts", datetime.now(timezone.utc).isoformat())
        logger.info("Fetched %d HR prop quotes for %s (quota remaining: %s)",
                    len(df), slate_date, self.remaining_quota)
        return df

    # -- internals ----------------------------------------------------------

    def _get_json(self, slate_date: str, key: str, url: str, params: dict) -> Any:
        """Cache-first GET. Raw body is persisted before parsing."""
        cache_path = self.cache_dir / f"date={slate_date}" / f"{key}.json"
        if cache_path.exists():
            logger.debug("odds cache hit: %s", cache_path)
            return json.loads(cache_path.read_text())

        if self.remaining_quota is not None and self.remaining_quota <= self.quota_floor:
            raise QuotaFloorReached(
                f"Remaining quota {self.remaining_quota} at/below floor "
                f"{self.quota_floor}; refusing to spend more"
            )

        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()

        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            self.remaining_quota = int(float(remaining))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text)          # raw first, parse second
        return resp.json()


def parse_event_odds_json(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten one event-odds payload to quote rows. Pure function.
    The Odds API encodes HR props as Over/Under 0.5 with the player in
    `description`; Over 0.5 == "Yes, hits a HR".
    """
    rows = []
    for bm in payload.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != TheOddsAPISource.MARKET:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "event_id": payload.get("id"),
                    "commence_time": payload.get("commence_time"),
                    "home_team": payload.get("home_team"),
                    "away_team": payload.get("away_team"),
                    "book": bm.get("key"),
                    "player_name": outcome.get("description"),
                    "side": "Yes" if outcome.get("name") == "Over" else "No",
                    "american": outcome.get("price"),
                })
    return rows


# ── CSV (offline/manual) ────────────────────────────────────────────────────────

class CSVOddsSource(OddsSource):
    """Manual/offline lines: slate_date, player_name, book, side, american."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def fetch_hr_props(self, slate_date: str) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df = df[df["slate_date"].astype(str) == slate_date].copy()
        for col in ("event_id", "commence_time", "home_team", "away_team"):
            if col not in df.columns:
                df[col] = None
        if "side" not in df.columns:
            df["side"] = "Yes"
        df["snapshot_ts"] = datetime.now(timezone.utc).isoformat()
        return df[PROP_COLUMNS]
