"""
pipeline/config.py
==================
Central configuration for all paths, constants, and environment variables.
Every module imports from here — never hardcode paths elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project root ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data layer paths ────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"
BRONZE_DIR  = DATA_DIR / "bronze"
SILVER_DIR  = DATA_DIR / "silver"
GOLD_DIR    = DATA_DIR / "gold"
DUCKDB_PATH = SILVER_DIR / "baseballiq.duckdb"

# ── Model artifacts ─────────────────────────────────────────────────────────────
MODELS_DIR    = PROJECT_ROOT / "models" / "artifacts"
MODEL_PATH    = MODELS_DIR / "pitcher_effectiveness_v1.pkl"

# ── Reports output ──────────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports" / "output"

# ── API keys ────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Demo mode (set DEMO_MODE=1 to skip live API calls) ──────────────────────────
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# ── League averages (2024 Statcast baseline) ─────────────────────────────────────
LEAGUE_AVG = {
    "csw_rate":            0.281,
    "whiff_rate":          0.254,
    "barrel_rate":         0.078,
    "avg_xwoba":           0.312,
    "avg_exit_velo":       88.5,
    "zone_rate":           0.475,
    "chase_rate":          0.295,
}

# ── Ensure directories exist on import ──────────────────────────────────────────
for _dir in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
