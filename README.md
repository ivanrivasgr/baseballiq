# ⚾ BaseballIQ — Production MLB Analytics Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-ff4b4b.svg)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-yellow.svg)](https://duckdb.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Claude API](https://img.shields.io/badge/Claude-Sonnet-black.svg)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A portfolio-grade sports analytics system demonstrating production data engineering, machine learning, and LLM integration on MLB Statcast data.

**[→ Live Demo](https://baseballiq.streamlit.app)** · **[Architecture Doc](ARCHITECTURE.md)** · **[Engineering Write-up](WRITEUP.md)**

---

## What This Is

BaseballIQ is an end-to-end MLB analytics platform built to demonstrate what a modern sports analytics engineering team would realistically ship. It ingests pitch-by-pitch Statcast data, processes it through a **Bronze/Silver/Gold medallion architecture**, trains a predictive model for **pitcher effectiveness (CSW rate)**, and surfaces results through an **AI-powered scouting report system** backed by Claude.

This is not a tutorial project. Every architectural decision — from using DuckDB over Pandas for the analytical layer, to doing time-series cross-validation for the ML model, to positioning the LLM *after* all statistics are computed — reflects real production engineering judgment.

---

## Architecture

```
RAW STATCAST (pybaseball)
        │
        ▼
┌─────────────────────┐
│  BRONZE  (Parquet)  │  Raw, immutable, partitioned by game_date
│  ~3M rows/season    │  No transformations
└─────────┬───────────┘
          │  cleaning · typing · dedup
          ▼
┌─────────────────────┐
│  SILVER  (DuckDB)   │  Normalized schema: pitches, at_bats, games
│  4 normalized tables│  Feature engineering via SQL window functions
└─────────┬───────────┘
          │  aggregation · business logic
          ▼
┌─────────────────────┐
│  GOLD  (Parquet)    │  pitcher_game_summary · batter_game_summary
│  analytics-ready    │  Rolling 30d averages · delta features · tiers
└──────┬──────────────┘
       │                    │
       ▼                    ▼
 XGBoost Model        LLM Enrichment
 CSW prediction       Claude: narratives,
 + SHAP values        anomaly detection
       │                    │
       └──────────┬─────────┘
                  ▼
        AI Scouting Reports
        (stats + ML + LLM merged)
                  │
                  ▼
        Streamlit Dashboard
```

---

## Key Technical Decisions

### Why DuckDB instead of Pandas?
DuckDB runs complex SQL window functions (30-day rolling averages, cross-pitcher percentiles) on millions of rows in seconds with no cluster. The Gold layer aggregation SQL is more readable, testable, and 3-5x faster than equivalent Pandas code. It also produces Parquet-native output, keeping the pipeline stateless.

### Why CSW rate as the ML target?
Called strikes + whiffs / total pitches is the strongest single-game pitcher quality signal, preferred by front offices over ERA (noisy, defense-dependent), K% (doesn't capture command), or WHIP (context-dependent). CSW is also more stable across small samples, making it a better regression target.

### Why time-series cross-validation?
Standard k-fold shuffles the data, allowing the model to train on "future" games. In production, you never have future data. `TimeSeriesSplit` preserves temporal ordering — each validation fold is strictly after its training fold. This is the most common ML leakage mistake in sports analytics portfolios.

### Why LLM as a layer, not the core?
The LLM receives fully-computed statistics and writes narrative around them. It never computes numbers. This makes the system auditable (every stat is verifiable), testable (outputs don't change if the LLM is replaced), and cost-efficient (one API call per report, not per calculation).

---

## Project Structure

```
baseballiq/
├── data/
│   ├── bronze/          # Raw Parquet, partitioned by game_date
│   ├── silver/          # DuckDB analytical database
│   └── gold/            # Aggregated Parquet (checked into repo for demo)
│       ├── pitcher_game_summary.parquet
│       ├── batter_game_summary.parquet
│       └── llm_insights.json
├── pipeline/
│   ├── ingestion/       # statcast_ingestion.py
│   ├── silver/          # cleaning.py, feature_engineering.py
│   └── gold/            # aggregations.py
├── enrichment/          # llm_client.py, prompt_templates.py
├── models/              # train.py, evaluate.py, predict.py
├── reports/             # scouting_report.py (assembly engine)
├── dashboard/           # app.py (Streamlit, 4 pages)
├── tests/               # pytest suite
├── generate_demo_data.py
├── Makefile
└── pyproject.toml
```

---

## Quickstart

### Option A: Run the demo (no setup required)
```bash
git clone https://github.com/yourhandle/baseballiq
cd baseballiq
pip install -r requirements.txt
streamlit run dashboard/app.py
```
The demo uses pre-generated synthetic Statcast data included in `data/gold/`. No API keys needed for cached reports.

### Option B: Full pipeline with real Statcast data
```bash
# 1. Set up environment
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# 2. Ingest one week of Statcast data
make ingest DATE_START=2024-07-01 DATE_END=2024-07-07

# 3. Clean → Gold
make clean && make gold

# 4. LLM enrichment
make enrich DATE=2024-07-07

# 5. Train pitcher effectiveness model
make train

# 6. Generate a scouting report
make report PLAYER_ID=605483 DATE=2024-07-06

# 7. Launch dashboard
make dashboard
```

---

## Deploy to Streamlit Community Cloud (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork → Branch: `main` → File: `dashboard/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy** ✓

The dashboard loads from pre-generated Gold Parquet files (~15MB total). No database, no heavy compute at startup. Live LLM report generation uses the API key on demand.

---

## ML Model: Pitcher Effectiveness

**Target:** CSW rate (called strikes + whiffs / total pitches)

**Features:**
- `rolling_30d_csw_rate` — recent baseline
- `rolling_30d_whiff_rate` — swing-and-miss trending
- `velo_vs_30d_avg` — velocity delta vs. recent mean
- `whiff_rate_delta` — whiff improvement/decline
- `stuff_diversity` — pitch mix entropy
- `zone_rate`, `chase_rate` — command metrics
- `barrel_rate_allowed`, `avg_xwoba_allowed` — contact quality

**Model:** XGBoost Regressor
**Validation:** 5-fold TimeSeriesSplit (no leakage)
**Explainability:** SHAP values per prediction, surfaced in scouting reports

---

## Example Scouting Report Output

```
═══════════════════════════════════════════════════════════════
  BASEBALLIQ SCOUTING REPORT · Internal Use Only
═══════════════════════════════════════════════════════════════

PITCHER: Marcus Delgado  (LAD)
GAME: July 6, 2024 vs. HOU    GRADE: A  (92nd percentile)

── GAME STATS ──────────────────────────────────────────────────
  Pitches:      98
  Avg Velo:     96.2 mph  (↑ +0.7 vs. 30d avg)
  Whiff Rate:   33.8%  ★ ELITE
  CSW Rate:     32.1%  ★ ELITE
  xwOBA:        .261

── ML PREDICTION ───────────────────────────────────────────────
  Next start projected CSW:  30.8%
  Stuff Score:  89 / 100
  Top SHAP drivers:
    1. rolling_30d_whiff_rate    (+0.018 ↑ positive)
    2. velo_vs_30d_avg           (+0.011 ↑ positive)
    3. stuff_diversity           (+0.007 ↑ positive)

── AI ANALYST SUMMARY ──────────────────────────────────────────
  Tier: ELITE

  "Delgado's slider was historic — generated 41% whiff rate
  against Houston's right-handed lineup."

  Key Finding: Elite velocity held above 96 mph through the
  7th inning with no fatigue-related decline. Pitch mix
  entropy suggests unpredictable sequencing that kept hitters
  from sitting on any single pitch. The .261 xwOBA allowed
  ranks in the 89th percentile for starters this season.

  Concern Flag: None identified.
═══════════════════════════════════════════════════════════════
```

---

## Data Source

**MLB Statcast** via [`pybaseball`](https://github.com/jldbc/pybaseball) — free, public, ~3M pitch events per MLB season. The demo version ships with synthetic data calibrated to real 2024 Statcast distributions. Swap in `make ingest` to pull live data.

---

## Stack

| Layer | Technology |
|---|---|
| Data ingestion | `pybaseball`, `pandas`, `pyarrow` |
| Bronze storage | Apache Parquet (partitioned) |
| Analytical engine | DuckDB 0.10 |
| Feature engineering | DuckDB SQL window functions |
| ML model | XGBoost + SHAP |
| LLM enrichment | Anthropic Claude (claude-sonnet-4) |
| Dashboard | Streamlit + Plotly |
| Task runner | GNU Make |
| Testing | pytest |
| Dependency management | uv / pyproject.toml |

---

## License

MIT — use freely for portfolio, education, or as a starting point for production systems.
