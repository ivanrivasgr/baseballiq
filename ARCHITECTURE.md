# ⚾ BaseballIQ — Production Sports Analytics Platform
### A Portfolio-Grade Data Engineering + AI Project

---

## Project Overview

**BaseballIQ** is a production-style MLB analytics platform built on Statcast data. It ingests raw pitch-by-pitch event data, processes it through a medallion architecture, enriches it with LLM-generated insights, trains a predictive model for pitcher effectiveness, and surfaces everything in an AI-powered scouting report system and Streamlit dashboard.

**Why MLB Statcast?**
- Richest free public sports dataset available (millions of events/season)
- Dense numeric features: exit velocity, launch angle, spin rate, pitch movement
- Well-suited for ML: strong signal, high volume, clean labels
- `pybaseball` makes ingestion trivial
- Familiar to recruiters in baseball analytics (Dodgers, Red Sox, Cubs all use variants of this stack)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA SOURCES                            │
│  pybaseball / MLB Statcast API  │  Baseball Reference (season)  │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              BRONZE LAYER  (raw/immutable)                      │
│   Parquet files partitioned by  game_date / pitcher_id          │
│   Schema: raw Statcast columns, no transformation               │
└───────────────────┬─────────────────────────────────────────────┘
                    │  cleaning + typing + deduplication
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              SILVER LAYER  (cleaned + normalized)               │
│   DuckDB tables: pitches, at_bats, games, players               │
│   Feature engineering: rolling averages, zone maps, batted ball │
└───────────────────┬─────────────────────────────────────────────┘
                    │  aggregation + business logic
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              GOLD LAYER  (analytical datasets)                  │
│   pitcher_game_summary  │  batter_game_summary  │  team_trends  │
│   Ready for ML, dashboards, and LLM enrichment                  │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│   ML MODEL LAYER     │    │       LLM ENRICHMENT LAYER          │
│  XGBoost: pitcher    │    │  Claude API → summaries, anomalies, │
│  effectiveness model │    │  narrative insights per game/player  │
│  Output: xFIP_pred,  │    │  Output: JSON insight blobs stored  │
│  stuff_score, risk   │    │  back into Gold layer               │
└──────────┬───────────┘    └──────────────┬──────────────────────┘
           │                               │
           └──────────────┬────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              SCOUTING REPORT ENGINE                             │
│   Merges Gold stats + ML scores + LLM text                      │
│   Renders PDF + JSON scouting report per player/game            │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD                                │
│   Player explorer │ Pitcher heatmaps │ ML predictions           │
│   Team trends     │ AI scouting reports (rendered inline)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
baseballiq/
│
├── README.md
├── pyproject.toml              # Poetry / uv project config
├── .env.example                # API keys template
├── Makefile                    # make ingest, make clean, make test, make dash
│
├── data/
│   ├── bronze/                 # Raw Parquet, partitioned
│   │   └── statcast/
│   │       └── game_date=2024-07-01/
│   │           └── part-000.parquet
│   ├── silver/                 # DuckDB database file
│   │   └── baseballiq.duckdb
│   └── gold/                   # Analytical Parquet exports
│       ├── pitcher_game_summary.parquet
│       ├── batter_game_summary.parquet
│       └── llm_insights.parquet
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py               # Paths, constants, env vars
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── statcast_ingestion.py
│   │   └── schedule_ingestion.py
│   ├── bronze/
│   │   ├── __init__.py
│   │   └── bronze_writer.py
│   ├── silver/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── feature_engineering.py
│   │   └── schema.sql          # DuckDB table definitions
│   ├── gold/
│   │   ├── __init__.py
│   │   └── aggregations.py
│   └── orchestrator.py         # Runs full pipeline end-to-end
│
├── enrichment/
│   ├── __init__.py
│   ├── llm_client.py           # Anthropic API wrapper
│   ├── prompt_templates.py     # All LLM prompts
│   └── insight_writer.py       # Writes insights back to Gold
│
├── models/
│   ├── __init__.py
│   ├── features.py             # Feature set definition
│   ├── train.py                # XGBoost training script
│   ├── evaluate.py             # Model evaluation + SHAP
│   ├── predict.py              # Inference wrapper
│   └── artifacts/              # Saved model files
│       └── pitcher_effectiveness_v1.pkl
│
├── reports/
│   ├── __init__.py
│   ├── scouting_report.py      # Report assembly
│   ├── templates/
│   │   └── report.html.jinja2
│   └── output/                 # Generated reports
│
├── dashboard/
│   ├── app.py                  # Streamlit entry point
│   ├── pages/
│   │   ├── 01_pitcher_explorer.py
│   │   ├── 02_batter_explorer.py
│   │   ├── 03_team_trends.py
│   │   └── 04_scouting_reports.py
│   └── components/
│       ├── pitch_heatmap.py
│       └── stat_cards.py
│
└── tests/
    ├── test_ingestion.py
    ├── test_cleaning.py
    ├── test_features.py
    └── test_llm_client.py
```

---

## Schema Design

### Silver Layer — DuckDB Tables

**`pitches`** (core fact table)
```sql
CREATE TABLE pitches (
    pitch_id        VARCHAR PRIMARY KEY,
    game_pk         INTEGER,
    game_date       DATE,
    pitcher_id      INTEGER,
    batter_id       INTEGER,
    inning          INTEGER,
    pitch_type      VARCHAR,        -- FF, SL, CH, CU, SI...
    release_speed   FLOAT,          -- mph
    release_spin    FLOAT,          -- rpm
    pfx_x           FLOAT,          -- horizontal break (inches)
    pfx_z           FLOAT,          -- vertical break (inches)
    plate_x         FLOAT,          -- location at plate
    plate_z         FLOAT,
    zone            INTEGER,        -- 1-14 Statcast zone
    description     VARCHAR,        -- called_strike, swinging_strike...
    events          VARCHAR,        -- strikeout, home_run, single...
    launch_speed    FLOAT,          -- exit velocity
    launch_angle    FLOAT,
    estimated_ba    FLOAT,          -- xBA
    estimated_woba  FLOAT,          -- xwOBA
    delta_run_exp   FLOAT,          -- run expectancy change
    is_barrel       BOOLEAN,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

**`pitcher_game_summary`** (Gold aggregation)
```sql
CREATE TABLE pitcher_game_summary AS
SELECT
    pitcher_id,
    game_date,
    game_pk,
    COUNT(*)                                    AS total_pitches,
    AVG(release_speed)                          AS avg_velo,
    AVG(release_spin)                           AS avg_spin,
    -- Whiff rate
    SUM(CASE WHEN description = 'swinging_strike' THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN description LIKE '%swing%' THEN 1 ELSE 0 END), 0)
                                                AS whiff_rate,
    -- CSW rate (called strike + whiff)
    SUM(CASE WHEN description IN ('called_strike','swinging_strike') THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0)                   AS csw_rate,
    AVG(estimated_woba)                         AS avg_xwoba_allowed,
    SUM(delta_run_exp)                          AS total_re_delta,
    -- Rolling 30-day context added in feature_engineering.py
    NULL::FLOAT                                 AS rolling_30d_whiff_rate,
    NULL::FLOAT                                 AS rolling_30d_avg_velo
FROM pitches
WHERE pitcher_id IS NOT NULL
GROUP BY 1,2,3;
```

---

## Feature Engineering (Silver → Gold)

Key engineered features per pitcher-game row:

| Feature | Logic |
|---|---|
| `velo_vs_season_avg` | `avg_velo - pitcher_season_avg_velo` |
| `whiff_rate_delta` | `whiff_rate - rolling_30d_whiff_rate` |
| `fastball_pct` | % pitches classified FF or SI |
| `stuff_diversity` | entropy of pitch_type distribution |
| `zone_rate` | pitches in zones 1-9 / total |
| `chase_rate` | swings on balls / balls |
| `barrel_rate_allowed` | barrels / batted balls |
| `high_leverage_csw` | CSW rate when delta_run_exp > 0.1 |

---

## LLM Enrichment Layer

After Gold tables are built, the enrichment module:
1. Pulls the top-N pitcher and batter rows for a given game
2. Constructs a structured prompt with stats
3. Calls Claude API
4. Parses and stores the response as a JSON blob in `llm_insights`

### Prompt Templates (see `enrichment/prompt_templates.py`)

**Pitcher Insight Prompt:**
```
You are a baseball analyst writing for an internal scouting system.

Given the following pitcher performance data for [PITCHER_NAME] on [DATE]:

Pitches thrown: {total_pitches}
Avg velocity: {avg_velo} mph (season avg: {season_avg_velo} mph)
Whiff rate: {whiff_rate:.1%} (league avg: 25.4%)
CSW rate: {csw_rate:.1%}
xwOBA allowed: {avg_xwoba_allowed:.3f}
Pitch mix: {pitch_mix_summary}

Respond in JSON with these exact keys:
- "performance_tier": one of ["elite", "above_avg", "average", "below_avg", "poor"]
- "headline": one sentence summary (max 15 words)
- "key_finding": 2-3 sentences on the most important statistical story
- "concern_flag": null OR one sentence on a red flag if present
- "comparable_game": historical comp if notable (optional)
```

**Anomaly Detection Prompt:**
```
You are a baseball data analyst reviewing game-level statistics.

The following pitcher metrics are statistical outliers (>2 SD from their 30-day mean):
{anomaly_list}

Identify which anomaly is most analytically significant and explain
why in 2-3 sentences. Be specific about what might have caused it
(fatigue, opposing lineup, pitch mix change, etc.)
```

---

## ML Model: Pitcher Effectiveness Prediction

**Goal:** Predict `csw_rate` (called strikes + whiffs / total pitches) for a pitcher's next game, given recent performance trends. CSW rate is widely accepted as the best single-game pitcher quality metric.

**Features (from Gold layer):**
```python
FEATURE_COLS = [
    "rolling_30d_avg_velo",
    "rolling_30d_whiff_rate",
    "rolling_30d_csw_rate",
    "velo_vs_season_avg",
    "stuff_diversity",
    "zone_rate",
    "chase_rate",
    "days_rest",
    "home_away",
    "opposing_team_k_rate",   # from schedule_ingestion
    "pitch_count_last_game",
    "barrel_rate_allowed_30d",
]
TARGET = "csw_rate"
```

**Model:** XGBoost Regressor
- Chosen for: tabular data performance, handles missing values, fast inference, SHAP-compatible
- Evaluation: 5-fold time-series cross-validation (no data leakage across dates)
- Metrics: RMSE, MAE, R²

**Evaluation Strategy:**
```python
# Time-series split — never train on future data
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    ...
```

**SHAP explainability** — every prediction includes feature attributions so the scouting report can say *why* a pitcher is projected to dominate.

---

## Scouting Report Engine

**Assembly logic** (`reports/scouting_report.py`):
1. Query Gold: get pitcher's last N games stats
2. Run `predict.py` → get `predicted_csw`, `stuff_score`, SHAP values
3. Pull LLM insight blob from `llm_insights`
4. Render Jinja2 HTML template → convert to PDF via `weasyprint`
5. Optionally return JSON for API/dashboard consumption

**Example Output:**
```
═══════════════════════════════════════════════════════
  BASEBALLIQ SCOUTING REPORT
  Generated: 2024-07-15  |  Confidential — Internal Use
═══════════════════════════════════════════════════════

PITCHER: Zack Wheeler  (#45, Philadelphia Phillies)
GAME: July 14, 2024 vs. Atlanta Braves

─── GAME PERFORMANCE ───────────────────────────────────
Pitches Thrown:      104
Avg Velocity:        95.8 mph  (↑ +0.9 vs. 30d avg)
Whiff Rate:          34.2%     (league avg: 25.4%)  ★ ELITE
CSW Rate:            33.7%     (league avg: 28.1%)  ★ ELITE
xwOBA Allowed:       .267      (league avg: .312)
Pitch Mix:           FF 54% | SL 28% | CH 13% | CU 5%

─── ML PREDICTION (next start) ─────────────────────────
Predicted CSW Rate:  31.4%   [↑ above-average projection]
Stuff Score:         87 / 100
Top SHAP driver:     rolling_whiff_rate (+0.024)

─── AI ANALYST SUMMARY ─────────────────────────────────
Performance Tier:    ELITE

Headline: "Wheeler's slider generated historic whiff rates
           against Atlanta's lineup."

Key Finding: Wheeler's slider produced a 52% whiff rate —
third-highest in a single game this season. His velocity
remained above 95 mph through the 7th inning, suggesting
no fatigue-related decline. Particularly effective pitching
to the outer third against left-handed hitters.

Concern Flag: None identified.
═══════════════════════════════════════════════════════
```

---

## Streamlit Dashboard Pages

| Page | Key Charts |
|---|---|
| **Pitcher Explorer** | Velocity trend line, pitch mix donut, whiff rate heatmap by zone, CSW sparklines |
| **Batter Explorer** | Exit velocity distribution, xwOBA rolling average, barrel rate by pitch type |
| **Team Trends** | Win probability over season, bullpen usage, rotation CSW trends |
| **AI Scouting Reports** | Player selector → renders full AI report inline |

**Stack:** Streamlit + Plotly + DuckDB (query directly from dashboard) + Anthropic API (on-demand generation)

---

## Engineering Best Practices

### Reproducibility
- `pyproject.toml` with pinned dependencies via `uv`
- `Makefile` targets: `make ingest DATE=2024-07-01`, `make train`, `make report PLAYER_ID=605483`
- `.env.example` for all secrets

### Testing
- `pytest` with fixtures loading small Bronze sample Parquet files
- Tests for: schema validation, feature shape, LLM response parsing, report rendering
- CI via GitHub Actions on push to `main`

### Documentation
- Every module has a docstring with: purpose, inputs, outputs, example usage
- `README.md` includes: architecture diagram, quickstart, data dictionary
- Inline comments explain non-obvious baseball domain logic

### GitHub Portfolio Readiness
- Clean commit history with conventional commits (`feat:`, `fix:`, `data:`)
- GitHub Actions badge in README (tests passing)
- Sample data included in `data/bronze/sample/` (1 game, ~300 pitches)
- Demo mode in Streamlit: runs on sample data without API keys

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/yourhandle/baseballiq
cd baseballiq
uv sync

# Set environment variables
cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

# Run full pipeline for one week of games
make ingest DATE_START=2024-07-01 DATE_END=2024-07-07
make clean
make gold
make enrich
make train

# Launch dashboard
make dashboard

# Generate a scouting report
make report PLAYER_ID=605483  # Zack Wheeler
```

---

## Why This Impresses Recruiters

| What They See | What It Signals |
|---|---|
| Medallion architecture (Bronze/Silver/Gold) | You understand data lakehouse patterns used at scale |
| DuckDB as analytical engine | Modern tooling awareness; not just pandas |
| Time-series CV on ML model | You understand leakage — a common interview filter |
| LLM enrichment as a layer, not the whole app | Mature AI integration vs. "I just called GPT" |
| SHAP explainability on predictions | You can explain model outputs to non-technical stakeholders |
| Statcast domain knowledge | Baseball-specific signal, not just generic DS |
| Modular ETL + Makefile | Production engineering mindset |
| Tests + CI | Software engineering discipline |
