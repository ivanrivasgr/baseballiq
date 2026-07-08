# MLB Home Run Prop Prediction System — Build Plan

**Status:** Planning document. This is the spec we build against, layer by layer.
**Base:** Adapts the existing BaseballIQ pitcher-CSW pipeline (Bronze/Silver/Gold medallion,
DuckDB, XGBoost + SHAP, LLM narrator) to a batter-matchup HR classification target,
and adds two new layers: odds→edge and parlay/confidence tiering.

---

## 0. Core invariants (non-negotiable)

1. **Temporal integrity.** Every feature for a row dated `game_date` is computed from data
   strictly **before** `game_date`. All DuckDB rolling windows end at
   `INTERVAL 1 DAY PRECEDING`. Validation is `TimeSeriesSplit` on date-ordered data — never
   shuffled, never K-fold.
2. **LLM computes nothing.** The narrator receives already-computed numbers and writes prose.
   No number in any output originates from the LLM.
3. **Calibrated probabilities or no edge.** Raw XGBoost scores are not probabilities you can
   price against a sportsbook. A calibration step is part of the model artifact, and edge is
   only ever computed from calibrated output. (See §4.)

---

## 1. Project structure — pitcher pipeline → HR-prop pipeline

Legend: `KEEP` = reuse nearly as-is · `ADAPT` = same skeleton, new logic · `NEW` = doesn't exist in repo.

```
baseballiq/
├── pipeline/
│   ├── config.py                        # ADAPT: add park-factor path, odds paths, HR league
│   │                                    #        baselines, stadium lat/lon + CF bearing table
│   ├── orchestrator.py                  # ADAPT: new steps — schedule/weather ingestion,
│   │                                    #        PA table, matchup Gold build
│   ├── ingestion/
│   │   ├── statcast_ingestion.py        # KEEP, extend KEEP_COLS (see §1a)
│   │   ├── schedule_ingestion.py        # NEW: MLB Stats API — probable starters, lineups,
│   │   │                                #      game start time (day/night), venue, roof state
│   │   └── weather_ingestion.py         # NEW: temp, wind speed/direction per game
│   │                                    #      (Open-Meteo by stadium lat/lon at first pitch)
│   ├── silver/
│   │   ├── cleaning.py                  # KEEP: typing/dedup of pitches table
│   │   ├── plate_appearances.py         # NEW: collapse pitches → one row per PA, with
│   │   │                                #      is_hr flag, batter/pitcher ids, handedness,
│   │   │                                #      batted-ball type, spray angle
│   │   └── schema.sql                   # ADAPT: add plate_appearances, games, weather tables
│   └── gold/
│       ├── batter_features.py           # NEW: rolling batter form (replaces pitcher CSW aggs)
│       ├── pitcher_features.py          # NEW: opposing-starter HR susceptibility (rolling)
│       ├── context_features.py          # NEW: park factor, platoon, weather join, day/night
│       └── matchup_table.py             # NEW: assembles batter_matchup_features
│                                        #      (one row = batter × game; target = hr_hit)
│
├── models/
│   ├── features.py                      # NEW: FEATURE_COLS registry + leakage assertions
│   ├── train.py                         # ADAPT: XGBClassifier (binary:logistic), date-grouped
│   │                                    #        TimeSeriesSplit, calibration fit (§4)
│   ├── evaluate.py                      # NEW: Brier, log-loss, reliability curves,
│   │                                    #      per-month backtest, flat-stake ROI sim
│   ├── predict.py                       # ADAPT: → calibrated P(HR) + top-k SHAP drivers
│   └── artifacts/hr_prop_v1.pkl         # model + calibrator + feature list + metadata
│
├── betting/                             # ★ NEW LAYER 1+2 (not in repo)
│   ├── odds.py                          # american/decimal ↔ implied prob; devig (two-way)
│   ├── edge.py                          # model prob vs market prob → edge, EV, Kelly fraction
│   ├── parlay.py                        # multi-leg combiner; same-game correlation flags
│   ├── tiers.py                         # Voss Edge L1–L5 hooks (config-driven rubric)
│   └── odds_loader.py                   # source adapters: CSV first, API adapter interface
│
├── enrichment/
│   └── llm_narrator.py                  # ADAPT llm_enrichment.py: HR prop scouting card
│                                        # narrative FROM computed numbers only (rule §0.2)
│
├── reports/
│   └── prop_card.py                     # ADAPT scouting_report.py: per-batter card —
│                                        # prob, market line, edge, tier, SHAP drivers, narrative
│
├── dashboard/                           # ADAPT later (optional v1): slate view, edge board
├── data/{bronze,silver,gold}/           # KEEP layout; new gold parquet: batter_matchup_features
└── tests/
    ├── test_leakage.py                  # NEW: the most important test file (see §3c)
    ├── test_odds.py                     # NEW: conversion + devig round-trips
    └── test_features.py                 # ADAPT
```

### 1a. Bronze schema extension (`KEEP_COLS` additions)

The current ingestion drops columns we need. Add:

| Column | Why |
|---|---|
| `at_bat_number`, `pitch_number` | Define PA grain (last pitch of each at-bat = PA outcome) |
| `bb_type` | fly_ball / ground_ball / line_drive / popup → fly-ball % |
| `hc_x`, `hc_y` | Hit coordinates → spray angle → pull % (with `stand`) |
| `if_fielding_alignment` (optional) | shift context, cheap to keep |

Historical backfill re-pulls affected dates; `pybaseball` caching makes this tolerable.

---

## 2. Gold layer: `batter_matchup_features` (DuckDB)

**Grain:** one row = one batter in one game. **Target:** `hr_hit` (1 if any PA that game ended
in a home run). Positive rate will be roughly 6–9% — see §3 for how we handle that.

### 2a. Silver prerequisite: `plate_appearances`

Collapse `pitches` to PA grain: for each `(game_pk, at_bat_number)`, take the final pitch
(`MAX(pitch_number)`) — its `events` is the PA outcome.

```sql
CREATE OR REPLACE TABLE plate_appearances AS
SELECT
    game_pk, game_date, at_bat_number,
    batter_id, pitcher_id, stand, p_throws,
    inning, inning_topbot,
    events,
    (events = 'home_run')::INT                        AS is_hr,
    (launch_speed IS NOT NULL)::INT                   AS is_batted_ball,
    launch_speed, launch_angle, is_barrel, bb_type,
    -- spray angle: negative = pulled for RHB; flip sign for LHB so
    -- pull is always positive
    CASE WHEN hc_x IS NOT NULL THEN
        DEGREES(ATAN2(hc_x - 125.42, 198.27 - hc_y)) * (CASE stand WHEN 'L' THEN -1 ELSE 1 END)
    END                                               AS spray_angle_adj
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY game_pk, at_bat_number ORDER BY pitch_number DESC) AS rn
    FROM pitches
) WHERE rn = 1;
```

### 2b. Batter form block (rolling, strictly historical)

All windows use the repo's proven pattern —
`RANGE BETWEEN INTERVAL N DAYS PRECEDING AND INTERVAL 1 DAY PRECEDING` — partitioned by
batter, ordered by `game_date`. Computed over **batted balls / PAs**, not games, so we
aggregate PA-level rows with SUM/COUNT windows rather than averaging per-game rates
(avoids small-denominator noise).

| Feature | Definition (window) |
|---|---|
| `b_barrel_rate_30d` | barrels / batted balls, 30d |
| `b_avg_ev_30d`, `b_max_ev_30d` | mean / max `launch_speed`, 30d |
| `b_hard_hit_30d` | EV ≥ 95 / batted balls, 30d |
| `b_fb_rate_30d` | `bb_type='fly_ball'` / batted balls, 30d |
| `b_pull_fb_rate_30d` | pulled (spray_angle_adj < −15°) fly balls / fly balls, 30d — pulled FBs are where HRs live |
| `b_hr_per_pa_30d`, `b_hr_per_pa_7d` | HR pace, two horizons |
| `b_hr_per_pa_365d` | long-horizon skill prior |
| `b_pa_count_30d` | sample-size signal (also feeds shrinkage) |
| `b_hr_per_pa_shrunk` | `(n30·rate30 + k·rate365) / (n30 + k)` — empirical-Bayes blend so a 2-HR week on 20 PA doesn't scream |
| `b_avg_pa_per_game_30d` | expected-exposure proxy (see §2e) |

Platoon splits: compute `b_hr_per_pa_vs_hand_365d` (batter's HR rate vs the starter's
throwing hand, long window only — 30d platoon splits are pure noise).

### 2c. Opposing-starter block

Joined via `schedule_ingestion` (probable starter per game). Same window pattern,
partitioned by pitcher:

| Feature | Definition |
|---|---|
| `p_hr_per_pa_allowed_30d` / `_365d` | HR allowed per batter faced (more stable than HR/9 — no innings dependency) |
| `p_barrel_rate_allowed_30d` | barrels allowed / batted balls |
| `p_fb_rate_allowed_30d` | fly balls allowed / batted balls |
| `p_avg_ev_allowed_30d` | contact quality allowed |
| `p_hand` → `same_handed` | platoon flag (batter stand == pitcher throws) |
| `bp_hr_per_pa_allowed_30d` | **opposing team bullpen** HR/PA — the starter only faces a batter ~2–3 times; ignoring the bullpen misprices ~40% of the batter's exposure |

(HR/9 can be emitted alongside for display/familiarity, but the model uses per-PA rates.)

### 2d. Context block

| Feature | Source |
|---|---|
| `park_hr_factor` | static seed table (published 3-yr handedness-aware park factors, e.g. Statcast park factors), keyed by venue × batter hand. Static = no leakage risk. Recomputing from our own data is a v2 option (train-years-only). |
| `is_home` | schedule |
| `day_night` | game start time from schedule |
| `temp_f` | weather ingestion at first-pitch hour |
| `wind_out_mph` | `wind_speed × cos(wind_bearing − park_cf_bearing)` — one signed "blowing out" scalar instead of raw direction; requires a small static stadium-orientation table |
| `roof_closed` | flag; when closed/dome, zero out weather features |
| `lineup_slot` | if available at prediction time (see §2e) |

### 2e. The exposure problem (called out explicitly)

The #1 confounder for per-game HR probability is **how many PAs the batter gets**. A leadoff
hitter gets ~4.7 PA/game, a #9 hitter ~3.7. Two options:

- **v1 (chosen):** direct per-game classifier with `lineup_slot` and
  `b_avg_pa_per_game_30d` as features. Simple, matches the repo skeleton.
  ⚠️ Never use the *actual* PA count from the game itself — that's an in-game outcome (leakage).
- **v2 (recommended upgrade):** model **per-PA** HR probability, then aggregate:
  `P(HR in game) = 1 − (1 − p_pa)^E[PA]`. Cleaner decomposition, better calibrated across
  lineup slots, and lets one model serve pinch-hit / partial-game questions. The Gold layer
  above already supports this — `plate_appearances` is the training grain, matchup features
  join the same way.

We build v1 first; the schema is designed so v2 is a training-script change, not a pipeline change.

---

## 3. Training + validation

### 3a. Model

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",          # NOT accuracy/AUC-driven early stopping
    n_estimators=600, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
)
```

- **Class imbalance:** keep natural class rates (no `scale_pos_weight`, no under/oversampling).
  We need honest probabilities, not balanced classes — reweighting distorts the probability
  scale and would have to be undone at calibration anyway.
- **SHAP:** `TreeExplainer` per prediction; top-3 signed drivers stored with every prediction
  row and passed verbatim to the narrator and prop card.

### 3b. TimeSeriesSplit — done on dates, not rows

`TimeSeriesSplit` over raw row indices can split a single date across train/val. Split over
**unique sorted game_dates**, then map back to rows:

```python
dates = np.sort(df["game_date"].unique())
tscv = TimeSeriesSplit(n_splits=5)
for train_d, val_d in tscv.split(dates):
    train = df[df.game_date.isin(dates[train_d])]
    val   = df[df.game_date.isin(dates[val_d])]
```

Every fold trains on strictly earlier dates. No shuffling anywhere; `shuffle=False` is not
enough — there is simply no code path that permutes rows.

### 3c. Leakage guardrails (the checklist)

1. **Window audit:** every Gold window ends `INTERVAL 1 DAY PRECEDING`. Grep-able convention;
   `tests/test_leakage.py` scans the SQL for any window missing the bound.
2. **No partition-wide aggregates.** The existing repo computes
   `AVG(x) OVER (PARTITION BY pitcher_id)` as "season average" — that includes **future
   games** and must not be copied into any model feature. (Fine for dashboards; banned in
   `FEATURE_COLS`.)
3. **Recompute test:** for sampled rows, `test_leakage.py` recomputes each rolling feature
   using only `plate_appearances WHERE game_date < row.game_date` and asserts equality with
   the Gold value.
4. **No same-game outcomes as features:** actual PA count, actual pitcher pulled early, final
   score — all in-game information, all banned. `features.py` is the single allowlist;
   training refuses columns not registered there.
5. **Feature-availability timestamp:** each feature in `features.py` declares when it's
   knowable (`T-1day`, `lineup-release`, `first-pitch`). Training for a given prediction
   window only uses features available at that time.
6. **Fold date-range logging** (repo already does this) so any overlap is visible in logs.

### 3d. Calibration + evaluation (new vs repo, and mandatory)

- Reserve the **final** time slice (e.g., last ~20% of dates) as a calibration+holdout block:
  fit isotonic regression (or Platt if the block is small) on its first half, evaluate on its
  second half. The calibrator ships inside the model artifact; `predict.py` output is always
  calibrated.
- **Metrics that matter here:** log loss and **Brier score** primary; reliability curve
  (predicted decile vs observed HR rate) is the go/no-go artifact; AUC secondary.
  Accuracy is meaningless at a 7% base rate.
- **Betting backtest:** on the holdout, simulate flat-stake bets where
  `model_prob − devigged_market_prob > threshold`, sweep thresholds, report ROI and volume.
  This — not RMSE — is the KPI for this system.

---

## 4. New layer 1: odds → edge (`betting/`)

```
american odds ──► decimal ──► implied prob (w/ vig) ──► devig ──► market prob
                                                                    │
model calibrated P(HR) ────────────────────────────────────────────┤
                                                                    ▼
                                              edge = model_p − market_p
                                              EV/$1 = model_p·(dec−1) − (1−model_p)
                                              kelly = edge / (dec − 1)
```

- `odds.py`: pure functions, round-trip tested. American ↔ decimal ↔ implied.
- **Devig:** HR props are two-way markets (Yes/No). When both sides are available, remove
  overround (multiplicative default; power/Shin methods behind the same interface for later).
  When only "Yes" is posted, fall back to a configurable assumed-vig haircut — flagged as
  lower-confidence in the tier layer.
- `odds_loader.py`: `OddsSource` adapter interface. v1 = **The Odds API** adapter
  (quota-aware, response-cached — see §10); a CSV loader
  (`date, batter, book, line_american, side`) ships alongside as the offline/test source.
  Edge logic never touches a source directly, only the adapter interface.
- Output table `gold/prop_edges.parquet`: one row per batter × game × book —
  `model_p, market_p_devig, edge, ev, kelly_frac, book, line`.

## 5. New layer 2: parlay + Voss Edge tiers (`betting/parlay.py`, `betting/tiers.py`)

- **Parlay math:** combined fair prob = ∏ leg model probs; combined decimal = ∏ leg decimals;
  parlay EV from the two. **Correlation flag:** legs from the same game (or same-team stack
  vs one starter) are NOT independent — v1 doesn't model the correlation, it **labels** the
  parlay `correlated=True` and caps its tier. Honest flag > fake precision.
- **Voss Edge L1–L5 hooks:** `tiers.py` exposes
  `assign_tier(edge, model_p, market_p, sample_flags, data_completeness) -> Tier`.
  The rubric lives in `config/voss_edge.yaml` (thresholds on edge size, calibration-window
  confidence, feature completeness, devig quality) so your proprietary notation is
  configuration, not code — you can tune L1–L5 cutoffs without touching the pipeline.
  Every tier assignment stores the rubric inputs alongside it (auditable, and the narrator
  can explain *why* something is L4).

## 6. LLM narrator (adapted rule, kept absolute)

Same architecture as the repo's enrichment layer: structured prompt, numbers injected,
JSON out, stored in Gold. The prop-card prompt receives: calibrated prob, market prob, edge,
tier, top-3 SHAP drivers (name + signed value), and the raw feature values behind them.
Schema-validated response with keys like `headline`, `matchup_story`, `risk_note`.
Any number appearing in the narrative is string-formatted **by us before the prompt**;
the LLM never sees a request to compute, sum, or compare beyond wording.

---

## 7. Build order (layer by layer, as agreed)

1. Bronze/Silver: `KEEP_COLS` extension, `plate_appearances`, schedule ingestion
2. Gold: batter block → pitcher block → context block → `matchup_table`
3. Model: `features.py` registry + `train.py` + `test_leakage.py` + calibration + evaluate
4. Betting: `odds.py` + `edge.py` (CSV loader) → `parlay.py` + `tiers.py`
5. Narrator + prop card
6. Dashboard (optional)

## 8. Decisions (confirmed)

1. **Backfill window:** 2022–2025 (consistent ball/rules era, ~2.8M pitches).
2. **Odds source v1:** The Odds API behind the `OddsSource` adapter (`ODDS_API_KEY` env var;
   player-props tier required). Quota-aware client — see §10.
3. **Voss Edge L1–L5:** config-driven rubric in `config/voss_edge.yaml`; cutoffs designed
   from holdout backtest results once they exist.
4. **Prediction time:** morning of slate — probable starters known, `lineup_slot` imputed
   from recent games. Feature registry only exposes `T-1day`-available features.
5. **Weather scope v1:** temp + wind-out scalar + roof flag (Open-Meteo, free).

## 9. Talent-backed QA gates (from the `agency-talent` repo)

Three talents from the VelozLabs `agency-talent` library are converted into project
subagents under `.claude/agents/`, pinned to cheap models, and used as mandatory review
gates between build layers:

| Subagent | Source talent | Model | Gate |
|---|---|---|---|
| `model-qa` | `specialized-model-qa` | sonnet | After model layer: independently re-verify leakage recompute test, reliability curve, fold date ranges — sign-off required before betting layer |
| `pipeline-reviewer` | `engineering-data-engineer` | sonnet | After Gold layer: audit every window for `INTERVAL 1 DAY PRECEDING`, schema/row-count sanity |
| `odds-qa` | `testing-api-tester` | haiku | After betting layer: adapter fixture tests, devig round-trips, quota accounting |

`agency-talent` stays read-only source; each agent file cites its source talent id.

## 10. Cost posture

- **Odds:** every The Odds API response cached to `data/bronze/odds/date=*/` before parsing;
  only the `batter_home_runs` market, configurable bookmaker filter; one snapshot per slate
  morning (optional pre-lock snapshot behind a flag, default off); client tracks remaining
  quota from response headers and hard-stops at a floor. **No paid historical-odds backfill** —
  model quality is validated on Statcast alone; betting-ROI backtests run on forward-collected
  snapshots (`snapshot_ts` in the `prop_edges` schema from day one).
- **Narrator:** top-N edge plays only (default 10/slate) on Haiku, optional Batch API flag;
  full slate always available as numbers in `prop_edges.parquet`.
- **Everything else free/local:** DuckDB + Parquet single-node (no warehouse, no GPU),
  pybaseball cached incremental ingestion, MLB Stats API + Open-Meteo.

## 11. Infrastructure (recommended, cost-first)

| Concern | Recommendation | Cost |
|---|---|---|
| Durable parquet lake | Cloudflare R2 free tier (10 GB, zero egress); DuckDB reads via `httpfs`, sync with `rclone` | $0 |
| Daily slate run (~9am ET) | GitHub Actions scheduled workflow: pull gold from R2 → ingest increment → score → odds snapshot → edges/cards to R2 | $0 |
| Fallback runner | Small VPS (~$5/mo) with cron + persistent pybaseball cache, only if GH Actions gets cramped | ~$5/mo |
| Retraining | Monthly, CPU-only, GH Actions manual dispatch or local | $0 |
| Dashboard | Railway (already wired via `railway.toml`) or Streamlit Community Cloud, reading from R2 | $0–5/mo |
| Secrets | GH Actions secrets / Railway env vars: `ODDS_API_KEY`, `ANTHROPIC_API_KEY`; local `.env` | — |

Expected infra cost ~$0–10/mo; the real spend is The Odds API subscription + pennies of
Haiku narration, both capped by §10.
