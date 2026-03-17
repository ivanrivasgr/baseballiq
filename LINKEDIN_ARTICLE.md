# I Built a Production MLB Analytics Platform. Here's Every Architecture Decision I Made — and Why.

*A deep-dive into BaseballIQ: Statcast ingestion, DuckDB, XGBoost, and LLMs as a data enrichment layer.*

---

After spending time studying how analytics departments inside MLB organizations actually work, I built **BaseballIQ** — a production-grade sports analytics system that ingests MLB Statcast data, processes it through a medallion data architecture, trains a predictive model for pitcher effectiveness, and generates AI-powered scouting reports using Claude.

This post is not a tutorial. It's a write-up of every meaningful engineering decision I made, why I made it, and what the alternatives were.

**[GitHub](https://github.com/yourhandle/baseballiq) · [Live Demo](https://baseballiq.streamlit.app)**

---

## The Problem I Was Trying to Solve

Most sports analytics portfolio projects have the same structure: download a dataset, train a model, make a chart, done. They demonstrate familiarity with scikit-learn but say nothing about how data actually moves through a production system, why certain architectural choices age better than others, or how AI fits into an analytical workflow without becoming a liability.

I wanted to build something that mirrors what an internal analytics team at a baseball organization would actually ship — with real engineering judgment visible in every layer.

---

## The Data Layer: Why I Chose the Medallion Architecture

The system uses a classic **Bronze / Silver / Gold** tiered architecture:

- **Bronze** — Raw Statcast data as immutable Parquet files, partitioned by `game_date`. Nothing is modified here. If something downstream breaks, I can always re-derive everything from Bronze.
- **Silver** — Cleaned, typed, deduplicated data loaded into DuckDB. This is where schema enforcement and normalization happen.
- **Gold** — Aggregated analytical datasets: `pitcher_game_summary`, `batter_game_summary`. These are what the ML model and dashboard consume.

The key design constraint: each layer is a checkpoint. Bronze is the source of truth. Silver is the query layer. Gold is the business layer. If the Gold logic changes — say, I decide to redefine "stuff diversity" differently — I can rebuild Gold without re-downloading a single byte of Statcast data.

---

## Why DuckDB Instead of Pandas

This is the decision I get asked about most.

When people think "Python + data," they reach for Pandas. But for analytical queries — rolling averages, window functions, cross-pitcher percentile rankings — DuckDB is materially better for this use case:

**Speed.** DuckDB runs vectorized SQL on Parquet files. The window function to compute a 30-day rolling CSW rate across all pitchers in a season runs in ~400ms. The equivalent Pandas code, with groupby + rolling + shift to avoid lookahead, takes 8-12 seconds and requires careful index management.

**Readability.** The Gold aggregation query is 80 lines of SQL that any analyst on the team can read, audit, and modify. The equivalent Pandas code would be 200+ lines with nested groupby operations that require deep Pandas knowledge to understand.

**Parquet-native.** DuckDB reads and writes Parquet directly. No `pd.read_parquet → transform → df.to_parquet` round-trips. The pipeline is stateless — each stage reads from Parquet and writes to Parquet.

The tradeoff: DuckDB is less familiar to data scientists who live in Pandas. In a team environment, I'd document the SQL schema and make sure the query logic is well-commented. For a solo portfolio project, the performance and readability wins are decisive.

---

## The Machine Learning Component: Getting the Target Right

Most pitcher ML projects predict ERA or WHIP. Both are wrong targets for a game-level prediction model.

**ERA** is heavily influenced by sequencing, defense, and strand rates — factors a pitcher controls only partially. **WHIP** mixes two different outcome types (hits and walks) without weighting them. Both are also highly variable game-to-game.

**CSW rate** (called strikes + whiffs / total pitches) is my target because:
1. It measures what the pitcher directly controls: inducing chases and weak contact
2. It stabilizes faster than ERA (meaningful at ~40 pitches; ERA needs ~100 innings)
3. Modern front offices use it as their primary in-game pitcher quality indicator
4. It has a strong predictive relationship with next-game outcomes

The features are all about trending: 30-day rolling CSW, velocity delta vs. baseline, whiff rate momentum. The model answers: *given how this pitcher has been pitching recently, how dominant will they be in their next start?*

### The Cross-Validation Decision That Matters Most

I used `TimeSeriesSplit` for cross-validation, not standard k-fold. This is the most commonly violated principle in sports analytics ML projects.

With k-fold cross-validation, the model might train on July games and validate on April games. In production, you never have future data. `TimeSeriesSplit` ensures every validation fold is strictly after its training fold — mimicking real-world inference conditions.

The RMSE difference between k-fold and time-series CV on this dataset is ~0.004. That sounds small, but it's the difference between a model that genuinely generalizes and one that's memorized seasonal patterns in the training data.

---

## The LLM Layer: What I Got Right and What I Got Wrong (Initially)

My first instinct was to use the LLM as a core analytical layer — have Claude analyze raw statistics and generate insights directly. This was wrong, and understanding why is important.

**The problem with LLM-first analytics:**
- LLMs are unreliable calculators. Asking Claude to compute batting averages or compare percentiles produces plausible-sounding but occasionally incorrect numbers.
- The outputs are not auditable. When the GM asks "why does this report say the barrel rate was 12.4%?", you need to trace that to a DuckDB query, not to a language model's reasoning.
- It's expensive to call at query time for every statistic.

**The correct architecture:** LLM as enrichment layer.

Every statistic in the scouting report is computed deterministically by the pipeline before the LLM ever gets involved. The prompt I send to Claude looks like this:

```
Pitcher: Marcus Delgado
CSW Rate: 32.1% (league avg: 28.1%)
Whiff Rate: 33.8% (league avg: 25.4%)
Velocity delta vs. 30d avg: +0.7 mph
xwOBA allowed: .261
[...more stats...]

Respond in JSON with: performance_tier, headline, key_finding, concern_flag
```

Claude's job is to write the narrative that connects these numbers into an analytical story — identifying *which* statistic matters most, *what* it implies for upcoming scouting, and *whether* any combination of metrics constitutes a flag worth escalating.

This is what LLMs are genuinely good at: synthesis and narrative from structured context. It's also auditable — every number in the LLM output is traceable to a DuckDB query.

The one thing I'd do differently: I'd add structured output validation (Pydantic or similar) to enforce that the LLM response JSON always contains the required keys and reasonable value ranges. In the current system, a malformed JSON response fails gracefully, but validation would make it more robust.

---

## The Anomaly Detection Layer

One of the more interesting features is automated anomaly detection. Before generating insights, the pipeline computes Z-scores for key metrics against each pitcher's 30-day baseline:

```sql
(whiff_rate - rolling_30d_whiff_rate)
    / NULLIF(STDDEV(whiff_rate) OVER (PARTITION BY pitcher_id), 0) AS whiff_z
```

Any pitcher with a metric 2+ standard deviations from their baseline — and at least 40 pitches (minimum meaningful sample) — gets flagged for LLM analysis. The prompt asks Claude to identify the most analytically significant anomaly and explain possible causes: fatigue, mechanical change, pitch mix shift, or opposing lineup quality.

The key design principle here: **the statistical identification is done in SQL, not by the LLM**. Claude receives a list of human-readable anomalies and explains them. It doesn't decide which games are anomalous.

---

## Deployment: Free Tier Without Compromising the Architecture

Deploying to Streamlit Community Cloud (free tier, 1GB RAM) created a real constraint. The live `pybaseball` download approach — ingesting Statcast data at app startup — would timeout within 30 seconds. Loading a full DuckDB database with a season of data would exhaust available memory.

The solution: **pre-generated Gold Parquet files committed to the repository.**

The demo ships with ~600 pitcher-game rows and ~500 batter-game rows generated from distributions calibrated to real 2024 Statcast data. These load in ~200ms. The full pipeline (`make ingest`, `make clean`, `make gold`) is available for anyone who wants to run it locally with real Statcast data.

The live LLM component uses `st.secrets` to access the Anthropic API key, so the scouting report generation works in real time. The split: cached reports for the pre-generated data, live generation for any game the user selects.

This is actually a reasonable production pattern: pre-aggregate expensive computations, cache results, expose live AI generation for on-demand requests.

---

## What I Learned

**The hardest part wasn't the ML model.** It was the feature engineering — specifically making sure the rolling features had no lookahead leakage. Window functions that include the current game in the rolling average create a subtle but fatal bug: the model trains on information it wouldn't have had at prediction time.

**SHAP values are non-negotiable for sports analytics.** The scouting report doesn't just say "projected CSW: 30.8%." It says "the top driver of this prediction is rolling_30d_whiff_rate (+0.018)." This is the difference between a model an analyst will trust and one they'll ignore.

**Domain knowledge is a multiplier.** Choosing CSW rate over ERA, understanding that whiff rate stabilizes faster than other metrics, knowing that barrel rate is the right proxy for contact quality — these decisions came from studying how analytics departments actually think, not from the data itself. The engineering was straightforward once the domain framing was right.

**The LLM layer is genuinely useful — when it's scoped correctly.** The scouting reports are better with AI-generated narrative than without. But only because the system never asks the LLM to be a calculator. Scoping AI to synthesis and language, not computation, is the insight that made the integration work.

---

## What's Next

The obvious extensions:
- **Bullpen analytics**: extend the pitcher model to relief appearances (different feature set — leverage index, inherited runners)
- **Opponent-adjusted metrics**: factor in opposing lineup quality for context-adjusted CSW projections
- **Real-time ingestion**: replace the batch Statcast pull with a live feed during the season
- **Report versioning**: track how LLM-generated assessments change as a pitcher's season progresses

The repo is fully open — feel free to use it as a starting point, extend it, or reference the architecture.

**[GitHub → BaseballIQ](https://github.com/yourhandle/baseballiq)**

---

*If you're building something in sports analytics or have questions about any of the architecture decisions, I'd genuinely like to hear from you.*

---

*Tags: #MLBanalytics #DataEngineering #MachineLearning #LLM #Python #DuckDB #Streamlit #SportsAnalytics*
