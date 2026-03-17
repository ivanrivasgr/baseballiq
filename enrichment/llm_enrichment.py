"""
enrichment/llm_client.py  +  enrichment/insight_writer.py
==========================================================
LLM Enrichment Layer — generates structured analytical insights
from Gold-layer statistics using the Anthropic Claude API.

Design principles:
    - LLM is the LAST step, after all data is computed
    - Prompts inject real numbers; LLM provides narrative + tier labels
    - Outputs are structured JSON stored back in Gold layer
    - Anomaly detection uses Z-score pre-filtering (LLM doesn't compute stats)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import anthropic
import duckdb
import pandas as pd

from pipeline.config import DUCKDB_PATH, GOLD_DIR

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS    = 600
RATE_LIMIT_S  = 0.5    # seconds between API calls (respect rate limits)

LEAGUE_AVG = {
    "whiff_rate": 0.254,
    "csw_rate":   0.281,
    "barrel_rate_allowed": 0.078,
    "avg_xwoba_allowed":   0.312,
}

# ─── Prompt Templates ───────────────────────────────────────────────────────────

PITCHER_INSIGHT_PROMPT = """\
You are a baseball analyst writing for an internal scouting system used by MLB front offices.

Pitcher: {pitcher_name}
Date: {game_date}
Opponent: {opponent}

Performance data (this game):
  Pitches thrown:    {total_pitches}
  Avg velocity:      {avg_velo} mph  (season avg: {season_avg_velo} mph, delta: {velo_delta:+.1f})
  Whiff rate:        {whiff_rate:.1%}  (league avg: 25.4%)
  CSW rate:          {csw_rate:.1%}  (league avg: 28.1%)
  Zone rate:         {zone_rate:.1%}
  Chase rate:        {chase_rate:.1%}
  xwOBA allowed:     {avg_xwoba_allowed:.3f}  (league avg: .312)
  Barrel rate:       {barrel_rate_allowed:.1%}
  Stuff diversity:   {stuff_diversity:.3f}  (higher = more varied mix)
  Run value delta:   {total_re_delta:+.2f} runs

Respond ONLY with valid JSON — no preamble, no markdown, no explanation outside the JSON.

{{
  "performance_tier": "<elite|above_avg|average|below_avg|poor>",
  "headline": "<one sentence, max 15 words>",
  "key_finding": "<2-3 sentences: most important statistical story, specific and analytical>",
  "concern_flag": "<null or one sentence on a meaningful red flag>",
  "pitch_mix_note": "<one sentence on pitch mix or velocity trend if notable, else null>"
}}
"""

ANOMALY_INSIGHT_PROMPT = """\
You are a baseball data scientist reviewing statistical anomalies from today's games.

The following pitcher-game metrics are outliers (Z-score > 2.0 vs. that pitcher's 30-day baseline):

{anomaly_list}

Select the single most analytically interesting anomaly and explain it in 2-3 sentences.
Consider potential causes: fatigue, pitch mix change, opposing lineup quality, mechanics shift,
weather, or sample size caveats. Be precise — cite the actual numbers.

Respond ONLY with valid JSON:
{{
  "pitcher_name": "<name>",
  "anomaly_metric": "<metric name>",
  "observed_value": <number>,
  "baseline_value": <number>,
  "explanation": "<2-3 sentence analytical explanation>"
}}
"""

GAME_SUMMARY_PROMPT = """\
You are a baseball analyst writing a post-game intelligence brief for the front office.

Game: {home_team} vs {away_team}, {game_date}
Final: {home_score} - {away_score}

Top performers (pitchers by CSW rate):
{pitcher_table}

Top performers (batters by xwOBA):
{batter_table}

Write a 3-4 sentence game intelligence summary. Focus on the most statistically significant
performance, what it means in context, and one forward-looking scouting observation.

Respond ONLY with valid JSON:
{{
  "game_headline": "<one sentence>",
  "game_summary": "<3-4 sentences>",
  "scouting_flag": "<one forward-looking observation for the front office>"
}}
"""


# ─── LLM Client ────────────────────────────────────────────────────────────────

@dataclass
class InsightResult:
    pitcher_id: int
    game_pk: int
    game_date: str
    insight_type: str
    raw_json: dict[str, Any]


class LLMClient:
    """Thin wrapper around Anthropic client with retry + rate limiting."""

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    def generate(self, prompt: str, insight_type: str) -> dict[str, Any]:
        """Call Claude API and return parsed JSON response."""
        time.sleep(RATE_LIMIT_S)

        for attempt in range(3):
            try:
                message = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw_text = message.content[0].text.strip()
                return json.loads(raw_text)

            except json.JSONDecodeError as e:
                logger.warning("JSON parse error (attempt %d): %s", attempt + 1, e)
                if attempt == 2:
                    return {"error": "json_parse_failed", "raw": raw_text}
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning("Rate limit hit — sleeping %ds", wait)
                time.sleep(wait)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return {"error": str(e)}

        return {"error": "max_retries_exceeded"}


# ─── Insight Writer ─────────────────────────────────────────────────────────────

class InsightWriter:
    """
    Reads Gold pitcher_game_summary, calls LLM for each game row,
    and writes results to llm_insights table in DuckDB + Gold Parquet.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con    = con
        self.client = LLMClient()

    def enrich_game(self, game_pk: int, game_date: str) -> list[InsightResult]:
        """Generate LLM insights for all pitchers in a given game."""
        rows = self.con.execute("""
            SELECT
                pgs.*,
                p.full_name AS pitcher_name,
                CASE WHEN g.home_pitcher_id = pgs.pitcher_id
                     THEN g.away_team ELSE g.home_team END AS opponent,
                g.home_score,
                g.away_score,
                g.home_team,
                g.away_team
            FROM pitcher_game_summary pgs
            LEFT JOIN players p      ON p.player_id = pgs.pitcher_id
            LEFT JOIN games g        ON g.game_pk   = pgs.game_pk
            WHERE pgs.game_pk = ? AND pgs.game_date = ?
            ORDER BY pgs.csw_rate DESC
        """, [game_pk, game_date]).df()

        results = []
        for _, row in rows.iterrows():
            insight = self._generate_pitcher_insight(row)
            results.append(InsightResult(
                pitcher_id=int(row.pitcher_id),
                game_pk=game_pk,
                game_date=game_date,
                insight_type="pitcher_game",
                raw_json=insight,
            ))

        self._write_insights(results)
        return results

    def detect_anomalies(self, game_date: str) -> dict[str, Any] | None:
        """
        Find statistical outliers (Z > 2.0 vs 30d baseline) in today's games
        and generate an LLM explanation of the most notable one.
        """
        anomalies = self.con.execute("""
            SELECT
                pitcher_id,
                pitcher_name,
                game_date,
                whiff_rate,
                rolling_30d_whiff_rate,
                (whiff_rate - rolling_30d_whiff_rate)
                    / NULLIF(STDDEV(whiff_rate) OVER (PARTITION BY pitcher_id), 0) AS whiff_z,
                csw_rate,
                rolling_30d_csw_rate,
                avg_velo,
                rolling_30d_avg_velo,
                velo_vs_30d_avg
            FROM pitcher_game_summary
            WHERE game_date = ?
              AND ABS(whiff_rate - rolling_30d_whiff_rate) > 0.08
              AND total_pitches >= 40   -- meaningful sample
            ORDER BY ABS(whiff_rate - rolling_30d_whiff_rate) DESC
            LIMIT 5
        """, [game_date]).df()

        if anomalies.empty:
            logger.info("No significant anomalies detected for %s", game_date)
            return None

        anomaly_text = "\n".join([
            f"- {r.pitcher_name}: whiff_rate {r.whiff_rate:.1%} "
            f"(30d avg: {r.rolling_30d_whiff_rate:.1%}, "
            f"velo delta: {r.velo_vs_30d_avg:+.1f} mph)"
            for _, r in anomalies.iterrows()
        ])

        prompt  = ANOMALY_INSIGHT_PROMPT.format(anomaly_list=anomaly_text)
        result  = self.client.generate(prompt, "anomaly")
        logger.info("Anomaly insight generated for %s", game_date)
        return result

    def _generate_pitcher_insight(self, row: pd.Series) -> dict[str, Any]:
        """Build prompt for one pitcher-game row and call LLM."""
        prompt = PITCHER_INSIGHT_PROMPT.format(
            pitcher_name       = row.get("pitcher_name", f"Pitcher #{row.pitcher_id}"),
            game_date          = row.game_date,
            opponent           = row.get("opponent", "Unknown"),
            total_pitches      = int(row.total_pitches),
            avg_velo           = round(row.avg_velo, 1),
            season_avg_velo    = round(row.season_avg_velo, 1),
            velo_delta         = row.velo_vs_30d_avg or 0.0,
            whiff_rate         = row.whiff_rate or 0.0,
            csw_rate           = row.csw_rate or 0.0,
            zone_rate          = row.zone_rate or 0.0,
            chase_rate         = row.chase_rate or 0.0,
            avg_xwoba_allowed  = row.avg_xwoba_allowed or 0.0,
            barrel_rate_allowed= row.barrel_rate_allowed or 0.0,
            stuff_diversity    = row.stuff_diversity or 0.0,
            total_re_delta     = row.total_re_delta or 0.0,
        )
        return self.client.generate(prompt, "pitcher_game")

    def _write_insights(self, results: list[InsightResult]) -> None:
        """Persist insights to DuckDB llm_insights table."""
        if not results:
            return

        rows = [
            {
                "pitcher_id":   r.pitcher_id,
                "game_pk":      r.game_pk,
                "game_date":    r.game_date,
                "insight_type": r.insight_type,
                "insight_json": json.dumps(r.raw_json),
                "generated_at": pd.Timestamp.utcnow().isoformat(),
            }
            for r in results
        ]

        df = pd.DataFrame(rows)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS llm_insights (
                pitcher_id   INTEGER,
                game_pk      INTEGER,
                game_date    DATE,
                insight_type VARCHAR,
                insight_json VARCHAR,
                generated_at VARCHAR,
                PRIMARY KEY (pitcher_id, game_pk, insight_type)
            )
        """)
        self.con.register("_insight_batch", df)
        self.con.execute("""
            INSERT OR REPLACE INTO llm_insights
            SELECT * FROM _insight_batch
        """)
        logger.info("Wrote %d insight rows to DuckDB", len(rows))

        # Export to Gold Parquet
        out = GOLD_DIR / "llm_insights.parquet"
        self.con.execute(
            f"COPY llm_insights TO '{out}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
        )


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run LLM enrichment for a game date")
    parser.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    parser.add_argument("--game-pk", type=int, help="Specific game PK (optional)")
    args = parser.parse_args()

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        writer = InsightWriter(con)

        if args.game_pk:
            results = writer.enrich_game(args.game_pk, args.date)
            logger.info("Generated %d insights", len(results))
        else:
            # All games on this date
            game_pks = con.execute(
                "SELECT DISTINCT game_pk FROM pitcher_game_summary WHERE game_date = ?",
                [args.date]
            ).fetchall()
            for (pk,) in game_pks:
                writer.enrich_game(pk, args.date)

        anomaly = writer.detect_anomalies(args.date)
        if anomaly:
            logger.info("Top anomaly: %s", json.dumps(anomaly, indent=2))
