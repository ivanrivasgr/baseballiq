# BaseballIQ — Production Pipeline Makefile
# ==========================================
# Usage:
#   make ingest DATE_START=2024-07-01 DATE_END=2024-07-07
#   make clean
#   make gold
#   make enrich DATE=2024-07-07
#   make train
#   make report PLAYER_ID=605483 DATE=2024-07-14
#   make dashboard
#   make test

.PHONY: all ingest clean gold enrich train report dashboard test lint help

# ── Config ──────────────────────────────────────────────────────────────────────
PYTHON       := python -m
DATE_START   ?= 2024-04-01
DATE_END     ?= 2024-04-07
DATE         ?= 2024-07-01
PLAYER_ID    ?= 605483
MODEL_PATH   ?= models/artifacts/pitcher_effectiveness_v1.pkl

# ── Default ──────────────────────────────────────────────────────────────────────
all: help

# ── Pipeline Steps ──────────────────────────────────────────────────────────────

## Ingest raw Statcast data to Bronze layer
ingest:
	@echo "🟤 Ingesting Statcast: $(DATE_START) → $(DATE_END)"
	$(PYTHON) pipeline.ingestion.statcast_ingestion \
		--start $(DATE_START) \
		--end   $(DATE_END)

## Clean and load Bronze → Silver (DuckDB)
clean:
	@echo "🥈 Cleaning Bronze → Silver"
	$(PYTHON) pipeline.silver.cleaning
	$(PYTHON) pipeline.silver.feature_engineering

## Build Gold analytical datasets from Silver
gold:
	@echo "🥇 Building Gold layer"
	$(PYTHON) pipeline.gold.aggregations

## LLM enrichment for a specific date
enrich:
	@echo "🤖 Running LLM enrichment for $(DATE)"
	$(PYTHON) enrichment.insight_writer --date $(DATE)

## Train pitcher effectiveness model
train:
	@echo "📈 Training XGBoost pitcher effectiveness model"
	$(PYTHON) models.train --output $(MODEL_PATH)

## Run full pipeline (ingest → clean → gold → enrich → train)
pipeline: ingest clean gold enrich train
	@echo "✅ Full pipeline complete"

## Generate scouting report for a player
report:
	@echo "📋 Generating scouting report: pitcher=$(PLAYER_ID), date=$(DATE)"
	$(PYTHON) reports.scouting_report \
		--pitcher-id $(PLAYER_ID) \
		--game-date  $(DATE) \
		--json-out   reports/output/pitcher_$(PLAYER_ID)_$(DATE).json

## Launch Streamlit dashboard
dashboard:
	@echo "📊 Starting Streamlit dashboard"
	streamlit run dashboard/app.py

## Run test suite
test:
	@echo "🧪 Running tests"
	pytest tests/ -v --tb=short

## Lint and type check
lint:
	ruff check pipeline/ enrichment/ models/ reports/
	mypy pipeline/ --ignore-missing-imports

## Set up development environment
setup:
	uv sync
	cp -n .env.example .env || true
	mkdir -p data/bronze data/silver data/gold models/artifacts reports/output
	@echo "✅ Environment ready. Add your ANTHROPIC_API_KEY to .env"

## Run sample pipeline on included demo data (no API key needed)
demo:
	@echo "🎮 Running demo pipeline on sample data"
	DEMO_MODE=1 $(PYTHON) pipeline.orchestrator \
		--start 2024-07-01 --end 2024-07-01
	DEMO_MODE=1 $(PYTHON) reports.scouting_report \
		--pitcher-id 605483 --game-date 2024-07-01

help:
	@echo ""
	@echo "BaseballIQ — Sports Analytics Pipeline"
	@echo "======================================="
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
	@echo "Variables:"
	@echo "  DATE_START   Start date for ingestion (default: $(DATE_START))"
	@echo "  DATE_END     End date for ingestion   (default: $(DATE_END))"
	@echo "  DATE         Single game date         (default: $(DATE))"
	@echo "  PLAYER_ID    MLB player ID            (default: $(PLAYER_ID))"
	@echo ""
