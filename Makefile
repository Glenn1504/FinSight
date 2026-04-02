.PHONY: install ingest app evals test lint clean

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	python3 -m pip install --upgrade pip
	pip3 install -r requirements.txt

# ── Ingestion ──────────────────────────────────────────────────────────────
ingest:
	python3 scripts/ingest.py \
		--tickers AAPL MSFT GOOGL AMZN NVDA \
		--years 2022 2023 2024

ingest-quick:
	python3 scripts/ingest.py \
		--tickers AAPL MSFT NVDA \
		--years 2023

# ── App ────────────────────────────────────────────────────────────────────
app:
	streamlit run src/app/chat.py

# ── Evaluation ─────────────────────────────────────────────────────────────
evals:
	python3 -m src.evaluation.run_evals \
		--dataset data/golden/golden_qa.json \
		--output  reports/

evals-sample:
	python3 -m src.evaluation.run_evals \
		--dataset data/golden/golden_qa.json \
		--output  reports/ \
		--sample  10

evals-dry-run:
	python3 -m src.evaluation.run_evals \
		--dataset data/golden/golden_qa.json \
		--output  reports/ \
		--skip-chain

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short -k "not integration"

# ── Lint ───────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

lint-fix:
	ruff check --fix src/ tests/ scripts/

# ── Full run (dev) ──────────────────────────────────────────────────────────
run-all: install ingest-quick app

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage coverage.xml htmlcov/
