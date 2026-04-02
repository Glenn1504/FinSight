"""
scripts/ingest.py
-----------------
End-to-end ingestion pipeline:
  1. Download 10-K filings from SEC EDGAR
  2. Parse HTML → structured sections
  3. Chunk with section-aware strategy
  4. Embed and upsert to ChromaDB
  5. Build BM25 index

Usage:
    python scripts/ingest.py --tickers AAPL MSFT GOOGL AMZN NVDA --years 2022 2023 2024
    python scripts/ingest.py --tickers NVDA --years 2023  # single company
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.edgar import EdgarClient
from src.ingestion.parser import parse_filing
from src.ingestion.chunker import chunk_filing
from src.retrieval.vectorstore import VectorStore
from src.retrieval.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def ingest(tickers: list[str], years: list[int]) -> None:
    client = EdgarClient()
    vs     = VectorStore()
    bm25   = BM25Retriever()

    all_chunks = []

    for ticker in tickers:
        log.info("=" * 50)
        log.info("Processing %s ...", ticker)

        try:
            filings = client.get_10k_filings(ticker, years=years)
        except Exception as e:
            log.error("Failed to fetch filings for %s: %s", ticker, e)
            continue

        if not filings:
            log.warning("No filings found for %s in years %s", ticker, years)
            continue

        for filing in filings:
            try:
                # Download
                local_path = client.download_filing(filing, output_dir="data/raw/")

                # Parse
                parsed = parse_filing(
                    local_path,
                    ticker=filing.ticker,
                    company=filing.company,
                    fiscal_year=filing.fiscal_year,
                )

                # Chunk
                chunks = chunk_filing(parsed)
                all_chunks.extend(chunks)

                log.info(
                    "✓ %s %d: %d chunks",
                    filing.ticker, filing.fiscal_year, len(chunks),
                )

            except Exception as e:
                log.error("Failed to process %s %d: %s", ticker, filing.fiscal_year, e)
                continue

    if not all_chunks:
        log.error("No chunks produced — check your tickers and years.")
        sys.exit(1)

    log.info("=" * 50)
    log.info("Total chunks to index: %d", len(all_chunks))

    # Upsert to ChromaDB
    log.info("Upserting to ChromaDB ...")
    vs.upsert(all_chunks)

    # Build BM25 index
    log.info("Building BM25 index ...")
    bm25_docs = [
        {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.to_metadata()}
        for c in all_chunks
    ]
    bm25.build(bm25_docs)

    log.info("=" * 50)
    log.info("Ingestion complete.")
    log.info("  ChromaDB: %d chunks", vs.count())
    log.info("  BM25:     %d chunks", len(bm25_docs))
    log.info("")
    log.info("Run the app:  streamlit run src/app/chat.py")
    log.info("Run evals:    python -m src.evaluation.run_evals")


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC 10-K filings into FinSight.")
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        help="Ticker symbols to ingest",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023, 2024],
        help="Fiscal years to ingest",
    )
    args = parser.parse_args()

    log.info("Ingesting tickers=%s years=%s", args.tickers, args.years)
    ingest(args.tickers, args.years)


if __name__ == "__main__":
    main()
