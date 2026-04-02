"""
src/evaluation/run_evals.py
----------------------------
RAGAS evaluation runner for FinSight.

Runs the golden Q&A dataset through the RAG pipeline and computes:
  - Faithfulness        (are answer claims supported by context?)
  - Answer Relevance    (does the answer address the question?)
  - Context Precision   (what fraction of retrieved chunks were useful?)
  - Context Recall      (did we retrieve all relevant information?)
  - Hallucination Rate  (fraction of answers with unsupported claims)

Results are saved as JSON + an HTML report.

CI GATE: Fails (exit 1) if faithfulness < 0.85 or hallucination_rate > 0.08

Usage:
    python -m src.evaluation.run_evals \
        --dataset data/golden/golden_qa.json \
        --output  reports/ \
        --sample  20          # run on a subset for fast CI checks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# CI thresholds — fail the build if these are breached
FAITHFULNESS_THRESHOLD    = 0.85
HALLUCINATION_THRESHOLD   = 0.08


def _load_golden_dataset(path: str, sample: int | None = None) -> list[dict]:
    with open(path) as f:
        dataset = json.load(f)
    if sample:
        import random
        random.seed(42)
        dataset = random.sample(dataset, min(sample, len(dataset)))
    log.info("Loaded %d golden Q&A pairs from %s", len(dataset), path)
    return dataset


def _build_chain():
    """Initialise the full RAG chain from environment config."""
    from src.retrieval.vectorstore import VectorStore
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.hybrid import HybridRetriever
    from src.generation.chain import FinSightChain

    vs      = VectorStore()
    bm25    = BM25Retriever()
    bm25.load()
    hybrid  = HybridRetriever(vs, bm25)
    chain   = FinSightChain(hybrid)
    return chain


def _compute_ragas_metrics(eval_rows: list[dict]) -> dict:
    """
    Compute RAGAS metrics using the ragas library.
    Falls back to simple heuristics if ragas isn't installed.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        hf_dataset = Dataset.from_list([
            {
                "question":  row["question"],
                "answer":    row["answer"],
                "contexts":  row["contexts"],
                "ground_truth": row.get("ground_truth", ""),
            }
            for row in eval_rows
            if row.get("answer") and row.get("contexts")
        ])

        result = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return {
            "faithfulness":      round(float(result["faithfulness"]), 4),
            "answer_relevance":  round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall":    round(float(result["context_recall"]), 4),
            "method":            "ragas",
        }

    except ImportError:
        log.warning("ragas not installed — computing simplified metrics.")
        return _compute_simple_metrics(eval_rows)


def _compute_simple_metrics(eval_rows: list[dict]) -> dict:
    """
    Simplified metrics when ragas isn't available:
      - faithfulness proxy: average grounding_score from hallucination checker
      - answer_relevance proxy: fraction of non-empty answers
      - context_precision proxy: avg(chunks_retrieved > 0)
    """
    grounding_scores = [r.get("grounding_score", 1.0) for r in eval_rows if r.get("answer")]
    non_empty = sum(1 for r in eval_rows if r.get("answer", "").strip())

    return {
        "faithfulness":      round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.0,
        "answer_relevance":  round(non_empty / len(eval_rows), 4),
        "context_precision": None,   # requires ground truth chunk labels
        "context_recall":    None,
        "method":            "simplified",
    }


def run_evals(
    dataset_path: str,
    output_dir: str,
    sample: int | None = None,
    skip_chain: bool = False,
) -> dict:
    """
    Main eval runner. Returns the metrics dict.

    Parameters
    ----------
    dataset_path : str
    output_dir   : str
    sample       : int | None   — evaluate on a random subset
    skip_chain   : bool         — if True, skip actual RAG calls (for testing)
    """
    dataset   = _load_golden_dataset(dataset_path, sample)
    out       = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not skip_chain:
        chain = _build_chain()

    eval_rows = []
    n_hallucinated = 0
    errors = 0

    for i, item in enumerate(dataset):
        log.info("[%d/%d] %s", i + 1, len(dataset), item["question"][:80])

        if skip_chain:
            # Mock response for testing
            eval_rows.append({
                **item,
                "answer":         "Mock answer for testing.",
                "contexts":       ["Mock context."],
                "grounding_score": 1.0,
                "is_grounded":    True,
                "latency_ms":     0,
            })
            continue

        try:
            filters = {}
            if "ticker" in item:
                filters["ticker"] = item["ticker"]
            if "fiscal_year" in item:
                filters["fiscal_year"] = item["fiscal_year"]

            response = chain.query(item["question"], filters=filters or None)

            if not response.is_grounded:
                n_hallucinated += 1

            eval_rows.append({
                **item,
                "answer":         response.answer,
                "contexts":       [s["text_snippet"] for s in response.sources],
                "grounding_score": response.grounding_score,
                "is_grounded":    response.is_grounded,
                "latency_ms":     response.latency_ms,
                "chunks_retrieved": response.chunks_retrieved,
            })

        except Exception as e:
            log.error("Error on item %s: %s", item["id"], e)
            errors += 1
            eval_rows.append({**item, "answer": "", "contexts": [], "error": str(e)})

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    ragas_metrics = _compute_ragas_metrics(eval_rows)

    hallucination_rate = n_hallucinated / len(dataset) if dataset else 0.0
    avg_latency = (
        sum(r.get("latency_ms", 0) for r in eval_rows) / len(eval_rows)
        if eval_rows else 0
    )

    metrics = {
        "timestamp":          datetime.utcnow().isoformat(),
        "n_questions":        len(dataset),
        "n_errors":           errors,
        "hallucination_rate": round(hallucination_rate, 4),
        "avg_latency_ms":     round(avg_latency),
        **ragas_metrics,
    }

    log.info("=" * 55)
    log.info("EVAL RESULTS")
    log.info("=" * 55)
    for k, v in metrics.items():
        log.info("  %-30s %s", k, v)
    log.info("=" * 55)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    metrics_path = out / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    detail_path = out / "eval_detail.json"
    with open(detail_path, "w") as f:
        json.dump(eval_rows, f, indent=2, default=str)
    log.info("Detail saved → %s", detail_path)

    # ----------------------------------------------------------------
    # CI gate
    # ----------------------------------------------------------------
    faithfulness = metrics.get("faithfulness", 1.0)
    if faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD:
        log.error(
            "CI GATE FAILED: faithfulness=%.3f < threshold=%.2f",
            faithfulness, FAITHFULNESS_THRESHOLD,
        )
        sys.exit(1)

    if hallucination_rate > HALLUCINATION_THRESHOLD:
        log.error(
            "CI GATE FAILED: hallucination_rate=%.3f > threshold=%.2f",
            hallucination_rate, HALLUCINATION_THRESHOLD,
        )
        sys.exit(1)

    log.info("CI gates passed.")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/golden/golden_qa.json")
    parser.add_argument("--output",  default="reports/")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Evaluate on a random subset of N questions")
    parser.add_argument("--skip-chain", action="store_true",
                        help="Skip RAG calls (for testing the eval framework itself)")
    args = parser.parse_args()

    run_evals(args.dataset, args.output, args.sample, args.skip_chain)


if __name__ == "__main__":
    main()
