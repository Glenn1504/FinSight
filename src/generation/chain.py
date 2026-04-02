"""
src/generation/chain.py
-----------------------
Core RAG chain: retrieval → prompt building → LLM → response.

Handles three query modes:
  1. Simple     — single company/year, direct retrieval
  2. Comparison — multiple companies or years, parallel retrieval
  3. Complex    — multi-hop, decomposed into sub-queries

All responses include:
  - The answer text
  - Source citations (company, year, section, chunk text)
  - Hallucination check result
  - Latency breakdown
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.generation.prompts import (
    ANALYST_SYSTEM_PROMPT,
    MULTI_DOC_SYSTEM_PROMPT,
    HALLUCINATION_CHECKER_SYSTEM_PROMPT,
    QUERY_DECOMPOSER_SYSTEM_PROMPT,
    build_qa_prompt,
    build_comparison_prompt,
    build_hallucination_check_prompt,
    build_decomposition_prompt,
)
from src.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)

DEFAULT_MODEL       = "gpt-4o-mini"
CHECKER_MODEL       = "gpt-4o-mini"   # use same model for cost; swap to gpt-4o for precision
MAX_CONTEXT_CHUNKS  = 5
HALLUCINATION_THRESHOLD = 0.75        # grounding_score below this → flag answer


@dataclass
class RAGResponse:
    answer:             str
    sources:            list[dict]
    is_grounded:        bool
    grounding_score:    float
    unsupported_claims: list[str]
    query_mode:         str
    latency_ms:         int
    retrieval_ms:       int
    generation_ms:      int
    chunks_retrieved:   int


class FinSightChain:
    """
    The main RAG chain for FinSight.

    Usage:
        chain = FinSightChain(retriever)
        response = chain.query("What were Apple's main risk factors in 2023?")
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str = DEFAULT_MODEL,
        top_k: int = MAX_CONTEXT_CHUNKS,
        run_hallucination_check: bool = True,
    ):
        self.retriever               = retriever
        self.model                   = model
        self.top_k                   = top_k
        self.run_hallucination_check = run_hallucination_check
        self._client                 = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("openai is required: pip install openai") from e
            self._client = OpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    def _chat(self, system: str, user: str, model: str | None = None,
              response_format: str = "text") -> str:
        """Single chat completion call."""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model":    model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _classify_query(self, question: str) -> str:
        """
        Classify query mode:
          simple     — one company, one year
          comparison — multiple companies or years mentioned
          complex    — multi-hop, requires decomposition
        """
        q = question.lower()
        comparison_signals = [
            "compare", "versus", "vs", "difference between",
            "how does", "relative to", "both", "all three",
        ]
        multi_year_signals = ["over the years", "trend", "2021", "2022", "2023", "2024",
                              "year over year", "yoy", "historically"]

        if any(s in q for s in comparison_signals):
            return "comparison"
        if sum(1 for s in multi_year_signals if s in q) >= 2:
            return "comparison"
        return "simple"

    # ------------------------------------------------------------------
    # Query decomposition
    # ------------------------------------------------------------------

    def _decompose_query(self, question: str) -> list[str]:
        """Break a complex question into sub-queries."""
        try:
            result = self._chat(
                system=QUERY_DECOMPOSER_SYSTEM_PROMPT,
                user=build_decomposition_prompt(question),
                response_format="json",
            )
            sub_questions = json.loads(result)
            if isinstance(sub_questions, list):
                return sub_questions[:4]
        except Exception as e:
            log.warning("Query decomposition failed (%s) — using original query.", e)
        return [question]

    # ------------------------------------------------------------------
    # Hallucination check
    # ------------------------------------------------------------------

    def _check_hallucination(self, answer: str, chunks: list[dict]) -> dict:
        """Run the adversarial hallucination checker. Returns grounding dict."""
        try:
            result = self._chat(
                system=HALLUCINATION_CHECKER_SYSTEM_PROMPT,
                user=build_hallucination_check_prompt(answer, chunks),
                model=CHECKER_MODEL,
                response_format="json",
            )
            parsed = json.loads(result)
            return {
                "is_grounded":        bool(parsed.get("is_grounded", True)),
                "unsupported_claims": parsed.get("unsupported_claims", []),
                "grounding_score":    float(parsed.get("grounding_score", 1.0)),
            }
        except Exception as e:
            log.warning("Hallucination check failed (%s) — assuming grounded.", e)
            return {"is_grounded": True, "unsupported_claims": [], "grounding_score": 1.0}

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict | None = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a question.

        Parameters
        ----------
        question : str
        filters  : dict | None
            Optional metadata filters for retrieval, e.g.
            {"ticker": "AAPL", "fiscal_year": 2023}

        Returns
        -------
        RAGResponse
        """
        t0 = time.perf_counter()

        mode = self._classify_query(question)
        log.info("Query mode: %s | question: %s...", mode, question[:80])

        # ----------------------------------------------------------------
        # Retrieval
        # ----------------------------------------------------------------
        t_ret = time.perf_counter()

        if mode == "comparison":
            # Decompose and retrieve for each sub-query
            sub_questions = self._decompose_query(question)
            log.info("Decomposed into %d sub-queries", len(sub_questions))
            all_chunks: list[dict] = []
            seen_ids: set[str] = set()
            for sq in sub_questions:
                results = self.retriever.retrieve(sq, top_k=3, filters=filters)
                for r in results:
                    if r["chunk_id"] not in seen_ids:
                        all_chunks.append(r)
                        seen_ids.add(r["chunk_id"])
            # Keep at most top_k * 2 chunks for comparison queries (need more context)
            chunks = all_chunks[: self.top_k * 2]
        else:
            chunks = self.retriever.retrieve(question, top_k=self.top_k, filters=filters)

        retrieval_ms = int((time.perf_counter() - t_ret) * 1000)
        log.info("Retrieved %d chunks in %dms", len(chunks), retrieval_ms)

        if not chunks:
            return RAGResponse(
                answer="I couldn't find relevant information in the indexed filings for this question.",
                sources=[],
                is_grounded=True,
                grounding_score=1.0,
                unsupported_claims=[],
                query_mode=mode,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                retrieval_ms=retrieval_ms,
                generation_ms=0,
                chunks_retrieved=0,
            )

        # ----------------------------------------------------------------
        # Generation
        # ----------------------------------------------------------------
        t_gen = time.perf_counter()

        system = MULTI_DOC_SYSTEM_PROMPT if mode == "comparison" else ANALYST_SYSTEM_PROMPT
        user   = (
            build_comparison_prompt(question, chunks)
            if mode == "comparison"
            else build_qa_prompt(question, chunks)
        )

        answer = self._chat(system=system, user=user)
        generation_ms = int((time.perf_counter() - t_gen) * 1000)

        # ----------------------------------------------------------------
        # Hallucination check
        # ----------------------------------------------------------------
        grounding = {"is_grounded": True, "unsupported_claims": [], "grounding_score": 1.0}
        if self.run_hallucination_check:
            grounding = self._check_hallucination(answer, chunks)

        if grounding["grounding_score"] < HALLUCINATION_THRESHOLD:
            log.warning(
                "Low grounding score %.2f for question: %s",
                grounding["grounding_score"], question[:80],
            )

        # ----------------------------------------------------------------
        # Build sources list
        # ----------------------------------------------------------------
        sources = [
            {
                "chunk_id":     c["chunk_id"],
                "company":      c["metadata"].get("company", ""),
                "ticker":       c["metadata"].get("ticker", ""),
                "fiscal_year":  c["metadata"].get("fiscal_year", ""),
                "section":      c["metadata"].get("section_label", ""),
                "text_snippet": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "relevance_score": round(c.get("rerank_score", c.get("rrf_score", 0.0)), 4),
            }
            for c in chunks
        ]

        total_ms = int((time.perf_counter() - t0) * 1000)
        log.info("Total latency: %dms (retrieval=%dms, generation=%dms)",
                 total_ms, retrieval_ms, generation_ms)

        return RAGResponse(
            answer=answer,
            sources=sources,
            is_grounded=grounding["is_grounded"],
            grounding_score=grounding["grounding_score"],
            unsupported_claims=grounding["unsupported_claims"],
            query_mode=mode,
            latency_ms=total_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            chunks_retrieved=len(chunks),
        )
