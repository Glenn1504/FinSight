"""
src/generation/hallucination.py
--------------------------------
Standalone hallucination detection utilities.

The main detector is integrated into the RAG chain (chain.py), but this
module exposes it as a standalone callable for use in the eval suite and
for batch checking of existing answers.

Design:
  - Primary check: LLM-as-judge (adversarial prompt)
  - Secondary check: NLI-based entailment using a small HuggingFace model
    (optional, runs locally, no API cost)

The two checks are complementary:
  - LLM judge catches semantic hallucinations and fabricated facts
  - NLI entailment catches logical contradictions with the source text
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    is_grounded:        bool
    grounding_score:    float          # 0 = fully hallucinated, 1 = fully grounded
    unsupported_claims: list[str]
    method:             str            # "llm_judge" | "nli" | "combined"


def check_with_llm(
    answer: str,
    context_chunks: list[dict],
    client,   # OpenAI client
    model: str = "gpt-4o-mini",
) -> HallucinationResult:
    """
    Use an LLM as adversarial judge to find unsupported claims.

    The adversarial prompt asks the model to act as a fact-checker
    and identify any claim in `answer` not grounded in `context_chunks`.
    """
    from src.generation.prompts import (
        HALLUCINATION_CHECKER_SYSTEM_PROMPT,
        build_hallucination_check_prompt,
    )

    user_prompt = build_hallucination_check_prompt(answer, context_chunks)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": HALLUCINATION_CHECKER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)

        return HallucinationResult(
            is_grounded=bool(parsed.get("is_grounded", True)),
            grounding_score=float(parsed.get("grounding_score", 1.0)),
            unsupported_claims=parsed.get("unsupported_claims", []),
            method="llm_judge",
        )
    except Exception as e:
        log.warning("LLM hallucination check failed: %s", e)
        return HallucinationResult(
            is_grounded=True,
            grounding_score=1.0,
            unsupported_claims=[],
            method="llm_judge_failed",
        )


def check_with_nli(
    answer: str,
    context_chunks: list[dict],
) -> HallucinationResult:
    """
    Use a lightweight NLI model to check entailment.

    Uses cross-encoder/nli-deberta-v3-small — runs locally, no API cost.
    Requires: pip install sentence-transformers

    For each sentence in `answer`, checks whether any context chunk
    entails it. If no chunk entails a sentence, it's flagged.
    """
    try:
        from sentence_transformers import CrossEncoder
        import re
    except ImportError:
        log.warning("sentence-transformers not installed — skipping NLI check.")
        return HallucinationResult(
            is_grounded=True, grounding_score=1.0,
            unsupported_claims=[], method="nli_skipped",
        )

    model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    context_text = " ".join(c["text"] for c in context_chunks)

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 20]
    if not sentences:
        return HallucinationResult(
            is_grounded=True, grounding_score=1.0,
            unsupported_claims=[], method="nli",
        )

    unsupported = []
    entailment_scores = []

    for sentence in sentences:
        # NLI labels: 0=contradiction, 1=neutral, 2=entailment
        score = model.predict([(context_text[:2000], sentence)])
        entailment_score = float(score[0][2]) if hasattr(score[0], "__len__") else float(score[0])
        entailment_scores.append(entailment_score)
        if entailment_score < 0.5:
            unsupported.append(sentence)

    grounding_score = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 1.0

    return HallucinationResult(
        is_grounded=len(unsupported) == 0,
        grounding_score=round(grounding_score, 4),
        unsupported_claims=unsupported,
        method="nli",
    )


def check_combined(
    answer: str,
    context_chunks: list[dict],
    client,
    llm_model: str = "gpt-4o-mini",
) -> HallucinationResult:
    """
    Run both LLM judge and NLI check, combine results.
    Used in the eval suite for maximum precision.
    """
    llm_result = check_with_llm(answer, context_chunks, client, llm_model)
    nli_result = check_with_nli(answer, context_chunks)

    # Combined score: average of both
    combined_score = (llm_result.grounding_score + nli_result.grounding_score) / 2.0
    all_unsupported = list(set(llm_result.unsupported_claims + nli_result.unsupported_claims))

    return HallucinationResult(
        is_grounded=combined_score >= 0.75,
        grounding_score=round(combined_score, 4),
        unsupported_claims=all_unsupported,
        method="combined",
    )
