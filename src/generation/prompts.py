"""
src/generation/prompts.py
--------------------------
Prompt templates for FinSight.

Separate templates for:
  - Single-document Q&A
  - Multi-document comparison
  - Query decomposition (for complex multi-hop questions)
  - Hallucination detection (adversarial checker)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """You are FinSight, an expert financial analyst assistant
specializing in SEC 10-K filings and earnings call transcripts.

Your job is to answer questions about public companies using ONLY the provided
context excerpts from their official filings. You must:

1. Base every claim on the provided context — never use outside knowledge
2. Cite your sources by referencing the company name, fiscal year, and section
   (e.g. "According to Apple's 2023 10-K Risk Factors section...")
3. If the context doesn't contain enough information to answer, say so clearly
4. When discussing numbers, quote them exactly as they appear in the filings
5. Flag any forward-looking statements as such

Do not speculate, hallucinate, or use knowledge outside the provided context."""


MULTI_DOC_SYSTEM_PROMPT = """You are FinSight, an expert financial analyst assistant
specializing in comparative analysis of SEC 10-K filings.

You are comparing information across multiple companies and/or fiscal years.
Structure your answer clearly, using company names and fiscal years as headers
where appropriate. Cite every claim with its source filing."""


HALLUCINATION_CHECKER_SYSTEM_PROMPT = """You are a rigorous fact-checker for a
financial AI system. Your job is to identify any claims in an AI-generated answer
that are NOT supported by the provided source context.

Be strict: if a specific number, date, name, or factual claim cannot be found
verbatim or clearly implied in the context, flag it as unsupported.

Output a JSON object with:
{
  "is_grounded": true/false,
  "unsupported_claims": ["claim 1", "claim 2", ...],
  "grounding_score": 0.0-1.0
}"""


QUERY_DECOMPOSER_SYSTEM_PROMPT = """You are a query decomposition assistant for a
financial document retrieval system.

Given a complex question about multiple companies or time periods, break it into
simpler sub-questions that can each be answered with a single document retrieval.

Output a JSON array of sub-questions:
["sub-question 1", "sub-question 2", ...]

Keep sub-questions focused and specific. Maximum 4 sub-questions."""


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

def build_qa_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the user prompt for single/multi-doc Q&A."""
    context_text = _format_context(context_chunks)
    return f"""Answer the following question using ONLY the context provided below.

QUESTION: {question}

CONTEXT:
{context_text}

Provide a clear, structured answer with citations to specific filings."""


def build_comparison_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build prompt for multi-company or multi-year comparisons."""
    context_text = _format_context(context_chunks)
    return f"""Compare and contrast the following based on the provided filing excerpts.

QUESTION: {question}

FILING EXCERPTS:
{context_text}

Structure your comparison clearly. Use specific numbers and cite sources."""


def build_hallucination_check_prompt(answer: str, context_chunks: list[dict]) -> str:
    """Build the adversarial prompt for hallucination detection."""
    context_text = _format_context(context_chunks)
    return f"""Check whether every factual claim in the ANSWER below is supported
by the CONTEXT. Be strict about numbers, percentages, dates, and named entities.

ANSWER:
{answer}

CONTEXT:
{context_text}

Return a JSON object as specified in your system prompt."""


def build_decomposition_prompt(question: str) -> str:
    """Build prompt for query decomposition."""
    return f"""Decompose this complex financial question into simpler sub-questions:

QUESTION: {question}

Return a JSON array of sub-questions."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        header = (
            f"[{i}] {meta.get('company', 'Unknown')} "
            f"({meta.get('ticker', '?')}) — "
            f"FY{meta.get('fiscal_year', '?')} — "
            f"{meta.get('section_label', 'Unknown Section')}"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
