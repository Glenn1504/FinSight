"""
tests/test_hallucination.py
----------------------------
Tests for the hallucination detection module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json
import pytest

from src.generation.hallucination import HallucinationResult


def _make_chunks(texts: list[str]) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "text": t, "metadata": {"ticker": "AAPL"}}
        for i, t in enumerate(texts)
    ]


class TestHallucinationResult:
    def test_dataclass_fields(self):
        result = HallucinationResult(
            is_grounded=True,
            grounding_score=0.95,
            unsupported_claims=[],
            method="llm_judge",
        )
        assert result.is_grounded
        assert result.grounding_score == 0.95
        assert result.method == "llm_judge"

    def test_not_grounded_when_low_score(self):
        result = HallucinationResult(
            is_grounded=False,
            grounding_score=0.3,
            unsupported_claims=["Claim X not in context"],
            method="llm_judge",
        )
        assert not result.is_grounded
        assert len(result.unsupported_claims) == 1


class TestCheckWithLLM:
    def _mock_openai_response(self, payload: dict):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(payload)
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        return mock_client

    def test_grounded_response(self):
        from src.generation.hallucination import check_with_llm

        client = self._mock_openai_response({
            "is_grounded": True,
            "unsupported_claims": [],
            "grounding_score": 0.95,
        })
        chunks = _make_chunks(["Apple revenue grew 8% in FY2023."])
        result = check_with_llm(
            answer="Apple's revenue grew 8% in FY2023.",
            context_chunks=chunks,
            client=client,
        )
        assert result.is_grounded
        assert result.grounding_score == 0.95
        assert result.unsupported_claims == []

    def test_hallucinated_response(self):
        from src.generation.hallucination import check_with_llm

        client = self._mock_openai_response({
            "is_grounded": False,
            "unsupported_claims": ["Apple acquired OpenAI in 2023"],
            "grounding_score": 0.2,
        })
        chunks = _make_chunks(["Apple reported strong iPhone sales."])
        result = check_with_llm(
            answer="Apple acquired OpenAI in 2023 for $10 billion.",
            context_chunks=chunks,
            client=client,
        )
        assert not result.is_grounded
        assert result.grounding_score == 0.2
        assert len(result.unsupported_claims) == 1

    def test_graceful_failure_on_api_error(self):
        from src.generation.hallucination import check_with_llm

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = check_with_llm(
            answer="Some answer",
            context_chunks=_make_chunks(["Some context"]),
            client=mock_client,
        )
        # Should default to grounded on failure
        assert result.is_grounded
        assert result.method == "llm_judge_failed"
