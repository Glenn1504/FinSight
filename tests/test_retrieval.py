"""
tests/test_retrieval.py
-----------------------
Tests for BM25 retrieval and RRF fusion.
"""

from __future__ import annotations

import pytest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import _reciprocal_rank_fusion


def _make_chunks(n: int = 20) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i:03d}",
            "text": f"This document discusses {['revenue growth', 'risk factors', 'operating expenses', 'cloud services', 'AI investments'][i % 5]} for fiscal year {2022 + (i % 3)}.",
            "metadata": {
                "ticker":      ["AAPL", "MSFT", "NVDA"][i % 3],
                "fiscal_year": 2022 + (i % 3),
                "section_key": ["business", "risk_factors", "mda"][i % 3],
            },
        }
        for i in range(n)
    ]


class TestBM25Retriever:
    def test_build_and_search(self, tmp_path):
        chunks = _make_chunks(20)
        bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        bm25.build(chunks)

        results = bm25.search("revenue growth", top_k=5)
        assert len(results) <= 5
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)

    def test_top_result_is_relevant(self, tmp_path):
        chunks = _make_chunks(20)
        bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        bm25.build(chunks)

        results = bm25.search("revenue growth", top_k=5)
        # The top result should mention "revenue growth"
        assert "revenue" in results[0]["text"].lower()

    def test_filter_by_ticker(self, tmp_path):
        chunks = _make_chunks(20)
        bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        bm25.build(chunks)

        results = bm25.search("risk factors", top_k=10, filters={"ticker": "AAPL"})
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"

    def test_save_and_load_roundtrip(self, tmp_path):
        chunks = _make_chunks(10)
        bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        bm25.build(chunks)

        bm25_loaded = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        assert bm25_loaded.load()
        results = bm25_loaded.search("cloud services", top_k=3)
        assert len(results) > 0

    def test_scores_normalised_between_0_and_1(self, tmp_path):
        chunks = _make_chunks(20)
        bm25 = BM25Retriever(index_path=str(tmp_path / "bm25.pkl"))
        bm25.build(chunks)

        results = bm25.search("AI investments", top_k=10)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0


class TestRRF:
    def _make_ranked_list(self, ids: list[str]) -> list[dict]:
        return [
            {"chunk_id": cid, "text": f"text {cid}", "metadata": {}}
            for cid in ids
        ]

    def test_combines_two_lists(self):
        list1 = self._make_ranked_list(["a", "b", "c", "d"])
        list2 = self._make_ranked_list(["b", "a", "e", "f"])
        merged = _reciprocal_rank_fusion([list1, list2])
        ids = [m["chunk_id"] for m in merged]
        # b and a should score highest (appear in both lists near the top)
        assert ids[0] in {"a", "b"}
        assert ids[1] in {"a", "b"}

    def test_deduplicates_chunks(self):
        list1 = self._make_ranked_list(["a", "b", "c"])
        list2 = self._make_ranked_list(["a", "b", "d"])
        merged = _reciprocal_rank_fusion([list1, list2])
        ids = [m["chunk_id"] for m in merged]
        assert len(ids) == len(set(ids))

    def test_rrf_score_attached(self):
        list1 = self._make_ranked_list(["a", "b"])
        list2 = self._make_ranked_list(["b", "a"])
        merged = _reciprocal_rank_fusion([list1, list2])
        for item in merged:
            assert "rrf_score" in item
            assert item["rrf_score"] > 0

    def test_single_list_passthrough(self):
        list1 = self._make_ranked_list(["x", "y", "z"])
        merged = _reciprocal_rank_fusion([list1])
        assert [m["chunk_id"] for m in merged] == ["x", "y", "z"]
