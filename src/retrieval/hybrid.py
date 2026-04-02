"""
src/retrieval/hybrid.py
-----------------------
Hybrid retrieval: dense + sparse → Reciprocal Rank Fusion → re-ranker.

Pipeline:
  1. Dense search (ChromaDB)     → top-20 chunks
  2. Sparse search (BM25)        → top-20 chunks
  3. Reciprocal Rank Fusion      → merged, deduplicated top-20
  4. Cross-encoder re-ranking    → final top-k

Why RRF instead of a weighted sum?
RRF (Raudys & Duda, 1972; popularised for IR by Cormack et al. 2009)
combines ranked lists without needing to tune a weighting hyperparameter.
The formula is: RRF(d) = Σ 1 / (k + rank(d)) where k=60 is standard.

Why a re-ranker after fusion?
Embedding models and BM25 optimise for broad recall. A cross-encoder
sees the (query, chunk) pair jointly, giving much better precision.
We only re-rank the top-20, keeping the latency hit to ~200ms.
"""

from __future__ import annotations

import logging
from typing import Any

from src.retrieval.vectorstore import VectorStore
from src.retrieval.bm25 import BM25Retriever

log = logging.getLogger(__name__)

RRF_K = 60   # standard constant from the original RRF paper


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked result lists (each item has chunk_id, text, metadata)
    k            : RRF constant (default 60)

    Returns
    -------
    Merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item["chunk_id"]
            scores[cid]    = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_map[cid] = item

    merged = sorted(chunk_map.values(), key=lambda x: scores[x["chunk_id"]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item["chunk_id"]]

    return merged


def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Re-rank chunks using a cross-encoder model.
    Falls back to RRF ordering if sentence-transformers not installed.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        log.warning("sentence-transformers not installed — skipping re-ranking.")
        return chunks[:top_k]

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked[:top_k]


class HybridRetriever:
    """
    Combines ChromaDB (dense) + BM25 (sparse) with RRF fusion and
    optional cross-encoder re-ranking.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        bm25: BM25Retriever,
        use_reranker: bool = True,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ):
        self.vectorstore   = vectorstore
        self.bm25          = bm25
        self.use_reranker  = use_reranker
        self.dense_top_k   = dense_top_k
        self.sparse_top_k  = sparse_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Parameters
        ----------
        query   : str
        top_k   : int   — final number of chunks to return
        filters : dict  — metadata filters applied to both retrievers

        Returns
        -------
        list of chunk dicts sorted by relevance, length = top_k
        """
        log.debug("Hybrid retrieval: query='%s...' filters=%s top_k=%d",
                  query[:60], filters, top_k)

        # 1. Dense retrieval
        dense_results = self.vectorstore.search(
            query, top_k=self.dense_top_k, filters=filters
        )

        # 2. Sparse retrieval
        sparse_results = self.bm25.search(
            query, top_k=self.sparse_top_k, filters=filters
        )

        # 3. RRF fusion
        fused = _reciprocal_rank_fusion([dense_results, sparse_results])
        log.debug("After RRF fusion: %d unique chunks", len(fused))

        # 4. Re-ranking (takes top 20 → returns top_k)
        if self.use_reranker and len(fused) > top_k:
            final = _rerank(query, fused[:20], top_k=top_k)
        else:
            final = fused[:top_k]

        log.debug("Returning %d chunks after reranking", len(final))
        return final
