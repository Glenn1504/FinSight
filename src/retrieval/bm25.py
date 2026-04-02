"""
src/retrieval/bm25.py
---------------------
BM25 sparse retrieval over the chunk corpus.

Why BM25 alongside dense search?
Financial filings contain exact figures ("revenue grew 12.3%"), product
names, and regulatory terms that dense embeddings handle poorly — they
retrieve semantically related but numerically wrong chunks. BM25 catches
exact keyword matches that semantic search misses.

We persist the BM25 index as a pickle alongside the vectorstore so it
doesn't need to be rebuilt on every query.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = "data/vectorstore/bm25_index.pkl"


class BM25Retriever:
    """
    Wrapper around rank_bm25.BM25Okapi for keyword retrieval over chunks.

    Stores the full chunk texts in memory alongside the index so we can
    return the same dict format as VectorStore.search().
    """

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self._index     = None
        self._chunks: list[dict] = []     # [{chunk_id, text, metadata}]

    # ------------------------------------------------------------------
    # Build / persist
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts.

        Parameters
        ----------
        chunks : list of dicts with keys: chunk_id, text, metadata
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from e

        self._chunks = chunks
        tokenized = [self._tokenize(c["text"]) for c in chunks]
        self._index = BM25Okapi(tokenized)

        self._save()
        log.info("BM25 index built: %d chunks", len(chunks))

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        text = text.lower()
        # Keep numbers and dollar amounts intact
        tokens = re.findall(r"\$?[\d,\.]+%?|\b\w+\b", text)
        return tokens

    def _save(self) -> None:
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "chunks": self._chunks}, f)
        log.info("BM25 index saved → %s", path)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        path = Path(self.index_path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index  = data["index"]
        self._chunks = data["chunks"]
        log.info("BM25 index loaded: %d chunks", len(self._chunks))
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        BM25 keyword search.

        Returns list of dicts: {chunk_id, text, metadata, score}
        Scores are normalised to [0, 1].
        """
        if self._index is None:
            if not self.load():
                raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        # Normalise
        max_score = scores.max() if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        # Apply metadata filters
        candidates = []
        for i, chunk in enumerate(self._chunks):
            if filters and not self._matches_filter(chunk["metadata"], filters):
                continue
            candidates.append((i, float(norm_scores[i])))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]

        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "text":     self._chunks[i]["text"],
                "metadata": self._chunks[i]["metadata"],
                "score":    score,
            }
            for i, score in top
        ]

    def _matches_filter(self, metadata: dict, filters: dict) -> bool:
        """Simple equality filter matching (mirrors ChromaDB where-clause)."""
        for key, val in filters.items():
            if key.startswith("$"):
                continue  # skip ChromaDB operators
            meta_val = metadata.get(key)
            if isinstance(val, dict):
                # Handle $in operator
                if "$in" in val and meta_val not in val["$in"]:
                    return False
            elif meta_val != val:
                return False
        return True
