"""
src/retrieval/vectorstore.py
-----------------------------
ChromaDB wrapper for dense retrieval.

Handles:
  - Collection creation with cosine similarity
  - Batch upsert of Chunks with metadata
  - Filtered similarity search (by ticker, year, section)
  - Persistence to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.chunker import Chunk

log = logging.getLogger(__name__)

DEFAULT_COLLECTION   = "finsight_10k"
DEFAULT_PERSIST_DIR  = "data/vectorstore"
EMBED_MODEL          = "text-embedding-3-small"   # OpenAI, 1536-dim, cheap
BATCH_SIZE           = 100


class VectorStore:
    """
    Thin wrapper around ChromaDB for 10-K chunk storage and retrieval.

    Uses OpenAI text-embedding-3-small for embeddings — good quality,
    cheap (~$0.02 per 1M tokens), and fast.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = EMBED_MODEL,
    ):
        self.persist_dir      = persist_dir
        self.collection_name  = collection_name
        self.embedding_model  = embedding_model
        self._client          = None
        self._collection      = None
        self._embed_fn        = None

    def _init(self):
        """Lazy init — don't import chromadb at module load time."""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise ImportError("chromadb is required: pip install chromadb") from e

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)

        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name=self.embedding_model,
            api_key=None,  # reads OPENAI_API_KEY from env
        )

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "VectorStore ready: %d chunks in collection '%s'",
            self._collection.count(), self.collection_name,
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into ChromaDB in batches."""
        self._init()

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            log.info(
                "Upserted batch %d/%d (%d chunks)",
                i // BATCH_SIZE + 1,
                (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
                len(batch),
            )

        log.info("VectorStore now contains %d chunks.", self._collection.count())

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search.

        Parameters
        ----------
        query   : str
        top_k   : int
        filters : dict | None
            ChromaDB where-clause filters, e.g.:
            {"ticker": "AAPL"}
            {"ticker": {"$in": ["AAPL", "MSFT"]}, "fiscal_year": 2023}

        Returns
        -------
        list of dicts with keys: chunk_id, text, metadata, score
        """
        self._init()

        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": top_k}
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "chunk_id": meta["chunk_id"],
                "text":     doc,
                "metadata": meta,
                "score":    1.0 - dist,   # cosine distance → similarity
            })

        return output

    def count(self) -> int:
        self._init()
        return self._collection.count()

    def delete_collection(self) -> None:
        self._init()
        self._client.delete_collection(self.collection_name)
        log.warning("Deleted collection: %s", self.collection_name)
