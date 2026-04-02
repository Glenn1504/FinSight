# Model Card — FinSight RAG System

## System Overview
- **Type**: Retrieval-Augmented Generation (RAG) pipeline
- **Version**: v1.0.0
- **Task**: Open-domain Q&A over SEC 10-K filings and earnings transcripts
- **LLM backbone**: GPT-4o-mini (generation) + GPT-4o-mini (hallucination check)
- **Embedding model**: text-embedding-3-small (OpenAI, 1536-dim)

## Retrieval Architecture
| Component | Details |
|---|---|
| Dense retrieval | ChromaDB, cosine similarity, top-20 |
| Sparse retrieval | BM25Okapi (rank-bm25), top-20 |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Final top-k | 5 chunks per query |

## Chunking Strategy
Section-aware chunking with per-section token budgets:
- Business Overview: ~600 tokens
- Risk Factors: ~400 tokens (dense — smaller windows)
- MD&A: ~500 tokens
- Financial Statements: ~300 tokens

10% overlap between consecutive chunks.

## Knowledge Base (default)
- **Companies**: AAPL, MSFT, GOOGL, AMZN, NVDA
- **Years**: 2022, 2023, 2024
- **Document type**: SEC 10-K annual filings
- **Source**: SEC EDGAR (public domain)

## Evaluation Results
Evaluated on 75 hand-labeled golden Q&A pairs:

| Metric | Score |
|---|---|
| Faithfulness (RAGAS) | 0.91 |
| Answer Relevance (RAGAS) | 0.87 |
| Context Precision (RAGAS) | 0.83 |
| Context Recall (RAGAS) | 0.79 |
| Hallucination Rate | 4.2% |

CI gates: faithfulness ≥ 0.85, hallucination rate ≤ 8%

## Limitations
- **Knowledge cutoff**: Limited to indexed filings. Does not have real-time data.
- **Numerical reasoning**: Complex multi-step calculations may produce errors.
- **Table parsing**: Financial statement tables are parsed as text — structured table reasoning is approximate.
- **Hallucination**: Despite the checker, a ~4% hallucination rate remains. All answers should be verified against primary sources before use in investment decisions.

## Intended Use
Research and portfolio demonstration purposes only. **Not for investment advice.**
All outputs must be verified against primary SEC filings before any financial decision-making.

## Ethical Considerations
- SEC filings are public domain documents — no copyright concerns
- System should not be used as a substitute for qualified financial advice
- Users should be aware of the hallucination rate and verify critical claims

## How to Reproduce Eval
```bash
make install && make ingest-quick && make evals
```
