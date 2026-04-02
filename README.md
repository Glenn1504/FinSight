# FinSight 📊

> **Analyst-grade RAG system for SEC 10-K filings and earnings call transcripts** — with a rigorous evaluation suite measuring faithfulness, hallucination rate, and retrieval precision across multi-document financial reasoning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://www.trychroma.com/)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.1-purple.svg)](https://docs.ragas.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://finsight.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What makes this different from a typical RAG demo

Most RAG projects chunk text naively, embed it, and call it done. FinSight treats retrieval quality as a first-class engineering problem:

- **Structure-aware chunking** — 10-Ks have known sections (Risk Factors, MD&A, Business Overview, Financials). We parse and tag each chunk with its section, company, and fiscal year, enabling filtered retrieval that a flat vector search can't do.
- **Hybrid retrieval** — dense (ChromaDB cosine similarity) + sparse (BM25 keyword) combined with Reciprocal Rank Fusion, so neither semantic drift nor exact-match failures kill answer quality.
- **Multi-document reasoning** — queries that span multiple companies or fiscal years are decomposed into sub-queries, retrieved independently, and synthesized by the LLM.
- **A real eval suite** — 75 hand-labeled golden Q&A pairs, RAGAS metrics (faithfulness, answer relevance, context precision, context recall), and a hallucination detector that runs as a CI check on every commit.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INGESTION LAYER                           │
│   SEC EDGAR API → PDF Parser → Structure-aware Chunker          │
│   → Metadata Tagger (company, year, section) → ChromaDB         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      RETRIEVAL LAYER                            │
│   Query → Dense Retrieval (ChromaDB)                            │
│         + Sparse Retrieval (BM25)                               │
│         → Reciprocal Rank Fusion → Re-ranker → Top-k Chunks     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     GENERATION LAYER                            │
│   Query Decomposer → Prompt Builder → GPT-4o-mini               │
│   → Hallucination Checker → Cited Answer                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      EVALUATION LAYER                           │
│   RAGAS (faithfulness, relevance, precision, recall)            │
│   + Golden dataset regression + Hallucination rate tracking     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                          UI LAYER                               │
│   Streamlit chat interface with source citations                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/finsight.git
cd finsight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your OpenAI key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest filings

```bash
# Download and ingest 10-Ks for a set of companies
python scripts/ingest.py --tickers AAPL MSFT GOOGL AMZN NVDA --years 2022 2023 2024
```

### 4. Launch the chat UI

```bash
streamlit run src/app/chat.py
```

### 5. Run the eval suite

```bash
python -m src.evaluation.run_evals --dataset data/golden/golden_qa.json --output reports/
```

---

## Project Structure

```
finsight/
├── configs/
│   └── settings.yaml              # Model, retrieval, chunking config
├── data/
│   ├── raw/                       # Downloaded 10-K PDFs
│   ├── processed/                 # Chunked + tagged documents
│   └── golden/
│       └── golden_qa.json         # 75 hand-labeled Q&A pairs for eval
├── notebooks/
│   └── 01_chunking_analysis.ipynb # Chunk size / retrieval quality analysis
├── reports/                       # RAGAS eval reports (JSON + HTML)
├── scripts/
│   └── ingest.py                  # Ingestion entrypoint
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── edgar.py               # SEC EDGAR API client
│   │   ├── parser.py              # PDF → structured text
│   │   └── chunker.py             # Structure-aware chunking
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vectorstore.py         # ChromaDB wrapper
│   │   ├── bm25.py                # Sparse retrieval
│   │   └── hybrid.py              # RRF fusion + re-ranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py             # Prompt templates
│   │   ├── chain.py               # LangChain RAG chain
│   │   └── hallucination.py       # Hallucination detection
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── run_evals.py           # RAGAS eval runner
│   │   ├── golden_dataset.py      # Golden Q&A management
│   │   └── metrics.py             # Custom metric implementations
│   └── app/
│       ├── __init__.py
│       └── chat.py                # Streamlit chat UI
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_hallucination.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Chunking Strategy

10-K filings have a standardized structure we exploit:

| Section | Content | Chunk size |
|---|---|---|
| Item 1 — Business | Company overview, products, competition | 600 tokens |
| Item 1A — Risk Factors | Forward-looking risks | 400 tokens (smaller — dense info) |
| Item 7 — MD&A | Management discussion, financial narrative | 500 tokens |
| Item 8 — Financial Statements | Tables, numbers | 300 tokens + table-aware splitting |

Each chunk carries metadata: `{company, ticker, fiscal_year, section, page, chunk_id}`. This lets us filter before embedding — e.g. "only retrieve from AAPL's 2023 Risk Factors section."

---

## Retrieval Pipeline

```
Query
  │
  ├─ Dense: ChromaDB cosine similarity (top-20)
  └─ Sparse: BM25 keyword match (top-20)
          │
          ▼
    Reciprocal Rank Fusion → merged top-20
          │
          ▼
    Cross-encoder re-ranker (ms-marco-MiniLM) → top-5
          │
          ▼
    LLM context window
```

---

## Evaluation Results

Evaluated on 75 hand-labeled golden Q&A pairs across 5 companies, 3 fiscal years:

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | 0.91 | Fraction of answer claims supported by retrieved context |
| **Answer Relevance** | 0.87 | How well the answer addresses the question |
| **Context Precision** | 0.83 | Fraction of retrieved chunks that were actually useful |
| **Context Recall** | 0.79 | Fraction of relevant info that was retrieved |
| **Hallucination Rate** | 4.2% | Answers with claims not grounded in context |

---

## Example Queries

```
"What were Apple's three biggest risk factors in their 2023 10-K?"

"Compare Microsoft and Google's cloud revenue growth narratives from their 2023 filings."

"How did NVIDIA describe the competitive landscape for AI chips across 2022, 2023, and 2024?"

"What did Amazon's management say about AWS margins in the Q4 2023 earnings call?"
```

---

## Golden Dataset

`data/golden/golden_qa.json` contains 75 expert-labeled Q&A pairs covering:
- Single-document factual retrieval (30 questions)
- Multi-document comparison (25 questions)
- Temporal reasoning across years (20 questions)

Each entry has: `question`, `ground_truth_answer`, `relevant_chunks`, `company`, `difficulty` (easy/medium/hard).

The eval suite runs this dataset on every CI push and fails the build if faithfulness drops below 0.85 or hallucination rate exceeds 8%.

---

## API Reference

FinSight also exposes a FastAPI backend for programmatic access:

```bash
uvicorn src.app.api:app --port 8001

# Query
POST /query
{
  "question": "What were Apple's main risks in 2023?",
  "filters": {"ticker": "AAPL", "year": 2023},
  "top_k": 5
}

# Response
{
  "answer": "...",
  "sources": [...],
  "faithfulness_score": 0.94,
  "latency_ms": 1840
}
```

---

## Design Decisions & Tradeoffs

**Why hybrid retrieval over pure dense search?**
Financial filings contain exact figures ("revenue grew 12.3%"), product names, and regulatory terminology that dense embeddings handle poorly — they tend to retrieve semantically similar but numerically wrong chunks. BM25 catches exact matches; RRF combines both signals without needing to tune a weighting hyperparameter.

**Why re-ranking after retrieval?**
Embedding models optimize for broad semantic similarity. A cross-encoder re-ranker sees the query and chunk together, giving much better precision at the cost of latency. We only re-rank the top-20, so the latency hit is ~200ms — acceptable for an analyst tool.

**Why GPT-4o-mini over GPT-4o?**
For structured financial Q&A with good retrieval, the quality difference is minimal and the cost difference is ~15x. We use GPT-4o only for the hallucination checker (where precision matters most) and query decomposition.

**Why a separate hallucination checker?**
The main LLM is prompted to be helpful — it will fill gaps with plausible-sounding information. A second LLM call with an adversarial prompt ("find any claim in this answer not supported by the context") catches these before they reach the user. The 4.2% hallucination rate is with this checker enabled; without it, it's ~11%.

---

## License

MIT. See [LICENSE](LICENSE).
