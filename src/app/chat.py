"""
src/app/chat.py
---------------
Streamlit chat interface for FinSight.

Features:
  - Conversational UI with chat history
  - Source citations expandable below each answer
  - Grounding score and hallucination flag displayed
  - Filter sidebar (ticker, year, section)
  - Query latency shown

Run:
    streamlit run src/app/chat.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is on the path when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinSight — Financial Filing Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .grounding-high  { color: #16a34a; font-weight: 600; }
    .grounding-med   { color: #d97706; font-weight: 600; }
    .grounding-low   { color: #dc2626; font-weight: 600; }
    .source-card     { background: #f8fafc; border-left: 3px solid #3b82f6;
                       padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0;
                       color: #111827; }
    .latency-badge   { color: #6b7280; font-size: 0.8rem; }
    .hallucination-warn { background: #fef2f2; border: 1px solid #fecaca;
                          padding: 0.5rem; border-radius: 4px; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — filters and settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.shields.io/badge/FinSight-📊_Financial_RAG-blue", use_column_width=True)
    st.markdown("---")

    st.subheader("🔍 Retrieval Filters")
    ticker_options = ["All", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    selected_ticker = st.selectbox("Company", ticker_options)

    year_options = ["All", 2024, 2023, 2022, 2021]
    selected_year = st.selectbox("Fiscal Year", year_options)

    section_options = ["All", "Business Overview", "Risk Factors", "MD&A", "Financial Statements"]
    selected_section = st.selectbox("Section", section_options)

    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    run_hallucination_check = st.toggle("Hallucination check", value=True)

    st.markdown("---")
    st.subheader("💡 Example Queries")
    example_queries = [
        "What were Apple's biggest risk factors in 2023?",
        "Compare Microsoft and Google's cloud strategies in 2023",
        "How did Nvidia describe AI chip demand across 2022 and 2023?",
        "What did Amazon say about AWS competitive risks?",
        "How has Meta's Reality Labs investment thesis evolved?",
    ]
    for eq in example_queries:
        if st.button(eq, key=eq, width='stretch'):
            st.session_state["prefill"] = eq

    st.markdown("---")
    st.caption("Built with LangChain · ChromaDB · GPT-4o-mini · RAGAS")

# ---------------------------------------------------------------------------
# Chain initialisation (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading FinSight...")
def load_chain(top_k: int, run_hallucination_check: bool):
    """Load the full RAG chain once and cache it."""
    try:
        from src.retrieval.vectorstore import VectorStore
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.hybrid import HybridRetriever
        from src.generation.chain import FinSightChain

        vs     = VectorStore()
        bm25   = BM25Retriever()
        bm25.load()
        hybrid = HybridRetriever(vs, bm25, use_reranker=False)
        chain  = FinSightChain(
            hybrid,
            top_k=top_k,
            run_hallucination_check=run_hallucination_check,
        )
        return chain, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("📊 FinSight")
st.caption("Analyst-grade Q&A over SEC 10-K filings and earnings transcripts")

# Warn if no API key
if not os.environ.get("OPENAI_API_KEY"):
    st.warning(
        "⚠️ OPENAI_API_KEY not set. Set it in your environment or `.env` file to use FinSight.",
        icon="⚠️",
    )

chain, load_error = load_chain(top_k, run_hallucination_check)

if load_error:
    st.error(f"Failed to load FinSight: {load_error}")
    st.info("Make sure you've run `python scripts/ingest.py` first to populate the vector store.")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            _render_response_metadata(msg["metadata"])

# ---------------------------------------------------------------------------
# Helper to render source citations and metadata
# ---------------------------------------------------------------------------

def _render_response_metadata(meta: dict):
    cols = st.columns([1, 1, 1, 2])
    grounding = meta.get("grounding_score", 1.0)
    g_class = "grounding-high" if grounding >= 0.85 else "grounding-med" if grounding >= 0.7 else "grounding-low"
    cols[0].markdown(f'<span class="{g_class}">Grounding: {grounding:.0%}</span>', unsafe_allow_html=True)
    cols[1].markdown(f'<span class="latency-badge">⏱ {meta.get("latency_ms", 0)}ms</span>', unsafe_allow_html=True)
    cols[2].markdown(f'<span class="latency-badge">📄 {meta.get("chunks_retrieved", 0)} chunks</span>', unsafe_allow_html=True)
    cols[3].markdown(f'<span class="latency-badge">Mode: {meta.get("query_mode", "?")}</span>', unsafe_allow_html=True)

    if not meta.get("is_grounded", True):
        st.markdown(
            f'<div class="hallucination-warn">⚠️ Potential hallucination detected. '
            f'Unsupported claims: {", ".join(meta.get("unsupported_claims", [])[:2])}</div>',
            unsafe_allow_html=True,
        )

    if meta.get("sources"):
        with st.expander(f"📎 Sources ({len(meta['sources'])} chunks)"):
            for src in meta["sources"]:
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong style="color:#1e3a5f">{src["company"]} ({src["ticker"]}) · FY{src["fiscal_year"]} · {src["section"]}</strong><br/>'
                    f'<small style="color:#374151">{src["text_snippet"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

prefill = st.session_state.pop("prefill", None)
prompt  = st.chat_input("Ask about any SEC filing...") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build filters
    filters: dict = {}
    if selected_ticker != "All":
        filters["ticker"] = selected_ticker
    if selected_year != "All":
        filters["fiscal_year"] = selected_year

    section_key_map = {
        "Business Overview": "business",
        "Risk Factors": "risk_factors",
        "MD&A": "mda",
        "Financial Statements": "financial_statements",
    }
    if selected_section != "All":
        filters["section_key"] = section_key_map[selected_section]

    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            try:
                response = chain.query(prompt, filters=filters or None)
                st.markdown(response.answer)

                meta = {
                    "grounding_score":    response.grounding_score,
                    "is_grounded":        response.is_grounded,
                    "unsupported_claims": response.unsupported_claims,
                    "latency_ms":         response.latency_ms,
                    "chunks_retrieved":   response.chunks_retrieved,
                    "query_mode":         response.query_mode,
                    "sources":            response.sources,
                }
                _render_response_metadata(meta)

                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  response.answer,
                    "metadata": meta,
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})