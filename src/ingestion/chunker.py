"""
src/ingestion/chunker.py
------------------------
Structure-aware chunking for 10-K filings.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from src.ingestion.parser import ParsedFiling, ParsedSection

log = logging.getLogger(__name__)

# Max chars per chunk — keep well under 8192 token OpenAI embedding limit
# 1 token ≈ 4 chars, so 1500 chars ≈ 375 tokens (safe headroom)
SECTION_CHUNK_CHARS = {
    "business":             1500,
    "risk_factors":         1200,
    "mda":                  1500,
    "financial_statements": 1000,
    "unknown":              1500,
}
OVERLAP_CHARS    = 200   # fixed overlap in chars
MIN_CHUNK_CHARS  = 100   # discard chunks shorter than this


@dataclass
class Chunk:
    chunk_id:                str
    text:                    str
    ticker:                  str
    company:                 str
    fiscal_year:             int
    section_key:             str
    section_label:           str
    chunk_index:             int
    total_chunks_in_section: int
    char_count:              int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_metadata(self) -> dict:
        return {
            "chunk_id":      self.chunk_id,
            "ticker":        self.ticker,
            "company":       self.company,
            "fiscal_year":   self.fiscal_year,
            "section_key":   self.section_key,
            "section_label": self.section_label,
            "chunk_index":   self.chunk_index,
            "total_chunks":  self.total_chunks_in_section,
        }


def _make_chunk_id(ticker: str, fiscal_year: int, section: str, idx: int) -> str:
    raw = f"{ticker}_{fiscal_year}_{section}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of at most max_chars.

    Strategy:
    1. Split into sentences (on '. ', '.\n', '\n')
    2. Greedily accumulate sentences until we hit max_chars
    3. Carry last `overlap` chars into the next chunk
    """
    # Split into sentences — split on period+space, newlines, or semicolons
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]

    if not sentences:
        # No sentences found — hard split every max_chars chars
        chunks = []
        for i in range(0, len(text), max_chars - overlap):
            chunk = text[i:i + max_chars]
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
        return chunks

    raw_chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len > max_chars and current:
            raw_chunks.append(' '.join(current))
            # Carry overlap: keep sentences from the end totalling ~overlap chars
            carry: list[str] = []
            carry_len = 0
            for s in reversed(current):
                if carry_len + len(s) > overlap:
                    break
                carry.insert(0, s)
                carry_len += len(s)
            current = carry
            current_len = carry_len

        current.append(sent)
        current_len += sent_len

    if current:
        raw_chunks.append(' '.join(current))

    return [c for c in raw_chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_section(section: ParsedSection, ticker: str, company: str,
                  fiscal_year: int) -> list[Chunk]:
    max_chars = SECTION_CHUNK_CHARS.get(section.section_key, 1500)
    raw_chunks = _split_text(section.text, max_chars, OVERLAP_CHARS)

    if not raw_chunks:
        return []

    total = len(raw_chunks)
    return [
        Chunk(
            chunk_id=_make_chunk_id(ticker, fiscal_year, section.section_key, idx),
            text=text,
            ticker=ticker,
            company=company,
            fiscal_year=fiscal_year,
            section_key=section.section_key,
            section_label=section.section_label,
            chunk_index=idx,
            total_chunks_in_section=total,
        )
        for idx, text in enumerate(raw_chunks)
    ]


def chunk_filing(filing: ParsedFiling) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for section in filing.sections:
        chunks = chunk_section(
            section,
            ticker=filing.ticker,
            company=filing.company,
            fiscal_year=filing.fiscal_year,
        )
        all_chunks.extend(chunks)
        log.debug("  %s: %d chunks", section.section_key, len(chunks))

    log.info("Chunked %s %d: %d sections → %d chunks",
             filing.ticker, filing.fiscal_year,
             len(filing.sections), len(all_chunks))
    return all_chunks