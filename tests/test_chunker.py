"""
tests/test_chunker.py
---------------------
Unit tests for the structure-aware chunking pipeline.
"""

from __future__ import annotations

import pytest
from src.ingestion.parser import ParsedSection, ParsedFiling
from src.ingestion.chunker import chunk_section, chunk_filing, SECTION_CHUNK_CHARS


def _make_section(key: str = "risk_factors", text: str = "") -> ParsedSection:
    if not text:
        # Generate text longer than one chunk
        text = " ".join([f"This is sentence number {i} in the risk factors section." for i in range(200)])
    from src.ingestion.parser import SECTION_LABELS
    return ParsedSection(
        section_key=key,
        section_label=SECTION_LABELS.get(key, "Other"),
        text=text,
    )


def _make_filing(n_sections: int = 3) -> ParsedFiling:
    keys = ["business", "risk_factors", "mda"]
    sections = [_make_section(k) for k in keys[:n_sections]]
    return ParsedFiling(
        ticker="TEST",
        company="Test Corp",
        fiscal_year=2023,
        source_path="test.htm",
        sections=sections,
    )


class TestChunkSection:
    def test_returns_list_of_chunks(self):
        section = _make_section()
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self):
        section = _make_section()
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        for chunk in chunks:
            assert chunk.ticker == "AAPL"
            assert chunk.company == "Apple Inc"
            assert chunk.fiscal_year == 2023
            assert chunk.section_key == "risk_factors"
            assert chunk.chunk_id
            assert chunk.text

    def test_chunk_ids_are_unique(self):
        section = _make_section()
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_size_respects_budget(self):
        section = _make_section("risk_factors")
        max_chars = SECTION_CHUNK_CHARS["risk_factors"]
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        # Allow some slack for overlap
        for chunk in chunks:
            assert chunk.char_count <= max_chars * 1.15, (
                f"Chunk too large: {chunk.char_count} > {max_chars * 1.15}"
            )

    def test_short_text_produces_single_chunk(self):
        short_text = "This is a very short section with minimal content."
        section = _make_section(text=short_text)
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        assert len(chunks) <= 1

    def test_metadata_in_to_metadata(self):
        section = _make_section()
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        meta = chunks[0].to_metadata()
        assert meta["ticker"] == "AAPL"
        assert meta["fiscal_year"] == 2023
        assert "section_key" in meta

    def test_total_chunks_field_correct(self):
        section = _make_section()
        chunks = chunk_section(section, "AAPL", "Apple Inc", 2023)
        expected_total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks_in_section == expected_total


class TestChunkFiling:
    def test_returns_flat_list(self):
        filing = _make_filing(n_sections=3)
        chunks = chunk_filing(filing)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_all_sections_represented(self):
        filing = _make_filing(n_sections=3)
        chunks = chunk_filing(filing)
        section_keys = {c.section_key for c in chunks}
        assert "business" in section_keys
        assert "risk_factors" in section_keys
        assert "mda" in section_keys

    def test_no_duplicate_chunk_ids_across_sections(self):
        filing = _make_filing(n_sections=3)
        chunks = chunk_filing(filing)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
