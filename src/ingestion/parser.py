"""
src/ingestion/parser.py
-----------------------
Parses SEC 10-K iXBRL filings into structured sections.

Key insight from Apple's 2023 10-K:
  - Section headers appear as "Item 1A.\xa0\xa0\xa0\xa0Risk Factors" (non-breaking spaces)
  - The same headers appear in the Table of Contents (short context, page numbers follow)
  - Real content sections have substantial text immediately after the header
  - We find ALL occurrences and keep the one with the most following content
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SECTION_LABELS = {
    "business":             "Business Overview",
    "risk_factors":         "Risk Factors",
    "mda":                  "Management Discussion & Analysis",
    "financial_statements": "Financial Statements",
}

# Patterns to detect section header lines — handle both "Item 1A." and "Item 1A.\xa0\xa0"
SECTION_PATTERNS = [
    (r'item\s+1a[\.\s\xa0]',  "risk_factors"),
    (r'item\s+1b[\.\s\xa0]',  "after_risk"),      # marks end of risk factors
    (r'item\s+1[\.\s\xa0]',   "business"),
    (r'item\s+7a[\.\s\xa0]',  "after_mda"),
    (r'item\s+7[\.\s\xa0]',   "mda"),
    (r'item\s+8[\.\s\xa0]',   "financial_statements"),
    (r'item\s+9[\.\s\xa0]',   "after_financials"),
]

# Which sections we actually want to index
KEEP_SECTIONS = {"business", "risk_factors", "mda", "financial_statements"}


@dataclass
class ParsedSection:
    section_key:   str
    section_label: str
    text:          str
    char_count:    int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ParsedFiling:
    ticker:      str
    company:     str
    fiscal_year: int
    source_path: str
    sections:    list[ParsedSection] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(s.char_count for s in self.sections)


def _extract_text(raw: str) -> str:
    """Strip HTML tags using BeautifulSoup, return clean plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = re.sub(r'<[^>]+>', '\n', raw)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _find_all_occurrences(lines: list[str], pattern: str) -> list[int]:
    """Return all line indices where pattern matches (case-insensitive)."""
    hits = []
    for i, line in enumerate(lines):
        # Normalize: replace \xa0 with space for matching
        normalized = line.strip().lower().replace('\xa0', ' ')
        if re.match(pattern, normalized):
            hits.append(i)
    return hits


def _best_occurrence(lines: list[str], candidates: list[int],
                     min_content_lines: int = 10) -> int | None:
    """
    Among candidate line indices, return the one followed by the most
    content lines before the next short/numeric line.
    This distinguishes real content from Table of Contents entries.
    """
    best_idx = None
    best_count = 0

    for idx in candidates:
        # Count non-trivial lines following this header
        count = 0
        for j in range(idx + 1, min(idx + 50, len(lines))):
            l = lines[j].strip()
            # Stop if we hit a page number or another item header
            if re.match(r'^\d+$', l):
                break
            if len(l) > 30:
                count += 1
        if count > best_count:
            best_count = count
            best_idx = idx

    if best_idx is not None and best_count >= min_content_lines:
        return best_idx
    # If nothing passes the threshold, just return the last occurrence
    # (ToC is always before content)
    return candidates[-1] if candidates else None


def parse_filing(path: Path, ticker: str, company: str, fiscal_year: int) -> ParsedFiling:
    log.info("Parsing %s %d from %s ...", ticker, fiscal_year, path)

    raw   = path.read_text(encoding="utf-8", errors="replace")
    text  = _extract_text(raw)
    lines = text.split('\n')

    # Find the best (real content, not ToC) line index for each section
    section_starts: dict[str, int] = {}
    for pattern, key in SECTION_PATTERNS:
        candidates = _find_all_occurrences(lines, pattern)
        if candidates:
            best = _best_occurrence(lines, candidates)
            if best is not None:
                section_starts[key] = best
                log.debug("  %-25s line %d (of %d candidates)", key, best, len(candidates))

    if not section_starts:
        log.warning("No sections found for %s %d", ticker, fiscal_year)
        return ParsedFiling(ticker=ticker, company=company,
                            fiscal_year=fiscal_year, source_path=str(path))

    # Sort sections by their line position
    ordered = sorted(section_starts.items(), key=lambda x: x[1])

    # Build section texts: from each section's start line to the next section's start
    parsed_sections = []
    for i, (key, start_line) in enumerate(ordered):
        if key not in KEEP_SECTIONS:
            continue

        # Find the end: the start of the next KEEP section (skip sentinel keys)
        end_line = len(lines)
        for j in range(i + 1, len(ordered)):
            next_key, next_start = ordered[j]
            end_line = next_start
            break

        body = '\n'.join(lines[start_line:end_line]).strip()
        # Remove the header line itself from the body start
        body_lines = body.split('\n')
        if body_lines and re.match(
            r'item\s+\d', body_lines[0].strip().lower().replace('\xa0', ' ')
        ):
            body = '\n'.join(body_lines[1:]).strip()

        if len(body) < 500:
            log.debug("  Skipping %s — too short (%d chars)", key, len(body))
            continue

        # Cap at 150k chars
        body = body[:150_000]

        parsed_sections.append(ParsedSection(
            section_key=key,
            section_label=SECTION_LABELS[key],
            text=body,
        ))
        log.debug("  %-25s %d chars", key, len(body))

    filing = ParsedFiling(
        ticker=ticker,
        company=company,
        fiscal_year=fiscal_year,
        source_path=str(path),
        sections=parsed_sections,
    )
    log.info("Parsed %s %d: %d sections, %d total chars",
             ticker, fiscal_year, len(parsed_sections), filing.total_chars)
    return filing