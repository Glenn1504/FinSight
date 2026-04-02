"""
src/ingestion/edgar.py
----------------------
SEC EDGAR API client for downloading 10-K filings.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

EDGAR_BASE       = "https://data.sec.gov"
TICKERS_URL      = "https://www.sec.gov/files/company_tickers.json"
HEADERS          = {"User-Agent": "FinSight research@finsight.ai"}
RATE_LIMIT_DELAY = 0.12


@dataclass
class Filing:
    ticker:      str
    company:     str
    cik:         str
    fiscal_year: int
    form_type:   str
    filed_date:  str
    accession:   str
    doc_url:     str


class EdgarClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        self._last_request = 0.0
        self._ticker_map: dict | None = None

    def _get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.time() - self._last_request
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        resp = self._session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        self._last_request = time.time()
        return resp

    def _load_ticker_map(self) -> dict:
        if self._ticker_map is not None:
            return self._ticker_map
        log.info("Loading SEC ticker → CIK map ...")
        data = self._get(TICKERS_URL).json()
        self._ticker_map = {v["ticker"].upper(): v for v in data.values()}
        log.info("Loaded %d tickers from SEC.", len(self._ticker_map))
        return self._ticker_map

    def get_cik(self, ticker: str) -> str:
        mapping = self._load_ticker_map()
        entry = mapping.get(ticker.upper())
        if not entry:
            raise ValueError(f"Ticker not found in SEC database: {ticker}")
        return str(entry["cik_str"]).zfill(10)

    def _get_primary_doc_url(self, cik: str, accession: str) -> str | None:
        """
        Fetch the filing index page and extract the primary 10-K document URL.

        EDGAR index pages have a table with columns: Seq, Description, Document, Type, Size.
        The main 10-K is always Type='10-K'. Its href may be wrapped in /ix?doc= 
        (inline XBRL viewer) — we strip that to get the raw document URL.
        """
        cik_int      = int(cik)
        index_url    = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_int}/{accession.replace('-', '')}/"
            f"{accession}-index.htm"
        )

        try:
            html = self._get(index_url).text

            # EDGAR table rows look like:
            # <td>10-K</td>
            # <td><a href="/ix?doc=/Archives/.../aapl-20230930.htm">aapl-20230930.htm</a></td>
            # <td>10-K</td>   ← Type column
            #
            # We find every <tr> block and check if the Type cell == "10-K"
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

            for row in rows:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
                if len(cells) < 4:
                    continue

                # Type is the 4th cell (index 3): Seq, Description, Document, Type, Size
                type_cell = re.sub(r'<[^>]+>', '', cells[3]).strip()
                if type_cell.upper() != "10-K":
                    continue

                # Document is the 3rd cell — extract href
                doc_cell = cells[2]

                # Handle /ix?doc=/Archives/... wrapper (inline XBRL)
                ix_match = re.search(r'/ix\?doc=(/Archives/[^"\']+)', doc_cell)
                if ix_match:
                    path = ix_match.group(1)
                    url  = f"https://www.sec.gov{path}"
                    log.info("Resolved 10-K doc (iXBRL): %s", path.split("/")[-1])
                    return url

                # Plain href
                plain_match = re.search(r'href="(/Archives/[^"\']+\.htm)"', doc_cell, re.IGNORECASE)
                if plain_match:
                    path = plain_match.group(1)
                    url  = f"https://www.sec.gov{path}"
                    log.info("Resolved 10-K doc (plain): %s", path.split("/")[-1])
                    return url

            log.warning("No Type=10-K row found in filing index: %s", index_url)

        except Exception as e:
            log.warning("Failed to parse filing index %s: %s", index_url, e)

        return None

    def get_10k_filings(self, ticker: str, years: list[int] | None = None) -> list[Filing]:
        cik          = self.get_cik(ticker)
        mapping      = self._load_ticker_map()
        company_name = mapping.get(ticker.upper(), {}).get("title", ticker)

        url  = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
        data = self._get(url).json()

        filings_raw = data.get("filings", {}).get("recent", {})
        forms       = filings_raw.get("form", [])
        dates       = filings_raw.get("filingDate", [])
        accessions  = filings_raw.get("accessionNumber", [])

        results = []
        for form, date, acc in zip(forms, dates, accessions):
            if form != "10-K":
                continue
            year = int(date[:4])
            if years and year not in years:
                continue

            doc_url = self._get_primary_doc_url(cik, acc)
            if not doc_url:
                log.warning("Could not resolve doc URL for %s %d — skipping.", ticker, year)
                continue

            results.append(Filing(
                ticker=ticker.upper(),
                company=company_name,
                cik=cik,
                fiscal_year=year,
                form_type="10-K",
                filed_date=date,
                accession=acc,
                doc_url=doc_url,
            ))
            log.info("Found 10-K: %s %d (filed %s) → %s",
                     ticker, year, date, doc_url.split("/")[-1])

        log.info("Found %d 10-K filing(s) for %s", len(results), ticker)
        return results

    def download_filing(self, filing: Filing, output_dir: str = "data/raw/") -> Path:
        out  = Path(output_dir) / filing.ticker / str(filing.fiscal_year)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "10k.htm"

        if dest.exists():
            size_kb = dest.stat().st_size / 1024
            if size_kb > 500:
                log.info("Already downloaded: %s (%.0f KB)", dest, size_kb)
                return dest
            else:
                log.info("Cached file too small (%.0f KB) — re-downloading.", size_kb)
                dest.unlink()

        log.info("Downloading %s %d → %s", filing.ticker, filing.fiscal_year, dest)
        resp    = self._get(filing.doc_url)
        content = resp.content
        dest.write_bytes(content)
        log.info("Saved %.0f KB → %s", len(content) / 1024, dest)
        return dest