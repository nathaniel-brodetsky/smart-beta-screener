import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/"

_SP100_FALLBACK: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "V", "XOM", "MA", "PG", "JNJ", "COST", "HD", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "NFLX", "WMT", "KO", "AMD", "PEP", "TMO",
    "ACN", "MCD", "CSCO", "ADBE", "ABT", "LIN", "DHR", "WFC", "TXN", "NEE",
    "PM", "INTC", "ORCL", "RTX", "UNP", "AMGN", "QCOM", "HON", "CAT", "IBM",
    "GE", "LOW", "INTU", "SPGI", "MDT", "GS", "BMY", "BLK", "SYK", "C",
    "ISRG", "DE", "AXP", "BKNG", "TJX", "ELV", "VRTX", "MMC", "PLD", "CB",
    "ADI", "GILD", "MO", "ZTS", "CI", "SCHW", "SO", "DUK", "EOG", "CL",
    "BSX", "ITW", "AON", "SHW", "FCX", "MU", "CME", "USB", "BDX", "NOC",
    "ETN", "PNC", "MSI", "WM", "GD", "HUM", "NSC", "FDX", "EMR", "AIG",
]


class UniverseLoader:
    def __init__(self, cache_dir: Optional[Path] = None)->None:
        self.cache_dir = cache_dir
        self.cache: Optional[Path] = (
            cache_dir / "sp100_tickers.csv" if cache_dir else None
        )

    def load(self) -> List[str]:
        if self._cache_file and self._cache_file.exists():
            tickers = self._read_cache()
            if tickers:
                logger.info(f"Loaded {len(tickers)} tickers from cache")
                return tickers
            tickers = self._scrape_wikipedia()
            if not tickers:
                logger.warning("Wikipedia has no tickers")
                tickers = SP100_FALLBACK.copy()

            tickers = sorted(set(tickers))
            self._write_cache(tickers)
            return tickers

    def _scrape_wikipedia(self) -> List[str]:
        try:
            response = requests.get(_WIKI_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            table = soup.find("table", {"id": "constituents"})
            if table is None:
                return []
            rows = table.find_all("tr")[1:]
            tickers: List[str] = []
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    raw = cells[0].get_text(strip=True).replace(".","-")
                    tickers.append(raw.upper())
                return tickers
        except Exception as exc:
            logger.error("Wikipedia scrape error %s", exc)
            return []

        def _read_cache() -> List[str]:
            try:
                df = pd.read_csv(self.cache, header=None)
                return df.iloc[:,0].astype(str).tolist()
            except Exception:
                return []

        def _write_cache(self, tickers: List[str]) -> None:
            if self.cache_file is None:
                return
            try:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                pd.Series(tickers).to_csv(self.cache, header=False, index=False)
            except OSError as exc:
                logger.warning(f"Failed to write cache to {self.cache}: {exc}")
