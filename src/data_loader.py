import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_100_companies"

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
        self._cache_dir = cache_dir
        self._cache_file: Optional[Path] = (
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
            tickers = _SP100_FALLBACK.copy()

        tickers = sorted(set(tickers))
        self._write_cache(tickers)
        return tickers

    def _scrape_wikipedia(self) -> List[str]:
        try:
            response = requests.get(_WIKI_URL, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
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

    def _read_cache(self) -> List[str]:
        try:
            df = pd.read_csv(self._cache_file, header=None)
            return df.iloc[:,0].astype(str).tolist()
        except Exception:
            return []

    def _write_cache(self, tickers: List[str]) -> None:
        if self._cache_file is None:
            return
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            pd.Series(tickers).to_csv(self._cache_file, header=False, index=False)
        except OSError as exc:
            logger.warning(f"Failed to write cache to {self._cache_file}: {exc}")



_BATCH_SIZE = 10
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 2.0

class MarketDataFetcher:
    def __init__(
            self,
            period: str = "2y",
            batch_size: int = _BATCH_SIZE,
            retry_attempts: int = _RETRY_ATTEMPTS,
            retry_backoff: float = _RETRY_BACKOFF,
    ) -> None:
        self._period = period
        self._batch_size = batch_size
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff

    def fetch(self, tickers: List[str])->Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        all_prices: List[pd.DataFrame] = []
        fundamentals: Dict[str, Dict[str, float]] = {}

        batches = [tickers[i: i + self._batch_size] for i in range(0, len(tickers), self._batch_size)]
        for batch in batches:
            price_chunk = self._download_prices_with_retry(batch)
            if price_chunk is not None and not price_chunk.empty:
                all_prices.append(price_chunk)

            for ticker in batch:
                fundamentals[ticker] = self._fetch_fundamentals(ticker)

        if not all_prices:
            return pd.DataFrame(), {}

        prices = pd.concat(all_prices, axis=1).ffill().dropna(axis=1,how="all")
        valid_tickers = [t for t in prices.columns if t in fundamentals]
        prices = prices[valid_tickers]

        return prices, fundamentals

    def _download_prices_with_retry(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        delay = self._retry_backoff
        for attempt in range(1, self._retry_attempts + 1):
            try:
                raw = yf.download(tickers, period=self._period, auto_adjust=True, progress=False, threads=True)
                if raw.empty:
                    raise ValueError("Empty DataFrame")

                if isinstance(raw.columns, pd.MultiIndex):
                    return raw["Close"].copy()
                else:
                    prices = raw[["Close"]].copy()
                    prices.columns = pd.Index(tickers)
                    return prices
            except Exception as exc:
                if attempt < self._retry_attempts:
                    time.sleep(delay)
                    delay *= 2
        return None

    def _fetch_fundamentals(self, ticker: str) -> Dict[str, float]:
        nan = float("nan")
        result: Dict[str, float] = {"pe_ratio": nan, "roe": nan}
        try:
            info = yf.Ticker(ticker).info
            raw_pe = info.get("trailingPE") or info.get("forwardPE")
            raw_roe = info.get("returnOnEquity")

            result["pe_ratio"] = float(raw_pe) if raw_pe is not None else nan
            result["roe"] = float(raw_roe) if raw_roe is not None else nan
        except Exception:
            pass
        return result