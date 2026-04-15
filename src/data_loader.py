import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/"

_SP100_FALLBACK: LIST[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO",
    "JPM", "TSLA", "UNH", "V", "XOM", "MA", "PG", "JNJ", "COST", "HD", "MRK"
]