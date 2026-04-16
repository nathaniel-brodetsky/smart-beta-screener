# 📈 Smart Beta Screener — Interactive Factor Investing Dashboard

A production-grade quantitative investing dashboard that ranks S&P 100 stocks
using classical Smart Beta factors (Momentum, Low Volatility, Value, Quality)
with user-configurable weights.

---

## Architecture

```
├── data/                  # Cached CSVs (auto-created at runtime)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # UniverseLoader, MarketDataFetcher
│   ├── factor_engine.py   # FactorEngine
│   └── ranker.py          # PortfolioRanker, FactorWeights
├── app.py                 # Streamlit UI entry point
├── requirements.txt
└── README.md
```

### Core Classes

| Class | Module | Responsibility |
|---|---|---|
| `UniverseLoader` | `data_loader.py` | Scrapes / caches S&P 100 tickers from Wikipedia |
| `MarketDataFetcher` | `data_loader.py` | Batched yfinance downloads with retry back-off |
| `FactorEngine` | `factor_engine.py` | Vectorised factor computation (no `iterrows`) |
| `PortfolioRanker` | `ranker.py` | Cross-sectional Z-score normalisation + weighting |
| `FactorWeights` | `ranker.py` | Dataclass container for user-supplied weights |

---

## Factors

| Factor | Formula | Direction |
|---|---|---|
| **Momentum** | 12M return − 1M return | Higher = better |
| **Low Volatility** | −(annualised σ of log-returns) | Higher = better (less vol) |
| **Value** | −(trailing P/E ratio) | Higher = better (cheaper) |
| **Quality** | Return on Equity (ROE) | Higher = better |

All four factors are Z-score normalised cross-sectionally before being combined:

$$ Z_i = \frac{x_i - \mu}{\sigma} $$

$$ \text{Composite}_i = \sum_{j \in F} (w_j \times Z_{i,j}) $$

*where* $F = \{\text{Momentum}, \text{Volatility}, \text{Value}, \text{Quality}\}$
```

---

## Setup & Run

```bash
# 1. Clone and enter project
git clone <repo>
cd smart-beta-screener

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch dashboard
streamlit run app.py
```

---

## Design Decisions

- **Caching**: `st.cache_data(ttl=3600)` prevents re-downloading on slider moves.
- **Batching**: yfinance calls are split into batches of 10 with exponential back-off.
- **Vectorisation**: All pandas operations use column-wise broadcasting — zero `iterrows`.
- **Robustness**: Every external call is wrapped in `try/except` with structured logging.
- **OOP**: Each class has a single responsibility, loose coupling, and full type hints.