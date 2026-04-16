from __future__ import annotations

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR: int=252

class FactorEngine:
    def __init__(self,
                 prices: pd.DataFrame,
                 fundamentals: Dict[str, Dict[str, float]],
                 )-> None:
        if prices.empty:
            raise ValueError("No prices")
        if len(prices) < 22:
            raise ValueError(f"Price must contain at least 22 raws; got {len(prices)}")
        self._prices = prices.copy()
        self._fundamentals = fundamentals
        self._log_returns: Optional[pd.DataFrame] = None

    def compute_all_factors(self) -> pd.DataFrame:
        momentum = self.compute_momentum()
        volatility = self.compute_volatility()
        value = self.compute_value()
        quality = self.compute_quality()

        factor_df = pd.DataFrame(
            {
                "momentum": momentum,
                "volatility": volatility,
                "value": value,
                "quality": quality,
            }
        )

        factor_df.dropna(how="all",inplace=True)

        logger.info(
            "Factor matrix built: %d tickers x %d factors",
            factor_df.shape[0],
            factor_df.shape[1],
        )
        return factor_df

    def compute_momentum(self) -> pd.Series:
        prices = self._prices
        try:
            ret_12m = (prices.iloc[-1] / prices.iloc[-min(252, len(prices))]) - 1
            ret_1m = (prices.iloc[-1] / prices.iloc[-min(21, len(prices))]) - 1
            momentum: pd.Series = ret_12m * ret_1m
        except Exception as exc:
            logger.error("Failed to compute momentum: %s", exc)
            momentum = pd.Series(dtype=float)

        momentum.name="momentum"

        return momentum

    def compute_volatility(self) -> pd.Series:
        log_returns=self.get_log_returns()
        try:
            raw_vol:pd.Series=log_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)
            volatility: pd.Series= -raw_vol
        except Exception as exc:
            logger.error("Failed to compute volatility: %s", exc)
            volatility = pd.Series(dtype=float)

        volatility.name="volatility"
        return volatility

    def compute_value(self) -> pd.Series:
        tickers = self._prices.columns.tolist()
        pe_series = pd.Series(
            {t: self._fundamentals.get(t, {}).get("pe_ratio", float("nan")) for t in tickers},
            dtype=float,
        )

        pe_series = pe_series.where(pe_series > 0)

        value: pd.Series = -pe_series
        value.name="value"
        return value

    def compute_quality(self) -> pd.Series:
        tickers = self._prices.columns.tolist()
        roe_series=pd.Series(
            {t: self._fundamentals.get(t, {}).get("roe", float("nan")) for t in tickers},
            dtype=float,
        )
        roe_series.name="roe_series"
        return roe_series

    def get_log_returns(self) -> pd.DataFrame:
        if self._log_returns is None:
            self._log_returns = np.log(
                self._prices / self._prices.shift(1)
            ).dropna(how="all")
        return self._log_returns
