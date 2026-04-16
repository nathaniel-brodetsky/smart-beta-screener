from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class FactorWeights:
    momentum: float = 1.0
    volatility: float = 1.0
    value: float = 1.0
    quality: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "momentum": self.momentum,
            "volatility": self.volatility,
            "value": self.value,
            "quality": self.quality,
        }

    def normalized(self) -> Dict[str, float]:
        raw = self.as_dict()
        total = sum(raw.values())
        if total == 0.0:
            n = len(raw)
            return{k: 1.0/n for k in raw}
        return{k: v/total for k, v in raw.items()}

class PortfolioRanker:

    _EXPECTED_FACTORS: List[str] = ["momentum", "volatility", "value", "quality"]

    def __init__(self,
                 factor_df: pd.DataFrame,
                 weights: FactorWeights,
                 min_valid_factors: int = 2)-> None:
        if factor_df.empty:
            raise ValueError("factor_df.empty")

        missing_cols = [c for c in self._EXPECTED_FACTORS if c not in factor_df.columns]
        if missing_cols:
            raise ValueError(f"missing columns: {missing_cols}")

        self._factor_df = factor_df[self._EXPECTED_FACTORS].copy()
        self.weights = weights
        self._min_valid_factors = min_valid_factors

    def rank(self) -> pd.DataFrame:
        z_scores = self._compute_z_scores()
        weighted = self._apply_weights(z_scores)
        composite = weighted.sum(axis=1, skipna=True)

        valid_factor_counts = z_scores.notna().sum(axis=1)
        composite = composite.where(valid_factor_counts>=self._min_valid_factors)

        result = z_scores.loc[composite.index].copy()
        result.columns = pd.Index([f"z_{c}" for c in result.columns])
        result["composite_score"] = composite
        result.sort_values("composite_score", ascending=False, inplace=True)
        result["rank"] = range(1, len(result)+1)

        logger.info("Ranking complete: %d tickers scored and ranked.", len(result))
        return result

    def get_z_scores(self) -> pd.DataFrame:
        return self._compute_z_scores()

    def _compute_z_scores(self) -> pd.DataFrame:
        z_df = self._factor_df.copy()

        for col in self._EXPECTED_FACTORS:
            series = z_df[col].dropna()
            if len(series) < 2:
                z_df[col]=float("nan")
                continue

            z_values: pd.Series = pd.Series(
                stats.zscore(series.values, ddof=1),
                index=series.index,
                dtype=float,
            )
            z_df[col] = z_values

        return z_df

    def _apply_weights(self, z_df: pd.DataFrame) -> pd.DataFrame:
        norm_weights = self.weights.normalized()
        weighted = z_df.copy()
        for col in self._EXPECTED_FACTORS:
            weighted[col] = z_df[col] * norm_weights[col]
        return weighted