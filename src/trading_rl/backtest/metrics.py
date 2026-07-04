from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    benchmark_return: float
    excess_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    average_turnover: float
    average_exposure: float
    final_value: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "benchmark_return": self.benchmark_return,
            "excess_return": self.excess_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "average_turnover": self.average_turnover,
            "average_exposure": self.average_exposure,
            "final_value": self.final_value,
        }


def add_performance_columns(history: pd.DataFrame) -> pd.DataFrame:
    required = {"portfolio_value", "benchmark_value"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = history.copy()
    out["portfolio_return"] = out["portfolio_value"].pct_change().fillna(0.0)
    out["benchmark_return"] = out["benchmark_value"].pct_change().fillna(0.0)
    out["portfolio_peak"] = out["portfolio_value"].cummax()
    out["benchmark_peak"] = out["benchmark_value"].cummax()
    out["drawdown"] = out["portfolio_value"] / out["portfolio_peak"] - 1.0
    out["benchmark_drawdown"] = out["benchmark_value"] / out["benchmark_peak"] - 1.0
    return out


def calculate_metrics(
    history: pd.DataFrame,
    periods_per_year: int = 365 * 24,
) -> PerformanceMetrics:
    enriched = add_performance_columns(history)
    first_value = float(enriched["portfolio_value"].iloc[0])
    last_value = float(enriched["portfolio_value"].iloc[-1])
    first_benchmark = float(enriched["benchmark_value"].iloc[0])
    last_benchmark = float(enriched["benchmark_value"].iloc[-1])
    returns = enriched["portfolio_return"].to_numpy(dtype=float)

    total_return = last_value / first_value - 1.0
    benchmark_return = last_benchmark / first_benchmark - 1.0
    return PerformanceMetrics(
        total_return=float(total_return),
        benchmark_return=float(benchmark_return),
        excess_return=float(total_return - benchmark_return),
        max_drawdown=float(abs(enriched["drawdown"].min())),
        sharpe=_sharpe(returns, periods_per_year),
        sortino=_sortino(returns, periods_per_year),
        average_turnover=float(enriched.get("turnover", pd.Series([0.0])).mean()),
        average_exposure=float(enriched.get("position_fraction", pd.Series([0.0])).abs().mean()),
        final_value=last_value,
    )


def _sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0 or math.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int) -> float:
    downside = returns[returns < 0.0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(np.std(downside, ddof=1))
    if downside_std == 0.0 or math.isnan(downside_std):
        return 0.0
    return float(np.mean(returns) / downside_std * math.sqrt(periods_per_year))
