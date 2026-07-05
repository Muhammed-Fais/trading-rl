import pandas as pd
import pytest

from trading_rl.backtest.portfolio_report import (
    activity_metrics,
    combine_equal_weight_portfolio,
    monthly_returns,
)


def _history(values: list[float], benchmark: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"),
            "portfolio_value": values,
            "benchmark_value": benchmark,
            "position_fraction": [0.5] * len(values),
            "turnover": [0.01] * len(values),
        }
    )


def test_combine_equal_weight_portfolio_sums_symbol_equity() -> None:
    combined = combine_equal_weight_portfolio(
        {
            "BTCUSDT": _history([100.0, 110.0], [100.0, 120.0]),
            "ETHUSDT": _history([100.0, 90.0], [100.0, 80.0]),
        }
    )

    assert combined["portfolio_value"].tolist() == [200.0, 200.0]
    assert combined["benchmark_value"].tolist() == [200.0, 200.0]
    assert combined["position_fraction"].iloc[-1] == pytest.approx(0.5)
    assert combined["portfolio_return"].iloc[-1] == pytest.approx(0.0)


def test_monthly_returns_compares_first_and_last_values() -> None:
    portfolio = combine_equal_weight_portfolio(
        {"BTCUSDT": _history([100.0, 110.0], [100.0, 120.0])}
    )

    monthly = monthly_returns(portfolio)

    assert monthly.iloc[0]["portfolio_return"] == pytest.approx(0.10)
    assert monthly.iloc[0]["benchmark_return"] == pytest.approx(0.20)


def test_activity_metrics_counts_months_above_exposure_threshold() -> None:
    monthly = pd.DataFrame({"average_exposure": [0.10, 0.02, 0.05]})

    metrics = activity_metrics(monthly, active_exposure_threshold=0.05)

    assert metrics["active_months"] == 2.0
    assert metrics["inactive_months"] == 1.0
    assert metrics["active_month_ratio"] == pytest.approx(2 / 3)
