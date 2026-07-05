import pandas as pd
import pytest

from trading_rl.backtest.portfolio_report import combine_equal_weight_portfolio, monthly_returns


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
