import pandas as pd
import pytest

from trading_rl.backtest.failure_diagnostics import _compound_return, _failure_events


def test_compound_return_multiplies_path_returns() -> None:
    returns = pd.Series([0.10, -0.10])

    assert _compound_return(returns) == pytest.approx(-0.01)


def test_failure_events_returns_worst_negative_months_first() -> None:
    rows = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "month": "2024-01",
                "portfolio_return": -0.10,
                "benchmark_return": 0.20,
                "drawdown": -0.1,
                "position_fraction": 0.5,
                "turnover": 0.01,
            },
            {
                "symbol": "BTCUSDT",
                "month": "2024-02",
                "portfolio_return": 0.05,
                "benchmark_return": 0.10,
                "drawdown": -0.02,
                "position_fraction": 0.5,
                "turnover": 0.01,
            },
            {
                "symbol": "ETHUSDT",
                "month": "2024-01",
                "portfolio_return": -0.20,
                "benchmark_return": -0.05,
                "drawdown": -0.2,
                "position_fraction": 0.2,
                "turnover": 0.0,
            },
        ]
    )

    failures = _failure_events(rows)

    assert failures.iloc[0]["symbol"] == "ETHUSDT"
    assert failures.iloc[1]["symbol"] == "BTCUSDT"
