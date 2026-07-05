import pandas as pd

from trading_rl.backtest.calendar_holdout import _filter_holdout, _rank_holdout


def test_filter_holdout_uses_inclusive_calendar_dates() -> None:
    df = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2023-12-31 23:00", "2024-01-01 00:00", "2024-01-02 00:00"],
                utc=True,
            ),
            "close": [1.0, 2.0, 3.0],
        }
    )

    out = _filter_holdout(df, "2024-01-01", "2024-01-01")

    assert out["close"].tolist() == [2.0]


def test_rank_holdout_rewards_breadth_and_drawdown_control() -> None:
    summary = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "policy": "stable",
                "total_return": 0.05,
                "max_drawdown": 0.1,
                "average_turnover": 0.01,
            },
            {
                "symbol": "ETHUSDT",
                "policy": "stable",
                "total_return": 0.04,
                "max_drawdown": 0.1,
                "average_turnover": 0.01,
            },
            {
                "symbol": "BTCUSDT",
                "policy": "fragile",
                "total_return": 0.1,
                "max_drawdown": 0.5,
                "average_turnover": 0.02,
            },
        ]
    )

    ranking = _rank_holdout(summary)

    assert ranking.iloc[0]["policy"] == "stable"
