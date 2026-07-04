import pandas as pd

from trading_rl.backtest.walk_forward import run_walk_forward
from trading_rl.data.features import add_basic_features


def _sample_df(path, rows: int = 500) -> None:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    prices = [100.0 + i * 0.05 for i in range(rows)]
    df = add_basic_features(
        pd.DataFrame(
            {
                "open_time": [base + pd.Timedelta(hours=i) for i in range(rows)],
                "open": prices,
                "high": [p + 1.0 for p in prices],
                "low": [p - 1.0 for p in prices],
                "close": prices,
                "volume": [1000.0 + i for i in range(rows)],
                "close_time": [base + pd.Timedelta(hours=i, minutes=59) for i in range(rows)],
            }
        )
    )
    df.to_parquet(path, index=False)


def test_walk_forward_writes_summary(tmp_path) -> None:
    data_path = tmp_path / "features.parquet"
    _sample_df(data_path)

    summary = run_walk_forward(
        {
            "data_path": str(data_path),
            "lookback": 16,
            "initial_cash": 10_000.0,
            "action_mode": "continuous",
        },
        ["cash", "buy_and_hold"],
        train_size=120,
        test_size=48,
        step_size=48,
        output_dir=tmp_path / "wf",
    )

    assert {"fold", "policy", "total_return", "max_drawdown"}.issubset(summary.columns)
    assert (tmp_path / "wf" / "walk_forward_summary.csv").exists()
    assert (tmp_path / "wf" / "walk_forward_summary.html").exists()
