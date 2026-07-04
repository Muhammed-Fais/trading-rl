import pandas as pd

from trading_rl.agents.evaluate import evaluate_policy, named_policy
from trading_rl.backtest.report import write_html_report
from trading_rl.data.features import add_basic_features
from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv


def _sample_df(rows: int = 120) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    prices = [100.0 + i * 0.1 for i in range(rows)]
    return add_basic_features(
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


def test_evaluation_history_and_html_report(tmp_path) -> None:
    env = SpotTradingEnv(
        _sample_df(),
        SpotTradingConfig(lookback=16, episode_length=8, min_trade_notional=0.0),
    )

    history, metrics = evaluate_policy(env, named_policy("buy_and_hold"), seed=7)
    report_path = write_html_report(history, metrics, tmp_path / "report.html")

    assert {"portfolio_value", "benchmark_value", "drawdown", "action"}.issubset(history.columns)
    assert metrics.final_value > 0.0
    assert report_path.exists()
    assert "Underwater Drawdown" in report_path.read_text(encoding="utf-8")
