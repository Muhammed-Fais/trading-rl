import pandas as pd

from trading_rl.backtest.multi_symbol_sweep import rank_multi_symbol


def test_rank_multi_symbol_rewards_cross_asset_robustness() -> None:
    summary = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "policy": "robust",
                "total_return": 0.05,
                "max_drawdown": 0.1,
                "average_turnover": 0.01,
            },
            {
                "symbol": "ETHUSDT",
                "policy": "robust",
                "total_return": 0.04,
                "max_drawdown": 0.1,
                "average_turnover": 0.01,
            },
            {
                "symbol": "BTCUSDT",
                "policy": "fragile",
                "total_return": 0.2,
                "max_drawdown": 0.2,
                "average_turnover": 0.02,
            },
            {
                "symbol": "ETHUSDT",
                "policy": "fragile",
                "total_return": -0.1,
                "max_drawdown": 0.2,
                "average_turnover": 0.02,
            },
        ]
    )

    ranking = rank_multi_symbol(summary)

    assert {"symbols_tested", "positive_symbols", "robust_score"}.issubset(ranking.columns)
    assert ranking.loc[ranking["policy"] == "robust", "positive_symbols"].iloc[0] == 2
