import pandas as pd

from trading_rl.backtest.strategy_sweep import rank_walk_forward


def test_rank_walk_forward_scores_policies() -> None:
    summary = pd.DataFrame(
        [
            {"policy": "a", "total_return": 0.1, "max_drawdown": 0.1, "average_turnover": 0.1},
            {"policy": "a", "total_return": 0.2, "max_drawdown": 0.2, "average_turnover": 0.1},
            {"policy": "b", "total_return": 0.0, "max_drawdown": 0.05, "average_turnover": 0.0},
        ]
    )

    ranking = rank_walk_forward(summary)

    assert ranking.iloc[0]["policy"] == "a"
    assert {"score", "win_rate", "mean_return"}.issubset(ranking.columns)
