import pandas as pd

from trading_rl.backtest.promotion_gates import evaluate_promotion_gates


def test_promotion_gates_marks_only_full_pass_as_eligible() -> None:
    ranking = pd.DataFrame(
        [
            {
                "policy": "stable",
                "symbols_tested": 2,
                "positive_symbols": 2,
                "mean_return": 0.05,
                "min_return": -0.1,
                "max_drawdown": 0.12,
                "max_fold_drawdown": 0.14,
                "mean_turnover": 0.005,
                "win_rate": 0.6,
                "robust_score": 0.12,
            },
            {
                "policy": "fragile",
                "symbols_tested": 2,
                "positive_symbols": 1,
                "mean_return": 0.2,
                "min_return": -0.3,
                "max_drawdown": 0.3,
                "max_fold_drawdown": 0.35,
                "mean_turnover": 0.02,
                "win_rate": 0.4,
                "robust_score": 0.05,
            },
        ]
    )

    results = evaluate_promotion_gates(ranking)

    assert results.iloc[0]["policy"] == "stable"
    assert bool(results.loc[results["policy"] == "stable", "eligible_for_next_stage"].iloc[0])
    assert not bool(
        results.loc[results["policy"] == "fragile", "eligible_for_next_stage"].iloc[0]
    )
