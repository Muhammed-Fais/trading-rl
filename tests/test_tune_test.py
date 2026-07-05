import pandas as pd

from trading_rl.backtest.tune_test import _apply_selection_objective, _selected_trend_risk_params


def test_selected_trend_risk_params_extracts_top_row_values() -> None:
    ranking = pd.DataFrame(
        [
            {"policy": "best", "short_window": 24, "target_hourly_vol": 0.008},
            {"policy": "other", "short_window": 48, "target_hourly_vol": 0.006},
        ]
    )

    params = _selected_trend_risk_params(ranking, ["short_window", "target_hourly_vol"])

    assert params == {"short_window": 24, "target_hourly_vol": 0.008}


def test_apply_selection_objective_can_prefer_active_candidate() -> None:
    ranking = pd.DataFrame(
        [
            {
                "policy": "inactive",
                "robust_score": 0.20,
                "positive_symbols": 3,
                "max_drawdown": 0.10,
                "max_fold_drawdown": 0.12,
                "mean_active_step_ratio": 0.20,
            },
            {
                "policy": "active",
                "robust_score": 0.18,
                "positive_symbols": 3,
                "max_drawdown": 0.12,
                "max_fold_drawdown": 0.14,
                "mean_active_step_ratio": 0.80,
            },
        ]
    )

    selected = _apply_selection_objective(
        ranking,
        {
            "base_score": "robust_score",
            "activity_weight": 0.05,
            "min_positive_symbols": 3,
            "max_mean_drawdown": 0.18,
            "max_max_fold_drawdown": 0.20,
            "min_mean_active_step_ratio": 0.60,
        },
    )

    assert selected.iloc[0]["policy"] == "active"
