import pandas as pd

from trading_rl.backtest.tune_test import _selected_trend_risk_params


def test_selected_trend_risk_params_extracts_top_row_values() -> None:
    ranking = pd.DataFrame(
        [
            {"policy": "best", "short_window": 24, "target_hourly_vol": 0.008},
            {"policy": "other", "short_window": 48, "target_hourly_vol": 0.006},
        ]
    )

    params = _selected_trend_risk_params(ranking, ["short_window", "target_hourly_vol"])

    assert params == {"short_window": 24, "target_hourly_vol": 0.008}
