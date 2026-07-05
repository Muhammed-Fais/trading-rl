from trading_rl.backtest.trend_risk_grid import _grid_params


def test_grid_params_names_and_values() -> None:
    params = list(_grid_params({"a": [1, 2], "b": [3]}))

    assert params[0][0] == "trend_risk_grid_000"
    assert params[0][1] == {"a": 1, "b": 3}
    assert params[1][0] == "trend_risk_grid_001"
    assert params[1][1] == {"a": 2, "b": 3}
