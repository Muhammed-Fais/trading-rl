from trading_rl.backtest.portfolio_gates import evaluate_portfolio_gates


def test_portfolio_gates_fail_inactive_portfolio() -> None:
    results = evaluate_portfolio_gates(
        {
            "total_return": 0.4,
            "max_drawdown": 0.12,
            "active_month_ratio": 0.25,
            "active_months": 3.0,
            "inactive_months": 9.0,
            "sharpe": 2.0,
        }
    )

    assert not bool(results["eligible_for_next_stage"].iloc[0])
    assert not bool(results.loc[results["gate"] == "active_month_ratio", "passed"].iloc[0])


def test_portfolio_gates_pass_active_risk_controlled_portfolio() -> None:
    results = evaluate_portfolio_gates(
        {
            "total_return": 0.4,
            "max_drawdown": 0.12,
            "active_month_ratio": 0.75,
            "active_months": 9.0,
            "inactive_months": 3.0,
            "sharpe": 2.0,
        }
    )

    assert bool(results["eligible_for_next_stage"].iloc[0])
