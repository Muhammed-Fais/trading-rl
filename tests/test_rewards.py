import pytest

from trading_rl.envs.rewards import RealMarketReward, RewardConfig, RewardInput


def test_real_market_reward_penalizes_churn_and_benchmark_lag() -> None:
    reward_fn = RealMarketReward(
        RewardConfig(
            benchmark_weight=0.5,
            turnover_penalty=0.1,
            drawdown_penalty=0.0,
            downside_penalty=0.0,
            volatility_penalty=0.0,
            reward_clip=1.0,
        )
    )

    result = reward_fn(
        RewardInput(
            previous_value=100.0,
            current_value=101.0,
            previous_price=100.0,
            current_price=104.0,
            drawdown=0.0,
            turnover=0.5,
            position_fraction=1.0,
        )
    )

    assert result.value < 0.0
    assert result.components["benchmark_drag"] < 0.0
    assert result.components["turnover_penalty"] == pytest.approx(-0.05)


def test_real_market_reward_penalizes_drawdown_and_downside_risk() -> None:
    reward_fn = RealMarketReward(
        RewardConfig(
            benchmark_weight=0.0,
            drawdown_penalty=0.5,
            downside_penalty=1.0,
            volatility_penalty=0.0,
            turnover_penalty=0.0,
            reward_clip=1.0,
        )
    )

    result = reward_fn(
        RewardInput(
            previous_value=100.0,
            current_value=95.0,
            previous_price=100.0,
            current_price=100.0,
            drawdown=0.1,
            turnover=0.0,
            position_fraction=0.5,
        )
    )

    assert result.value < -0.05
    assert result.components["drawdown_penalty"] == pytest.approx(-0.05)
    assert result.components["downside_penalty"] < 0.0
