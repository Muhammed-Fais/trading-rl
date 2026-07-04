import pandas as pd

from trading_rl.data.features import add_basic_features
from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv


def _sample_df(rows: int = 120) -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    prices = [100.0 + i * 0.25 for i in range(rows)]
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


def test_env_reset_and_step_contract() -> None:
    env = SpotTradingEnv(
        _sample_df(),
        SpotTradingConfig(lookback=16, episode_length=10, min_trade_notional=0.0),
    )

    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    assert info["portfolio_value"] == 10_000.0

    next_obs, reward, terminated, truncated, next_info = env.step(1)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert next_info["portfolio_value"] > 0
    assert "reward" in next_info["reward_components"]


def test_env_uses_chronological_split_and_random_start() -> None:
    df = _sample_df(1000)
    env = SpotTradingEnv(
        df,
        SpotTradingConfig(
            lookback=16,
            episode_length=24,
            split="test",
            random_start=True,
            min_trade_notional=0.0,
        ),
    )

    assert len(env.df) == 150

    _, first_info = env.reset(seed=1)
    _, second_info = env.reset(seed=2)
    assert first_info["step"] != second_info["step"]
    assert first_info["action_count"] == 6
