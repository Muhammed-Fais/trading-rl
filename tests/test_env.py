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
    assert obs["market"].shape[0] == 16
    assert obs["portfolio"].shape == (4,)
    assert info["portfolio_value"] == 10_000.0

    next_obs, reward, terminated, truncated, next_info = env.step(1)
    assert next_obs["market"].shape == obs["market"].shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert next_info["portfolio_value"] > 0
    assert "reward" in next_info["reward_components"]
