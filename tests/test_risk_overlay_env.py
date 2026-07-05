import numpy as np
import pandas as pd
import pytest

from trading_rl.envs.risk_overlay_env import RiskOverlayTradingEnv
from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv


def _df(rows: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(rows, 1_000.0),
            "log_return": np.concatenate([[0.0], np.diff(np.log(close))]),
        }
    )


def test_risk_overlay_scales_base_policy_target() -> None:
    env = SpotTradingEnv(
        _df(),
        SpotTradingConfig(
            action_mode="continuous",
            lookback=8,
            split="all",
            random_start=False,
            episode_length=8,
        ),
    )
    wrapped = RiskOverlayTradingEnv(
        env,
        base_policy=lambda _obs, _info: np.array([0.8], dtype=np.float32),
    )

    wrapped.reset(seed=7)
    _, _, _, _, info = wrapped.step(np.array([0.5], dtype=np.float32))

    assert info["base_target_fraction"] == pytest.approx(0.8)
    assert info["overlay_multiplier"] == pytest.approx(0.5)
    assert info["effective_target_fraction"] == pytest.approx(0.4)
    assert info["target_fraction"] == pytest.approx(0.4)


def test_risk_overlay_cannot_exceed_base_policy_target() -> None:
    env = SpotTradingEnv(
        _df(),
        SpotTradingConfig(
            action_mode="continuous",
            lookback=8,
            split="all",
            random_start=False,
            episode_length=8,
        ),
    )
    wrapped = RiskOverlayTradingEnv(
        env,
        base_policy=lambda _obs, _info: np.array([0.25], dtype=np.float32),
    )

    wrapped.reset(seed=7)
    _, _, _, _, info = wrapped.step(np.array([1.0], dtype=np.float32))

    assert info["effective_target_fraction"] == pytest.approx(0.25)
