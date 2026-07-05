from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading_rl.agents.evaluate import PolicyFn, trend_risk_policy
from trading_rl.envs.spot_trading_env import SpotTradingEnv


class RiskOverlayTradingEnv(gym.Wrapper):
    """Constrain RL actions to scale a deterministic risk policy's allowed exposure."""

    def __init__(
        self,
        env: SpotTradingEnv,
        *,
        base_policy: PolicyFn | None = None,
        base_policy_config: dict[str, Any] | None = None,
    ) -> None:
        if env.config.action_mode != "continuous":
            raise ValueError("RiskOverlayTradingEnv requires a continuous-action base env")
        super().__init__(env)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.base_policy_config = base_policy_config or {}
        self._custom_base_policy = base_policy is not None
        self.base_policy = base_policy or trend_risk_policy(**self.base_policy_config)
        self._last_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] | None = None

    @property
    def df(self) -> pd.DataFrame:
        return self.env.df

    @property
    def prices(self) -> np.ndarray:
        return self.env.prices

    @property
    def current_step(self) -> int:
        return self.env.current_step

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._custom_base_policy:
            self.base_policy = trend_risk_policy(**self.base_policy_config)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action: int | np.ndarray):
        if self._last_obs is None or self._last_info is None:
            raise RuntimeError("Environment must be reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid overlay action: {action}")

        multiplier = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        base_action = self.base_policy(self._last_obs, self._last_info)
        base_target = _continuous_action_value(base_action)
        effective_target = float(np.clip(base_target * multiplier, 0.0, 1.0))

        obs, reward, terminated, truncated, info = self.env.step(
            np.array([effective_target], dtype=np.float32)
        )
        info = dict(info)
        info["base_target_fraction"] = base_target
        info["overlay_multiplier"] = multiplier
        info["effective_target_fraction"] = effective_target
        self._last_obs = obs
        self._last_info = info
        return obs, reward, terminated, truncated, info


def _continuous_action_value(action: int | np.ndarray) -> float:
    value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    return float(np.clip(value, 0.0, 1.0))
