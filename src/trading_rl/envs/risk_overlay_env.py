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
        overlay_reward_config: dict[str, Any] | None = None,
    ) -> None:
        if env.config.action_mode != "continuous":
            raise ValueError("RiskOverlayTradingEnv requires a continuous-action base env")
        super().__init__(env)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.base_policy_config = base_policy_config or {}
        reward_config = overlay_reward_config or {}
        self.upside_participation_weight = float(
            reward_config.get("upside_participation_weight", 0.0)
        )
        self.upside_underuse_penalty = float(reward_config.get("upside_underuse_penalty", 0.0))
        self.min_base_target_for_bonus = float(reward_config.get("min_base_target_for_bonus", 0.05))
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

        previous_price = float(self._last_info["price"])
        obs, reward, terminated, truncated, info = self.env.step(
            np.array([effective_target], dtype=np.float32)
        )
        reward, overlay_components = self._shape_overlay_reward(
            reward=reward,
            previous_price=previous_price,
            current_price=float(info["price"]),
            base_target=base_target,
            multiplier=multiplier,
        )
        info = dict(info)
        reward_components = dict(info.get("reward_components", {}))
        reward_components.update(overlay_components)
        reward_components["reward"] = reward
        info["reward_components"] = reward_components
        info["base_target_fraction"] = base_target
        info["overlay_multiplier"] = multiplier
        info["effective_target_fraction"] = effective_target
        self._last_obs = obs
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def _shape_overlay_reward(
        self,
        *,
        reward: float,
        previous_price: float,
        current_price: float,
        base_target: float,
        multiplier: float,
    ) -> tuple[float, dict[str, float]]:
        if previous_price <= 0.0 or base_target < self.min_base_target_for_bonus:
            return reward, {
                "overlay_upside_participation": 0.0,
                "overlay_upside_underuse": 0.0,
            }
        market_return = current_price / previous_price - 1.0
        upside = max(market_return, 0.0)
        participation = self.upside_participation_weight * upside * base_target * multiplier
        underuse = -self.upside_underuse_penalty * upside * base_target * (1.0 - multiplier)
        shaped_reward = float(reward + participation + underuse)
        return shaped_reward, {
            "overlay_upside_participation": float(participation),
            "overlay_upside_underuse": float(underuse),
        }


def _continuous_action_value(action: int | np.ndarray) -> float:
    value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    return float(np.clip(value, 0.0, 1.0))
