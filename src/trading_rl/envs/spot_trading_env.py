from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading_rl.data.features import feature_columns
from trading_rl.envs.accounting import mark_to_market, rebalance_to_fraction
from trading_rl.envs.rewards import log_return_reward


@dataclass(frozen=True)
class SpotTradingConfig:
    initial_cash: float = 10_000.0
    lookback: int = 64
    episode_length: int | None = 2048
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_trade_notional: float = 10.0
    drawdown_penalty: float = 0.05
    turnover_penalty: float = 0.001


class SpotTradingEnv(gym.Env):
    """Single-asset spot trading environment with discrete target allocations."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: SpotTradingConfig | None = None):
        super().__init__()
        self.config = config or SpotTradingConfig()
        self.df = df.reset_index(drop=True)
        if len(self.df) <= self.config.lookback + 2:
            raise ValueError("Dataframe is too short for configured lookback")
        if "close" not in self.df.columns:
            raise ValueError("Dataframe must include close prices")

        self.feature_cols = feature_columns(self.df)
        if not self.feature_cols:
            raise ValueError("Dataframe must include at least one numeric feature column")

        self.prices = self.df["close"].astype(float).to_numpy()
        self.features = self.df[self.feature_cols].astype(np.float32).to_numpy()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "market": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.config.lookback, len(self.feature_cols)),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float32),
            }
        )
        self._rng = np.random.default_rng()
        self._reset_state()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._reset_state(options or {})
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_value = self.portfolio_value
        target_fraction = {0: self.position_fraction, 1: 1.0, 2: 0.0}[int(action)]
        price = float(self.prices[self.current_step])
        state = rebalance_to_fraction(
            cash=self.cash,
            asset_quantity=self.asset_quantity,
            price=price,
            target_fraction=target_fraction,
            fee_rate=self.config.fee_rate,
            slippage_rate=self.config.slippage_rate,
            min_trade_notional=self.config.min_trade_notional,
        )
        self.cash = state.cash
        self.asset_quantity = state.asset_quantity

        self.current_step += 1
        self.steps_elapsed += 1
        next_price = float(self.prices[self.current_step])
        self.portfolio_value = mark_to_market(self.cash, self.asset_quantity, next_price)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.position_fraction = (
            0.0
            if self.portfolio_value <= 0
            else self.asset_quantity * next_price / self.portfolio_value
        )
        drawdown = 0.0 if self.peak_value <= 0 else 1.0 - self.portfolio_value / self.peak_value
        reward = log_return_reward(
            previous_value=previous_value,
            current_value=self.portfolio_value,
            drawdown=drawdown,
            turnover=state.turnover,
            drawdown_penalty=self.config.drawdown_penalty,
            turnover_penalty=self.config.turnover_penalty,
        )

        terminated = self.portfolio_value <= 0
        truncated = self.current_step >= len(self.df) - 1
        if self.config.episode_length is not None:
            truncated = truncated or self.steps_elapsed >= self.config.episode_length
        return self._observation(), reward, terminated, truncated, self._info(state.turnover)

    def render(self) -> None:
        print(self._info())

    def _reset_state(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        max_start = len(self.df) - 2
        if self.config.episode_length is not None:
            max_start = max(self.config.lookback, len(self.df) - self.config.episode_length - 1)
        start = options.get("start_index")
        if start is None:
            start = self.config.lookback
        self.current_step = int(np.clip(start, self.config.lookback, max_start))
        self.steps_elapsed = 0
        self.cash = float(self.config.initial_cash)
        self.asset_quantity = 0.0
        self.portfolio_value = float(self.config.initial_cash)
        self.peak_value = float(self.config.initial_cash)
        self.position_fraction = 0.0

    def _observation(self) -> dict[str, np.ndarray]:
        start = self.current_step - self.config.lookback
        market = self.features[start : self.current_step]
        price = float(self.prices[self.current_step])
        portfolio = np.array(
            [
                self.cash / self.config.initial_cash,
                self.asset_quantity * price / self.config.initial_cash,
                self.portfolio_value / self.config.initial_cash,
                self.position_fraction,
            ],
            dtype=np.float32,
        )
        return {"market": market.astype(np.float32), "portfolio": portfolio}

    def _info(self, turnover: float = 0.0) -> dict[str, float | int]:
        return {
            "step": self.current_step,
            "cash": self.cash,
            "asset_quantity": self.asset_quantity,
            "portfolio_value": self.portfolio_value,
            "position_fraction": self.position_fraction,
            "turnover": turnover,
        }
