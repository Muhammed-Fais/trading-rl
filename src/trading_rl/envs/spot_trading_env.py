from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from trading_rl.data.features import feature_columns
from trading_rl.data.splits import chronological_split_indices
from trading_rl.envs.accounting import mark_to_market, rebalance_to_fraction
from trading_rl.envs.rewards import RealMarketReward, RewardConfig, RewardInput


@dataclass(frozen=True)
class SpotTradingConfig:
    initial_cash: float = 10_000.0
    lookback: int = 64
    episode_length: int | None = 2048
    split: str = "train"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    random_start: bool = True
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_trade_notional: float = 10.0
    action_mode: str = "discrete"
    no_trade_band: float = 0.02
    target_position_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    normalize_observations: bool = True
    observation_clip: float = 10.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if isinstance(self.reward, dict):
            reward_config = {key: value for key, value in self.reward.items() if key != "name"}
            object.__setattr__(self, "reward", RewardConfig(**reward_config))
        object.__setattr__(
            self,
            "target_position_fractions",
            tuple(float(value) for value in self.target_position_fractions),
        )


class SpotTradingEnv(gym.Env):
    """Single-asset spot trading environment with target allocation actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: SpotTradingConfig | None = None):
        super().__init__()
        self.config = config or SpotTradingConfig()
        self.df = _select_split(df.reset_index(drop=True), self.config)
        if len(self.df) <= self.config.lookback + 2:
            raise ValueError("Dataframe is too short for configured lookback")
        if "close" not in self.df.columns:
            raise ValueError("Dataframe must include close prices")

        self.feature_cols = feature_columns(self.df)
        if not self.feature_cols:
            raise ValueError("Dataframe must include at least one numeric feature column")

        self.prices = self.df["close"].astype(float).to_numpy()
        raw_features = self.df[self.feature_cols].astype(np.float32).to_numpy()
        self.features = self._normalize_features(raw_features)
        self.reward_fn = RealMarketReward(self.config.reward)
        if self.config.action_mode == "continuous":
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.config.action_mode == "discrete":
            self.action_space = spaces.Discrete(len(self.config.target_position_fractions) + 1)
        else:
            raise ValueError("action_mode must be one of: discrete, continuous")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.lookback * len(self.feature_cols) + 4,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._reset_state()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._reset_state(options or {})
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        previous_value = self.portfolio_value
        previous_price = float(self.prices[self.current_step])
        target_fraction = self._target_fraction(action)
        if abs(target_fraction - self.position_fraction) < self.config.no_trade_band:
            target_fraction = self.position_fraction
        target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
        state = rebalance_to_fraction(
            cash=self.cash,
            asset_quantity=self.asset_quantity,
            price=previous_price,
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
        portfolio_return = (
            -1.0
            if previous_value <= 0 or self.portfolio_value <= 0
            else float(np.log(self.portfolio_value / previous_value))
        )
        self.recent_portfolio_returns.append(portfolio_return)
        rolling_window = self.config.reward.rolling_window
        if len(self.recent_portfolio_returns) > rolling_window:
            self.recent_portfolio_returns = self.recent_portfolio_returns[-rolling_window:]

        reward_result = self.reward_fn(
            RewardInput(
                previous_value=previous_value,
                current_value=self.portfolio_value,
                previous_price=previous_price,
                current_price=next_price,
                drawdown=drawdown,
                turnover=state.turnover,
                position_fraction=self.position_fraction,
                recent_portfolio_returns=tuple(self.recent_portfolio_returns),
            )
        )
        reward = reward_result.value
        self.last_reward_components = reward_result.components

        terminated = self.portfolio_value <= 0
        truncated = self.current_step >= len(self.df) - 1
        if self.config.episode_length is not None:
            truncated = truncated or self.steps_elapsed >= self.config.episode_length
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(
                turnover=state.turnover,
                reward_components=reward_result.components,
                action=action,
                target_fraction=target_fraction,
            ),
        )

    def render(self) -> None:
        print(self._info())

    def _reset_state(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        max_start = len(self.df) - 2
        if self.config.episode_length is not None:
            max_start = max(self.config.lookback, len(self.df) - self.config.episode_length - 1)
        start = options.get("start_index")
        if start is None:
            if self.config.random_start and max_start > self.config.lookback:
                start = self._rng.integers(self.config.lookback, max_start + 1)
            else:
                start = self.config.lookback
        self.current_step = int(np.clip(start, self.config.lookback, max_start))
        self.steps_elapsed = 0
        self.cash = float(self.config.initial_cash)
        self.asset_quantity = 0.0
        self.portfolio_value = float(self.config.initial_cash)
        self.peak_value = float(self.config.initial_cash)
        self.position_fraction = 0.0
        self.recent_portfolio_returns: list[float] = []
        self.last_reward_components: dict[str, float] = {}

    def _observation(self) -> np.ndarray:
        start = self.current_step - self.config.lookback
        market = self.features[start : self.current_step].reshape(-1)
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
        return np.concatenate([market.astype(np.float32), portfolio]).astype(np.float32)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if not self.config.normalize_observations:
            return features
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        normalized = (features - mean) / std
        clipped = np.clip(normalized, -self.config.observation_clip, self.config.observation_clip)
        return clipped.astype(np.float32)

    def _target_fraction(self, action: int | np.ndarray) -> float:
        if self.config.action_mode == "continuous":
            return float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        if int(action) == 0:
            return self.position_fraction
        return float(np.clip(self.config.target_position_fractions[int(action) - 1], 0.0, 1.0))

    def _info(
        self,
        turnover: float = 0.0,
        reward_components: dict[str, float] | None = None,
        action: Any | None = None,
        target_fraction: float | None = None,
    ) -> dict[str, Any]:
        drawdown = 0.0 if self.peak_value <= 0 else 1.0 - self.portfolio_value / self.peak_value
        price = float(self.prices[self.current_step])
        high = float(self.df.loc[self.current_step, "high"]) if "high" in self.df.columns else price
        low = float(self.df.loc[self.current_step, "low"]) if "low" in self.df.columns else price
        return {
            "step": self.current_step,
            "price": price,
            "high": high,
            "low": low,
            "cash": self.cash,
            "asset_quantity": self.asset_quantity,
            "portfolio_value": self.portfolio_value,
            "position_fraction": self.position_fraction,
            "drawdown": drawdown,
            "turnover": turnover,
            "action_mode": self.config.action_mode,
            "action_count": getattr(self.action_space, "n", None),
            "action": _jsonable_action(action),
            "target_fraction": target_fraction,
            "reward_components": reward_components or self.last_reward_components,
        }


def _select_split(df: pd.DataFrame, config: SpotTradingConfig) -> pd.DataFrame:
    split = config.split.lower()
    if split == "all":
        return df
    splits = chronological_split_indices(
        len(df),
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    if split == "train":
        selected = df.iloc[splits.train]
    elif split in {"validation", "val"}:
        selected = df.iloc[splits.validation]
    elif split == "test":
        selected = df.iloc[splits.test]
    else:
        raise ValueError("split must be one of: train, validation, test, all")
    return selected.reset_index(drop=True)


def _jsonable_action(action: Any | None) -> int | float | None:
    if action is None:
        return None
    array = np.asarray(action)
    if array.shape == ():
        return int(array.item()) if float(array.item()).is_integer() else float(array.item())
    return float(array.reshape(-1)[0])
