from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    """Reward parameters tuned for realistic trading behavior, not raw PnL."""

    return_weight: float = 1.0
    benchmark_weight: float = 0.25
    drawdown_penalty: float = 0.20
    downside_penalty: float = 0.10
    volatility_penalty: float = 0.02
    turnover_penalty: float = 0.002
    exposure_penalty: float = 0.001
    max_position_fraction: float = 1.0
    reward_clip: float = 0.05
    rolling_window: int = 32


@dataclass(frozen=True)
class RewardInput:
    previous_value: float
    current_value: float
    previous_price: float
    current_price: float
    drawdown: float
    turnover: float
    position_fraction: float
    recent_portfolio_returns: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RewardResult:
    value: float
    components: dict[str, float]


class RealMarketReward:
    """Risk-adjusted reward shaped around tradable behavior."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def __call__(self, inputs: RewardInput) -> RewardResult:
        cfg = self.config
        portfolio_return = _safe_log_return(inputs.previous_value, inputs.current_value)
        market_return = _safe_log_return(inputs.previous_price, inputs.current_price)

        downside = min(portfolio_return, 0.0)
        volatility = _sample_std(inputs.recent_portfolio_returns)
        exposure_excess = max(abs(inputs.position_fraction) - cfg.max_position_fraction, 0.0)

        components = {
            "portfolio_return": cfg.return_weight * portfolio_return,
            "benchmark_drag": -cfg.benchmark_weight * market_return,
            "drawdown_penalty": -cfg.drawdown_penalty * max(inputs.drawdown, 0.0),
            "downside_penalty": -cfg.downside_penalty * downside * downside,
            "volatility_penalty": -cfg.volatility_penalty * volatility,
            "turnover_penalty": -cfg.turnover_penalty * max(inputs.turnover, 0.0),
            "exposure_penalty": -cfg.exposure_penalty * exposure_excess,
        }
        reward = components["portfolio_return"] + components["benchmark_drag"]
        reward += components["drawdown_penalty"]
        reward += components["downside_penalty"]
        reward += components["volatility_penalty"]
        reward += components["turnover_penalty"]
        reward += components["exposure_penalty"]
        reward = float(max(min(reward, cfg.reward_clip), -cfg.reward_clip))
        components["reward"] = reward
        return RewardResult(value=reward, components=components)


def _safe_log_return(previous: float, current: float) -> float:
    if previous <= 0 or current <= 0:
        return -1.0
    return float(math.log(current / previous))


def _sample_std(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))
