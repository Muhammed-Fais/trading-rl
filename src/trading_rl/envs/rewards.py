from __future__ import annotations

import math


def log_return_reward(
    previous_value: float,
    current_value: float,
    drawdown: float,
    turnover: float,
    drawdown_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
) -> float:
    if previous_value <= 0 or current_value <= 0:
        return -1.0
    reward = math.log(current_value / previous_value)
    reward -= drawdown_penalty * max(drawdown, 0.0)
    reward -= turnover_penalty * max(turnover, 0.0)
    return float(reward)
