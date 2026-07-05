from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_rl.backtest.metrics import PerformanceMetrics, calculate_metrics
from trading_rl.backtest.report import write_html_report
from trading_rl.envs.spot_trading_env import SpotTradingConfig, SpotTradingEnv
from trading_rl.utils.config import load_yaml

PolicyAction = int | np.ndarray
PolicyFn = Callable[[np.ndarray, dict[str, Any]], PolicyAction]


def evaluate_policy(
    env: SpotTradingEnv,
    policy_fn: PolicyFn,
    *,
    start_index: int | None = None,
    max_steps: int | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, PerformanceMetrics]:
    options = {"start_index": start_index} if start_index is not None else None
    obs, info = env.reset(seed=seed, options=options)
    start_price = float(env.prices[env.current_step])
    initial_value = float(info["portfolio_value"])
    rows: list[dict[str, Any]] = [_history_row(env, info, start_price, initial_value, action=0)]

    done = False
    steps = 0
    while not done:
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        row = _history_row(env, info, start_price, initial_value, action=action)
        row["reward"] = reward
        row.update({f"reward_{k}": v for k, v in info.get("reward_components", {}).items()})
        rows.append(row)
        steps += 1
        done = terminated or truncated or (max_steps is not None and steps >= max_steps)

    history = pd.DataFrame(rows)
    return history, calculate_metrics(history)


def named_policy(name: str, seed: int = 7) -> PolicyFn:
    rng = np.random.default_rng(seed)
    if name == "cash":
        return lambda _obs, info: _target_action(info, 0.0)
    if name == "buy_and_hold":
        return lambda _obs, info: _target_action(info, 1.0)
    if name == "random":
        return lambda _obs, info: _target_action(info, float(rng.random()))
    if name == "trend":
        return _trend_policy()
    if name == "vol_target":
        return _vol_target_policy()
    if name == "mean_reversion":
        return _mean_reversion_policy()
    if name == "trend_risk":
        return _trend_risk_policy()
    if name == "trend_risk_fast":
        return _trend_risk_policy(short_window=12, long_window=96, realized_window=48)
    if name == "trend_risk_slow":
        return _trend_risk_policy(short_window=48, long_window=336, realized_window=120)
    if name == "trend_risk_defensive":
        return _trend_risk_policy(
            short_window=24,
            long_window=168,
            realized_window=72,
            target_hourly_vol=0.006,
            max_portfolio_drawdown=0.08,
            trailing_stop=0.10,
            cooldown_steps=72,
            max_exposure=0.75,
        )
    if name == "trend_risk_atr":
        return _trend_risk_policy(
            short_window=24,
            long_window=336,
            realized_window=120,
            trailing_stop_mode="atr",
            atr_window=72,
            atr_multiplier=3.0,
            min_trailing_stop=0.06,
            max_trailing_stop=0.20,
        )
    raise ValueError(f"Unknown policy: {name}")


def trend_risk_policy(
    short_window: int = 24,
    long_window: int = 168,
    realized_window: int = 72,
    target_hourly_vol: float = 0.008,
    max_portfolio_drawdown: float = 0.12,
    trailing_stop: float = 0.15,
    trailing_stop_mode: str = "percent",
    atr_window: int = 72,
    atr_multiplier: float = 3.0,
    min_trailing_stop: float = 0.06,
    max_trailing_stop: float = 0.20,
    participation_floor: float = 0.0,
    participation_mode: str = "momentum",
    momentum_window: int = 72,
    momentum_threshold: float = 0.08,
    cooldown_steps: int = 48,
    reset_peak_after_drawdown: bool = False,
    recovery_reentry_mode: str = "off",
    recovery_exposure: float = 0.2,
    recovery_momentum_window: int = 72,
    recovery_momentum_threshold: float = 0.05,
    recovery_drawdown_buffer: float = 0.03,
    recovery_confirm_window: int = 168,
    recovery_confirm_threshold: float = 0.03,
    regime_window: int = 336,
    regime_momentum_window: int = 168,
    regime_momentum_threshold: float = 0.08,
    regime_max_volatility: float = 0.04,
    max_exposure: float = 1.0,
) -> PolicyFn:
    return _trend_risk_policy(
        short_window=short_window,
        long_window=long_window,
        realized_window=realized_window,
        target_hourly_vol=target_hourly_vol,
        max_portfolio_drawdown=max_portfolio_drawdown,
        trailing_stop=trailing_stop,
        trailing_stop_mode=trailing_stop_mode,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        min_trailing_stop=min_trailing_stop,
        max_trailing_stop=max_trailing_stop,
        participation_floor=participation_floor,
        participation_mode=participation_mode,
        momentum_window=momentum_window,
        momentum_threshold=momentum_threshold,
        cooldown_steps=cooldown_steps,
        reset_peak_after_drawdown=reset_peak_after_drawdown,
        recovery_reentry_mode=recovery_reentry_mode,
        recovery_exposure=recovery_exposure,
        recovery_momentum_window=recovery_momentum_window,
        recovery_momentum_threshold=recovery_momentum_threshold,
        recovery_drawdown_buffer=recovery_drawdown_buffer,
        recovery_confirm_window=recovery_confirm_window,
        recovery_confirm_threshold=recovery_confirm_threshold,
        regime_window=regime_window,
        regime_momentum_window=regime_momentum_window,
        regime_momentum_threshold=regime_momentum_threshold,
        regime_max_volatility=regime_max_volatility,
        max_exposure=max_exposure,
    )


def rllib_policy(algo: Any) -> PolicyFn:
    def _policy(obs: np.ndarray, _info: dict[str, Any]) -> PolicyAction:
        if hasattr(algo, "compute_single_action"):
            return algo.compute_single_action(obs)
        if hasattr(algo, "get_policy"):
            action, _, _ = algo.get_policy().compute_single_action(obs)
            return action
        raise TypeError("Unsupported RLlib algorithm object")

    return _policy


def build_env_from_config(env_config: dict[str, Any]) -> SpotTradingEnv:
    data_path = str(Path(env_config.pop("data_path")).expanduser().resolve())
    df = pd.read_parquet(data_path)
    return build_env_from_dataframe(df, env_config)


def build_env_from_dataframe(df: pd.DataFrame, env_config: dict[str, Any]) -> SpotTradingEnv:
    return SpotTradingEnv(df, SpotTradingConfig(**env_config))


def evaluate_and_report(
    env_config: dict[str, Any],
    policy_fn: PolicyFn,
    output_path: str | Path,
    *,
    title: str,
    start_index: int | None = None,
    max_steps: int | None = None,
    seed: int = 7,
) -> tuple[pd.DataFrame, PerformanceMetrics, Path]:
    env = build_env_from_config(dict(env_config))
    history, metrics = evaluate_policy(
        env,
        policy_fn,
        start_index=start_index,
        max_steps=max_steps,
        seed=seed,
    )
    report_path = write_html_report(history, metrics, output_path, title)
    return history, metrics, report_path


def _history_row(
    env: SpotTradingEnv,
    info: dict[str, Any],
    start_price: float,
    initial_value: float,
    action: PolicyAction,
) -> dict[str, Any]:
    timestamp = None
    if "open_time" in env.df.columns:
        timestamp = env.df.loc[env.current_step, "open_time"]
    price = float(env.prices[env.current_step])
    benchmark_value = initial_value * price / start_price
    return {
        "timestamp": timestamp,
        "step": int(info["step"]),
        "price": price,
        "portfolio_value": float(info["portfolio_value"]),
        "benchmark_value": float(benchmark_value),
        "cash": float(info["cash"]),
        "asset_quantity": float(info["asset_quantity"]),
        "position_fraction": float(info["position_fraction"]),
        "drawdown": float(info["drawdown"]),
        "turnover": float(info["turnover"]),
        "action": _action_value(action),
        "target_fraction": info.get("target_fraction"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline policy and write an HTML report."
    )
    parser.add_argument("--config", default="configs/train/ppo.yaml")
    parser.add_argument(
        "--policy",
        choices=[
            "cash",
            "buy_and_hold",
            "random",
            "trend",
            "vol_target",
            "mean_reversion",
            "trend_risk",
            "trend_risk_fast",
            "trend_risk_slow",
            "trend_risk_defensive",
        ],
        default="buy_and_hold",
    )
    parser.add_argument("--output", default="artifacts/reports/evaluation.html")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    _, metrics, report_path = evaluate_and_report(
        cfg["env_config"],
        named_policy(args.policy, seed=args.seed),
        args.output,
        title=f"{args.policy} evaluation",
        start_index=args.start_index,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print({"report": str(report_path), **metrics.as_dict()})


def _target_action(info: dict[str, Any], target_fraction: float) -> PolicyAction:
    target = float(np.clip(target_fraction, 0.0, 1.0))
    if info.get("action_mode") == "continuous":
        return np.array([target], dtype=np.float32)
    action_count = int(info.get("action_count", 6))
    if target == 0.0:
        return 1
    if target == 1.0:
        return action_count - 1
    fractions = np.linspace(0.0, 1.0, action_count - 1)
    return int(np.argmin(np.abs(fractions - target))) + 1


def _trend_policy(short_window: int = 24, long_window: int = 168) -> PolicyFn:
    prices: list[float] = []

    def _policy(_obs: np.ndarray, info: dict[str, Any]) -> PolicyAction:
        prices.append(float(info["price"]))
        if len(prices) < long_window:
            return _target_action(info, 0.0)
        short_ma = float(np.mean(prices[-short_window:]))
        long_ma = float(np.mean(prices[-long_window:]))
        return _target_action(info, 1.0 if short_ma > long_ma else 0.0)

    return _policy


def _vol_target_policy(
    trend_window: int = 168,
    realized_window: int = 72,
    target_hourly_vol: float = 0.01,
) -> PolicyFn:
    prices: list[float] = []

    def _policy(_obs: np.ndarray, info: dict[str, Any]) -> PolicyAction:
        prices.append(float(info["price"]))
        if len(prices) < max(trend_window, realized_window) + 1:
            return _target_action(info, 0.0)
        log_returns = np.diff(np.log(np.asarray(prices[-realized_window:])))
        realized_vol = float(np.std(log_returns, ddof=1))
        trend_on = prices[-1] > float(np.mean(prices[-trend_window:]))
        if not trend_on or realized_vol <= 0:
            return _target_action(info, 0.0)
        exposure = min(target_hourly_vol / realized_vol, 1.0)
        return _target_action(info, exposure)

    return _policy


def _mean_reversion_policy(
    window: int = 48,
    entry_z: float = -1.0,
    exit_z: float = 0.25,
) -> PolicyFn:
    prices: list[float] = []
    invested = False

    def _policy(_obs: np.ndarray, info: dict[str, Any]) -> PolicyAction:
        nonlocal invested
        prices.append(float(info["price"]))
        if len(prices) < window:
            return _target_action(info, 0.0)
        recent = np.asarray(prices[-window:], dtype=float)
        std = float(np.std(recent, ddof=1))
        zscore = 0.0 if std == 0.0 else float((recent[-1] - np.mean(recent)) / std)
        if zscore < entry_z:
            invested = True
        elif zscore > exit_z:
            invested = False
        return _target_action(info, 1.0 if invested else 0.0)

    return _policy


def _trend_risk_policy(
    short_window: int = 24,
    long_window: int = 168,
    realized_window: int = 72,
    target_hourly_vol: float = 0.008,
    max_portfolio_drawdown: float = 0.12,
    trailing_stop: float = 0.15,
    trailing_stop_mode: str = "percent",
    atr_window: int = 72,
    atr_multiplier: float = 3.0,
    min_trailing_stop: float = 0.06,
    max_trailing_stop: float = 0.20,
    participation_floor: float = 0.0,
    participation_mode: str = "momentum",
    momentum_window: int = 72,
    momentum_threshold: float = 0.08,
    cooldown_steps: int = 48,
    reset_peak_after_drawdown: bool = False,
    recovery_reentry_mode: str = "off",
    recovery_exposure: float = 0.2,
    recovery_momentum_window: int = 72,
    recovery_momentum_threshold: float = 0.05,
    recovery_drawdown_buffer: float = 0.03,
    recovery_confirm_window: int = 168,
    recovery_confirm_threshold: float = 0.03,
    regime_window: int = 336,
    regime_momentum_window: int = 168,
    regime_momentum_threshold: float = 0.08,
    regime_max_volatility: float = 0.04,
    max_exposure: float = 1.0,
) -> PolicyFn:
    prices: list[float] = []
    true_ranges: list[float] = []
    portfolio_peak = 0.0
    entry_peak = 0.0
    cooldown = 0
    cooldown_reason = ""
    risk_off = False
    recovery_mode = False
    invested = False
    if trailing_stop_mode not in {"percent", "atr"}:
        raise ValueError("trailing_stop_mode must be one of: percent, atr")
    if participation_mode not in {"momentum", "always"}:
        raise ValueError("participation_mode must be one of: momentum, always")
    if recovery_reentry_mode not in {
        "off",
        "momentum",
        "confirmed_reset",
        "regime_reset",
    }:
        raise ValueError(
            "recovery_reentry_mode must be one of: off, momentum, "
            "confirmed_reset, regime_reset"
        )

    def _policy(_obs: np.ndarray, info: dict[str, Any]) -> PolicyAction:
        nonlocal portfolio_peak, entry_peak, cooldown, cooldown_reason, invested
        nonlocal recovery_mode, risk_off
        price = float(info["price"])
        previous_price = prices[-1] if prices else price
        high = float(info.get("high", price))
        low = float(info.get("low", price))
        true_range = max(high - low, abs(high - previous_price), abs(low - previous_price))
        true_ranges.append(0.0 if price <= 0.0 else true_range / price)
        prices.append(price)
        portfolio_value = float(info["portfolio_value"])
        portfolio_peak = max(portfolio_peak, portfolio_value)

        if cooldown > 0:
            cooldown -= 1
            if (
                cooldown == 0
                and cooldown_reason == "portfolio_drawdown"
                and reset_peak_after_drawdown
            ):
                portfolio_peak = portfolio_value
                entry_peak = price
                risk_off = False
                recovery_mode = False
                cooldown_reason = ""
            invested = False
            return _target_action(info, 0.0)

        if len(prices) < max(long_window, realized_window) + 1:
            return _target_action(info, 0.0)

        short_ma = float(np.mean(prices[-short_window:]))
        long_ma = float(np.mean(prices[-long_window:]))
        trend_on = short_ma > long_ma and price > long_ma
        if risk_off and _confirmed_recovery_reset_allowed(
            prices=prices,
            short_ma=short_ma,
            long_ma=long_ma,
            mode=recovery_reentry_mode,
            confirm_window=recovery_confirm_window,
            confirm_threshold=recovery_confirm_threshold,
        ):
            portfolio_peak = portfolio_value
            entry_peak = price
            risk_off = False
            recovery_mode = False
            cooldown_reason = ""
        if risk_off and _regime_recovery_reset_allowed(
            prices=prices,
            short_ma=short_ma,
            long_ma=long_ma,
            true_ranges=true_ranges,
            mode=recovery_reentry_mode,
            regime_window=regime_window,
            momentum_window=regime_momentum_window,
            momentum_threshold=regime_momentum_threshold,
            max_volatility=regime_max_volatility,
        ):
            portfolio_peak = portfolio_value
            entry_peak = price
            risk_off = False
            recovery_mode = False
            cooldown_reason = ""

        recovery_exposure_value = _recovery_reentry_exposure(
            prices=prices,
            short_ma=short_ma,
            long_ma=long_ma,
            mode=recovery_reentry_mode,
            exposure=recovery_exposure,
            momentum_window=recovery_momentum_window,
            momentum_threshold=recovery_momentum_threshold,
            max_exposure=max_exposure,
        )
        participation_exposure = 0.0
        if not trend_on:
            participation_exposure = _momentum_participation_exposure(
                prices=prices,
                short_ma=short_ma,
                participation_floor=participation_floor,
                participation_mode=participation_mode,
                momentum_window=momentum_window,
                momentum_threshold=momentum_threshold,
                max_exposure=max_exposure,
            )

        portfolio_drawdown = (
            0.0 if portfolio_peak <= 0.0 else 1.0 - portfolio_value / portfolio_peak
        )
        recovery_allowed = risk_off and recovery_exposure_value > 0.0
        if portfolio_drawdown > max_portfolio_drawdown:
            if not recovery_allowed:
                cooldown = cooldown_steps
                cooldown_reason = "portfolio_drawdown"
                risk_off = True
                recovery_mode = False
                invested = False
                return _target_action(info, 0.0)
            recovery_mode = True
            if portfolio_drawdown > max_portfolio_drawdown + recovery_drawdown_buffer:
                cooldown = cooldown_steps
                cooldown_reason = "portfolio_drawdown"
                recovery_mode = False
                invested = False
                return _target_action(info, 0.0)
        elif risk_off:
            risk_off = False
            recovery_mode = False

        if not trend_on and participation_exposure <= 0.0 and recovery_exposure_value <= 0.0:
            invested = False
            return _target_action(info, 0.0)

        if recovery_mode:
            exposure = recovery_exposure_value
        else:
            if not invested:
                invested = True
                entry_peak = price
            entry_peak = max(entry_peak, price)
            price_drawdown = 0.0 if entry_peak <= 0.0 else 1.0 - price / entry_peak
            active_trailing_stop = _active_trailing_stop(
                mode=trailing_stop_mode,
                trailing_stop=trailing_stop,
                true_ranges=true_ranges,
                atr_window=atr_window,
                atr_multiplier=atr_multiplier,
                min_trailing_stop=min_trailing_stop,
                max_trailing_stop=max_trailing_stop,
            )
            if price_drawdown > active_trailing_stop:
                cooldown = cooldown_steps
                cooldown_reason = "trailing_stop"
                invested = False
                return _target_action(info, 0.0)

            log_returns = np.diff(np.log(np.asarray(prices[-realized_window:])))
            realized_vol = float(np.std(log_returns, ddof=1))
            if realized_vol <= 0.0:
                exposure = max_exposure
            else:
                exposure = min(target_hourly_vol / realized_vol, max_exposure)
            if participation_exposure > 0.0:
                exposure = participation_exposure
        return _target_action(info, exposure)

    return _policy


def _momentum_participation_exposure(
    *,
    prices: list[float],
    short_ma: float,
    participation_floor: float,
    participation_mode: str,
    momentum_window: int,
    momentum_threshold: float,
    max_exposure: float,
) -> float:
    if participation_floor <= 0.0:
        return 0.0
    if participation_mode == "always":
        return float(np.clip(participation_floor, 0.0, max_exposure))
    if len(prices) <= momentum_window:
        return 0.0
    price = prices[-1]
    base_price = prices[-momentum_window - 1]
    if base_price <= 0.0:
        return 0.0
    momentum = price / base_price - 1.0
    if momentum >= momentum_threshold and price > short_ma:
        return float(np.clip(participation_floor, 0.0, max_exposure))
    return 0.0


def _recovery_reentry_exposure(
    *,
    prices: list[float],
    short_ma: float,
    long_ma: float,
    mode: str,
    exposure: float,
    momentum_window: int,
    momentum_threshold: float,
    max_exposure: float,
) -> float:
    if mode != "momentum" or exposure <= 0.0 or len(prices) <= momentum_window:
        return 0.0
    price = prices[-1]
    base_price = prices[-momentum_window - 1]
    if base_price <= 0.0:
        return 0.0
    momentum = price / base_price - 1.0
    trend_confirmed = price > short_ma > long_ma
    if momentum >= momentum_threshold and trend_confirmed:
        return float(np.clip(exposure, 0.0, max_exposure))
    return 0.0


def _confirmed_recovery_reset_allowed(
    *,
    prices: list[float],
    short_ma: float,
    long_ma: float,
    mode: str,
    confirm_window: int,
    confirm_threshold: float,
) -> bool:
    if mode != "confirmed_reset" or confirm_window <= 0 or len(prices) < confirm_window:
        return False
    price = prices[-1]
    recent_high = max(prices[-confirm_window:])
    if recent_high <= 0.0:
        return False
    near_recovery_high = price >= recent_high * (1.0 - confirm_threshold)
    trend_confirmed = price > short_ma > long_ma
    return bool(near_recovery_high and trend_confirmed)


def _regime_recovery_reset_allowed(
    *,
    prices: list[float],
    short_ma: float,
    long_ma: float,
    true_ranges: list[float],
    mode: str,
    regime_window: int,
    momentum_window: int,
    momentum_threshold: float,
    max_volatility: float,
) -> bool:
    if mode != "regime_reset":
        return False
    lookback = max(regime_window, momentum_window)
    if lookback <= 0 or len(prices) <= lookback:
        return False
    price = prices[-1]
    base_price = prices[-momentum_window - 1]
    if base_price <= 0.0:
        return False
    momentum = price / base_price - 1.0
    regime_ma = float(np.mean(prices[-regime_window:]))
    recent_ranges = true_ranges[-regime_window:]
    volatility = 0.0 if not recent_ranges else float(np.mean(recent_ranges))
    volatility_ok = max_volatility <= 0.0 or volatility <= max_volatility
    trend_confirmed = price > short_ma > long_ma and price > regime_ma
    return bool(momentum >= momentum_threshold and trend_confirmed and volatility_ok)


def _active_trailing_stop(
    *,
    mode: str,
    trailing_stop: float,
    true_ranges: list[float],
    atr_window: int,
    atr_multiplier: float,
    min_trailing_stop: float,
    max_trailing_stop: float,
) -> float:
    if mode == "percent":
        return trailing_stop
    window = true_ranges[-atr_window:]
    atr = 0.0 if not window else float(np.mean(window))
    adaptive_stop = atr_multiplier * atr
    return float(np.clip(adaptive_stop, min_trailing_stop, max_trailing_stop))


def _action_value(action: PolicyAction) -> float:
    array = np.asarray(action)
    return float(array.reshape(-1)[0])


if __name__ == "__main__":
    main()
