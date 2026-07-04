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

PolicyFn = Callable[[np.ndarray, dict[str, Any]], int]


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
        action = int(policy_fn(obs, info))
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
        return lambda _obs, _info: 1
    if name == "buy_and_hold":
        return lambda _obs, info: int(info.get("action_count", 5)) - 1
    if name == "random":
        return lambda _obs, info: int(rng.integers(0, int(info.get("action_count", 5))))
    raise ValueError(f"Unknown policy: {name}")


def rllib_policy(algo: Any) -> PolicyFn:
    def _policy(obs: np.ndarray, _info: dict[str, Any]) -> int:
        if hasattr(algo, "compute_single_action"):
            return int(algo.compute_single_action(obs))
        if hasattr(algo, "get_policy"):
            action, _, _ = algo.get_policy().compute_single_action(obs)
            return int(action)
        raise TypeError("Unsupported RLlib algorithm object")

    return _policy


def build_env_from_config(env_config: dict[str, Any]) -> SpotTradingEnv:
    data_path = str(Path(env_config.pop("data_path")).expanduser().resolve())
    df = pd.read_parquet(data_path)
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
    action: int,
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
        "action": int(action),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline policy and write an HTML report."
    )
    parser.add_argument("--config", default="configs/train/ppo.yaml")
    parser.add_argument(
        "--policy",
        choices=["cash", "buy_and_hold", "random"],
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


if __name__ == "__main__":
    main()
