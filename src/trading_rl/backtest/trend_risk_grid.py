from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.agents.evaluate import (
    build_env_from_dataframe,
    evaluate_policy,
    trend_risk_policy,
)
from trading_rl.backtest.calendar_holdout import filter_calendar_range
from trading_rl.backtest.multi_symbol_sweep import rank_multi_symbol
from trading_rl.utils.config import load_yaml


def run_trend_risk_grid(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    max_policies: int | None = None,
    progress_every: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wf = config["walk_forward"]
    rows: list[dict[str, Any]] = []
    symbol_data = _load_symbol_data(config)
    policy_grid = list(_grid_params(config["grid"]))
    active_exposure_threshold = config.get("active_exposure_threshold")
    if max_policies is not None:
        policy_grid = policy_grid[:max_policies]

    total = len(policy_grid) * _fold_count(config, symbol_data)
    completed = 0

    for policy_name, params in policy_grid:
        for symbol, symbol_cfg in config["symbols"].items():
            env_config = dict(config["base_env_config"])
            env_config.update(symbol_cfg)
            env_config.pop("data_path", None)
            env_config["split"] = "all"
            env_config["random_start"] = False
            env_config["episode_length"] = None
            df = symbol_data[symbol]

            probe_env = build_env_from_dataframe(df, dict(env_config))
            fold = 0
            start = max(int(wf["train_size"]), probe_env.config.lookback)
            max_start = len(probe_env.df) - int(wf["test_size"]) - 1
            while start <= max_start:
                env = build_env_from_dataframe(df, dict(env_config))
                history, metrics = evaluate_policy(
                    env,
                    trend_risk_policy(**params),
                    start_index=start,
                    max_steps=int(wf["test_size"]),
                    seed=fold,
                )
                activity = {}
                if active_exposure_threshold is not None:
                    activity["active_step_ratio"] = _active_step_ratio(
                        history,
                        float(active_exposure_threshold),
                    )
                rows.append(
                    {
                        "policy": policy_name,
                        "symbol": symbol,
                        "fold": fold,
                        "start_step": start,
                        "end_step": start + int(wf["test_size"]),
                        **params,
                        **metrics.as_dict(),
                        **activity,
                    }
                )
                fold += 1
                start += int(wf["step_size"])
                completed += 1
                if progress_every > 0 and completed % progress_every == 0:
                    print(f"completed {completed}/{total} evaluations", flush=True)

    summary = pd.DataFrame(rows)
    ranking = rank_multi_symbol(summary)
    param_cols = list(config["grid"].keys())
    params = summary.groupby("policy")[param_cols].first().reset_index()
    ranking = ranking.merge(params, on="policy", how="left")
    summary.to_csv(output / "trend_risk_grid_summary.csv", index=False)
    ranking.to_csv(output / "trend_risk_grid_ranking.csv", index=False)
    return summary, ranking


def _load_symbol_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    data_range = config.get("data_range")
    for symbol, symbol_cfg in config["symbols"].items():
        data_path = Path(symbol_cfg["data_path"]).expanduser().resolve()
        df = pd.read_parquet(data_path)
        if data_range is not None:
            df = filter_calendar_range(df, data_range["start"], data_range["end"])
        data[symbol] = df
    return data


def _fold_count(config: dict[str, Any], symbol_data: dict[str, pd.DataFrame]) -> int:
    wf = config["walk_forward"]
    total = 0
    for symbol, symbol_cfg in config["symbols"].items():
        env_config = dict(config["base_env_config"])
        env_config.update(symbol_cfg)
        env_config.pop("data_path", None)
        env_config["split"] = "all"
        env_config["random_start"] = False
        env_config["episode_length"] = None
        probe_env = build_env_from_dataframe(symbol_data[symbol], env_config)
        start = max(int(wf["train_size"]), probe_env.config.lookback)
        max_start = len(probe_env.df) - int(wf["test_size"]) - 1
        while start <= max_start:
            total += 1
            start += int(wf["step_size"])
    return total


def _grid_params(grid: dict[str, list[Any]]):
    keys = list(grid.keys())
    for index, values in enumerate(itertools.product(*(grid[key] for key in keys))):
        params = dict(zip(keys, values, strict=True))
        name = f"trend_risk_grid_{index:03d}"
        yield name, params


def _active_step_ratio(history: pd.DataFrame, threshold: float) -> float:
    if history.empty or "position_fraction" not in history.columns:
        return 0.0
    active = history["position_fraction"].abs() >= threshold
    return float(active.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a robust trend-risk parameter grid.")
    parser.add_argument("--config", default="configs/sweeps/trend_risk_grid_btc_eth.yaml")
    parser.add_argument("--output-dir", default="artifacts/strategy_sweeps/trend_risk_grid_btc_eth")
    parser.add_argument("--max-policies", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config = load_yaml(args.config)
    _, ranking = run_trend_risk_grid(
        config,
        args.output_dir,
        max_policies=args.max_policies,
        progress_every=args.progress_every,
    )
    print(ranking.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
