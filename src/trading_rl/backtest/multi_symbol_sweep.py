from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.backtest.strategy_sweep import rank_walk_forward
from trading_rl.backtest.walk_forward import run_walk_forward, write_walk_forward_report
from trading_rl.utils.config import load_yaml


def run_multi_symbol_sweep(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policies = list(config["policies"])
    wf = config["walk_forward"]
    all_rows: list[pd.DataFrame] = []

    for symbol, symbol_cfg in config["symbols"].items():
        env_config = dict(config["base_env_config"])
        env_config.update(symbol_cfg)
        symbol_output = output / symbol.lower()
        summary = run_walk_forward(
            env_config,
            policies,
            train_size=int(wf["train_size"]),
            test_size=int(wf["test_size"]),
            step_size=int(wf["step_size"]),
            output_dir=symbol_output,
        )
        summary.insert(0, "symbol", symbol)
        summary.to_csv(symbol_output / "symbol_summary.csv", index=False)
        all_rows.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    ranking = rank_multi_symbol(combined)
    combined.to_csv(output / "combined_walk_forward_summary.csv", index=False)
    ranking.to_csv(output / "combined_strategy_ranking.csv", index=False)
    write_walk_forward_report(combined, output / "combined_walk_forward_summary.html")
    return combined, ranking


def rank_multi_symbol(summary: pd.DataFrame) -> pd.DataFrame:
    ranking = rank_walk_forward(summary)
    per_symbol = summary.groupby("policy")["symbol"].nunique().rename("symbols_tested")
    positive_symbols = (
        summary.groupby(["policy", "symbol"])["total_return"]
        .mean()
        .gt(0.0)
        .groupby("policy")
        .sum()
        .rename("positive_symbols")
    )
    min_symbol_mean = (
        summary.groupby(["policy", "symbol"])["total_return"]
        .mean()
        .groupby("policy")
        .min()
        .rename("min_symbol_mean_return")
    )
    out = ranking.merge(per_symbol, on="policy")
    out = out.merge(positive_symbols, on="policy")
    out = out.merge(min_symbol_mean, on="policy")
    if "active_step_ratio" in summary.columns:
        mean_active = summary.groupby("policy")["active_step_ratio"].mean().rename(
            "mean_active_step_ratio"
        )
        min_symbol_active = (
            summary.groupby(["policy", "symbol"])["active_step_ratio"]
            .mean()
            .groupby("policy")
            .min()
            .rename("min_symbol_active_step_ratio")
        )
        out = out.merge(mean_active, on="policy")
        out = out.merge(min_symbol_active, on="policy")
    out["robust_score"] = out["score"] + 0.05 * out["positive_symbols"] + out[
        "min_symbol_mean_return"
    ].clip(upper=0.0)
    return out.sort_values("robust_score", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strategy sweep across multiple symbols.")
    parser.add_argument("--config", default="configs/sweeps/btc_eth_1h.yaml")
    parser.add_argument("--output-dir", default="artifacts/strategy_sweeps/btc_eth")
    args = parser.parse_args()

    config = load_yaml(args.config)
    _, ranking = run_multi_symbol_sweep(config, args.output_dir)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
