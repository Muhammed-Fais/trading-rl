from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_rl.backtest.walk_forward import run_walk_forward
from trading_rl.utils.config import load_yaml

DEFAULT_POLICIES = [
    "cash",
    "buy_and_hold",
    "trend",
    "vol_target",
    "trend_risk",
    "trend_risk_fast",
    "trend_risk_slow",
    "trend_risk_defensive",
]


def rank_walk_forward(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("policy").agg(
        mean_return=("total_return", "mean"),
        min_return=("total_return", "min"),
        max_drawdown=("max_drawdown", "mean"),
        max_fold_drawdown=("max_drawdown", "max"),
        mean_turnover=("average_turnover", "mean"),
        win_rate=("total_return", lambda series: float((series > 0.0).mean())),
    )
    grouped["score"] = (
        grouped["mean_return"]
        - 0.50 * grouped["max_drawdown"]
        - 0.10 * grouped["mean_turnover"]
        + 0.10 * grouped["win_rate"]
    )
    return grouped.sort_values("score", ascending=False).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep baseline strategy variants.")
    parser.add_argument("--config", default="configs/train/ppo.yaml")
    parser.add_argument("--output-dir", default="artifacts/strategy_sweeps/btcusdt")
    parser.add_argument("--train-size", type=int, default=365 * 24)
    parser.add_argument("--test-size", type=int, default=90 * 24)
    parser.add_argument("--step-size", type=int, default=90 * 24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output = Path(args.output_dir)
    summary = run_walk_forward(
        cfg["env_config"],
        DEFAULT_POLICIES,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        output_dir=output,
        seed=args.seed,
    )
    ranking = rank_walk_forward(summary)
    ranking.to_csv(output / "strategy_ranking.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
