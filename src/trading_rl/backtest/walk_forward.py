from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from trading_rl.agents.evaluate import build_env_from_config, evaluate_policy, named_policy
from trading_rl.utils.config import load_yaml


def run_walk_forward(
    env_config: dict[str, Any],
    policies: list[str],
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    output_dir: str | Path,
    seed: int = 7,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_config = dict(env_config)
    base_config["split"] = "all"
    base_config["random_start"] = False
    base_config["episode_length"] = None
    probe_env = build_env_from_config(dict(base_config))

    rows: list[dict[str, Any]] = []
    fold = 0
    start = max(train_size, probe_env.config.lookback)
    max_start = len(probe_env.df) - test_size - 1
    while start <= max_start:
        for policy_name in policies:
            env = build_env_from_config(dict(base_config))
            history, metrics = evaluate_policy(
                env,
                named_policy(policy_name, seed=seed + fold),
                start_index=start,
                max_steps=test_size,
                seed=seed + fold,
            )
            history.to_parquet(output / f"fold_{fold:03d}_{policy_name}.parquet", index=False)
            rows.append(
                {
                    "fold": fold,
                    "policy": policy_name,
                    "start_step": start,
                    "end_step": start + test_size,
                    **metrics.as_dict(),
                }
            )
        fold += 1
        start += step_size

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "walk_forward_summary.csv", index=False)
    write_walk_forward_report(summary, output / "walk_forward_summary.html")
    return summary


def write_walk_forward_report(summary: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    for policy, group in summary.groupby("policy"):
        fig.add_trace(
            go.Scatter(
                x=group["fold"],
                y=group["total_return"] * 100.0,
                name=f"{policy} return %",
                mode="lines+markers",
            )
        )
    fig.update_layout(
        title=_summary_title(summary),
        template="plotly_white",
        height=700,
        xaxis_title="Walk-forward fold",
        yaxis_title="Fold return %",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _summary_title(summary: pd.DataFrame) -> str:
    grouped = summary.groupby("policy").agg(
        mean_return=("total_return", "mean"),
        mean_drawdown=("max_drawdown", "mean"),
        mean_turnover=("average_turnover", "mean"),
        win_rate=("total_return", lambda series: float((series > 0.0).mean())),
    )
    parts = [
        (
            f"{policy}: return {row.mean_return:.2%}, "
            f"DD {row.mean_drawdown:.2%}, turnover {row.mean_turnover:.2%}, "
            f"win {row.win_rate:.0%}"
        )
        for policy, row in grouped.iterrows()
    ]
    return "Walk-forward baseline evaluation<br><sup>" + " | ".join(parts) + "</sup>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward baseline evaluation.")
    parser.add_argument("--config", default="configs/train/ppo.yaml")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "cash",
            "buy_and_hold",
            "trend",
            "vol_target",
            "mean_reversion",
            "trend_risk",
            "trend_risk_defensive",
        ],
    )
    parser.add_argument("--train-size", type=int, default=365 * 24)
    parser.add_argument("--test-size", type=int, default=90 * 24)
    parser.add_argument("--step-size", type=int, default=90 * 24)
    parser.add_argument("--output-dir", default="artifacts/walk_forward/btcusdt")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    summary = run_walk_forward(
        cfg["env_config"],
        args.policies,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(summary.groupby("policy")["total_return"].agg(["mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
