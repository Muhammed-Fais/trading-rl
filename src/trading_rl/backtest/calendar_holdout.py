from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.agents.evaluate import (
    build_env_from_dataframe,
    evaluate_policy,
    named_policy,
    trend_risk_policy,
)
from trading_rl.utils.config import load_yaml


def run_calendar_holdout(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    holdout = config["holdout"]
    rows: list[dict[str, Any]] = []

    for symbol, symbol_cfg in config["symbols"].items():
        raw = pd.read_parquet(Path(symbol_cfg["data_path"]).expanduser())
        df = filter_calendar_range(raw, holdout["start"], holdout["end"])
        env_config = dict(config["base_env_config"])
        env_config["split"] = "all"
        env_config["random_start"] = False
        env_config["episode_length"] = None

        for policy_name, policy_cfg in config["policies"].items():
            env = build_env_from_dataframe(df, dict(env_config))
            history, metrics = evaluate_policy(
                env,
                _policy_from_config(policy_cfg),
                seed=7,
            )
            symbol_output = output / symbol.lower()
            symbol_output.mkdir(parents=True, exist_ok=True)
            history.to_parquet(symbol_output / f"{policy_name}.parquet", index=False)
            rows.append({"symbol": symbol, "policy": policy_name, **metrics.as_dict()})

    summary = pd.DataFrame(rows)
    ranking = _rank_holdout(summary)
    summary.to_csv(output / "calendar_holdout_summary.csv", index=False)
    ranking.to_csv(output / "calendar_holdout_ranking.csv", index=False)
    return summary, ranking


def filter_calendar_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if "open_time" not in df.columns:
        raise ValueError("Calendar filtering requires an open_time column")
    timestamps = pd.to_datetime(df["open_time"], utc=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    out = df.loc[(timestamps >= start_ts) & (timestamps < end_ts)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows found for holdout range {start} to {end}")
    return out


def _filter_holdout(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return filter_calendar_range(df, start, end)


def _policy_from_config(policy_cfg: dict[str, Any]):
    kind = policy_cfg["kind"]
    if kind == "named":
        return named_policy(policy_cfg["name"])
    if kind == "trend_risk":
        return trend_risk_policy(**policy_cfg["params"])
    raise ValueError(f"Unknown holdout policy kind: {kind}")


def _rank_holdout(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby("policy").agg(
        mean_return=("total_return", "mean"),
        min_return=("total_return", "min"),
        max_drawdown=("max_drawdown", "mean"),
        max_symbol_drawdown=("max_drawdown", "max"),
        mean_turnover=("average_turnover", "mean"),
        positive_symbols=("total_return", lambda series: int((series > 0.0).sum())),
        symbols_tested=("symbol", "nunique"),
    )
    grouped["score"] = (
        grouped["mean_return"]
        - 0.5 * grouped["max_drawdown"]
        - 0.05 * grouped["mean_turnover"]
        + 0.05 * grouped["positive_symbols"]
    )
    return grouped.sort_values("score", ascending=False).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a calendar holdout evaluation.")
    parser.add_argument("--config", default="configs/sweeps/calendar_holdout_crypto5.yaml")
    parser.add_argument("--output-dir", default="artifacts/holdout/calendar_crypto5")
    args = parser.parse_args()

    _, ranking = run_calendar_holdout(load_yaml(args.config), args.output_dir)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
