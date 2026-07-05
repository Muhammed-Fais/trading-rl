from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.backtest.calendar_holdout import run_calendar_holdout
from trading_rl.backtest.trend_risk_grid import run_trend_risk_grid
from trading_rl.utils.config import load_yaml


def run_tune_test(
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tuning_config = {
        "symbols": config["symbols"],
        "base_env_config": config["base_env_config"],
        "walk_forward": config["walk_forward"],
        "grid": config["grid"],
        "data_range": config["selection_period"],
    }
    _, tuning_ranking = run_trend_risk_grid(
        tuning_config,
        output / "selection",
        progress_every=int(config.get("progress_every", 25)),
    )
    selected = _selected_trend_risk_params(tuning_ranking, list(config["grid"].keys()))

    holdout_config = {
        "symbols": config["symbols"],
        "base_env_config": config["base_env_config"],
        "holdout": config["holdout"],
        "policies": {
            "cash": {"kind": "named", "name": "cash"},
            "buy_and_hold": {"kind": "named", "name": "buy_and_hold"},
            "selected_trend_risk": {"kind": "trend_risk", "params": selected},
        },
    }
    holdout_summary, holdout_ranking = run_calendar_holdout(
        holdout_config,
        output / "holdout",
    )
    selected_artifact = {
        "selection_period": config["selection_period"],
        "holdout": config["holdout"],
        "selected_policy": str(tuning_ranking.iloc[0]["policy"]),
        "selected_params": selected,
    }
    (output / "selected_policy.json").write_text(
        json.dumps(selected_artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return holdout_summary, holdout_ranking, selected_artifact


def _selected_trend_risk_params(
    ranking: pd.DataFrame,
    param_keys: list[str],
) -> dict[str, Any]:
    if ranking.empty:
        raise ValueError("Cannot select trend-risk params from an empty ranking")
    row = ranking.iloc[0]
    return {key: _jsonable(row[key]) for key in param_keys}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune on one calendar range and test on holdout.")
    parser.add_argument("--config", default="configs/sweeps/tune_test_crypto5_2021_2024.yaml")
    parser.add_argument("--output-dir", default="artifacts/tune_test/crypto5_2021_2024")
    args = parser.parse_args()

    _, holdout_ranking, selected = run_tune_test(load_yaml(args.config), args.output_dir)
    print(json.dumps(selected, indent=2, sort_keys=True))
    print(holdout_ranking.to_string(index=False))


if __name__ == "__main__":
    main()
