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
    if "active_exposure_threshold" in config:
        tuning_config["active_exposure_threshold"] = config["active_exposure_threshold"]
    _, tuning_ranking = run_trend_risk_grid(
        tuning_config,
        output / "selection",
        progress_every=int(config.get("progress_every", 25)),
    )
    selection_ranking = _apply_selection_objective(
        tuning_ranking,
        config.get("selection_objective"),
    )
    if "selection_score" in selection_ranking.columns:
        selection_ranking.to_csv(output / "selection" / "selection_ranking.csv", index=False)
    selected = _selected_trend_risk_params(selection_ranking, list(config["grid"].keys()))

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
        "selected_policy": str(selection_ranking.iloc[0]["policy"]),
        "selected_params": selected,
    }
    if config.get("selection_objective") is not None:
        selected_artifact["selection_objective"] = config["selection_objective"]
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


def _apply_selection_objective(
    ranking: pd.DataFrame,
    objective: dict[str, Any] | None,
) -> pd.DataFrame:
    if ranking.empty or objective is None:
        return ranking

    out = ranking.copy()
    out = _filter_if_available(out, "positive_symbols", ">=", objective.get("min_positive_symbols"))
    out = _filter_if_available(out, "max_drawdown", "<=", objective.get("max_mean_drawdown"))
    out = _filter_if_available(
        out,
        "max_fold_drawdown",
        "<=",
        objective.get("max_max_fold_drawdown"),
    )
    out = _filter_if_available(
        out,
        "mean_active_step_ratio",
        ">=",
        objective.get("min_mean_active_step_ratio"),
    )
    out = _filter_if_available(
        out,
        "min_symbol_active_step_ratio",
        ">=",
        objective.get("min_symbol_active_step_ratio"),
    )

    base_score = str(objective.get("base_score", "robust_score"))
    if base_score not in out.columns:
        raise ValueError(f"Selection base score column not found: {base_score}")
    activity = (
        out["mean_active_step_ratio"]
        if "mean_active_step_ratio" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    activity_weight = float(objective.get("activity_weight", 0.0))
    out["selection_score"] = out[base_score] + activity_weight * activity
    return out.sort_values("selection_score", ascending=False).reset_index(drop=True)


def _filter_if_available(
    ranking: pd.DataFrame,
    column: str,
    operator: str,
    threshold: Any,
) -> pd.DataFrame:
    if threshold is None or column not in ranking.columns:
        return ranking
    if operator == ">=":
        filtered = ranking[ranking[column] >= float(threshold)]
    elif operator == "<=":
        filtered = ranking[ranking[column] <= float(threshold)]
    else:
        raise ValueError(f"Unsupported selection operator: {operator}")
    return filtered if not filtered.empty else ranking


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
