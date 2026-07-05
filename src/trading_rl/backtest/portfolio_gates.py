from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.utils.config import load_yaml

DEFAULT_CRITERIA = {
    "min_total_return": 0.20,
    "max_drawdown": 0.15,
    "min_active_month_ratio": 0.60,
    "min_active_months": 8,
    "max_inactive_months": 4,
    "min_sharpe": 1.0,
}


def run_portfolio_gates(config: dict[str, Any]) -> pd.DataFrame:
    metrics = pd.read_csv(Path(config["metrics_path"]).expanduser()).iloc[0].to_dict()
    criteria = {**DEFAULT_CRITERIA, **config.get("criteria", {})}
    results = evaluate_portfolio_gates(metrics, criteria)
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "portfolio_gate_results.csv", index=False)
    _write_html(output_dir / "portfolio_gate_results.html", results)
    return results


def evaluate_portfolio_gates(
    metrics: dict[str, float],
    criteria: dict[str, float] | None = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_CRITERIA, **(criteria or {})}
    rows = [
        _gate("total_return", metrics["total_return"], ">=", thresholds["min_total_return"]),
        _gate("max_drawdown", metrics["max_drawdown"], "<=", thresholds["max_drawdown"]),
        _gate(
            "active_month_ratio",
            metrics["active_month_ratio"],
            ">=",
            thresholds["min_active_month_ratio"],
        ),
        _gate("active_months", metrics["active_months"], ">=", thresholds["min_active_months"]),
        _gate(
            "inactive_months",
            metrics["inactive_months"],
            "<=",
            thresholds["max_inactive_months"],
        ),
        _gate("sharpe", metrics["sharpe"], ">=", thresholds["min_sharpe"]),
    ]
    results = pd.DataFrame(rows)
    results["eligible_for_next_stage"] = bool(results["passed"].all())
    return results


def _gate(name: str, value: float, operator: str, threshold: float) -> dict[str, Any]:
    passed = value >= threshold if operator == ">=" else value <= threshold
    return {
        "gate": name,
        "value": float(value),
        "operator": operator,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def _write_html(output_path: Path, results: pd.DataFrame) -> Path:
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Portfolio Gates</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;}",
        "table{border-collapse:collapse;width:100%;font-size:14px;}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:right;}",
        "th:first-child,td:first-child{text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Portfolio Gates</h1>",
        results.to_html(index=False, escape=True),
        "</body></html>",
    ]
    output_path.write_text("\n".join(html), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate portfolio promotion gates.")
    parser.add_argument(
        "--config",
        default="configs/reports/core_exposure_crypto3_portfolio_gates.yaml",
    )
    args = parser.parse_args()

    results = run_portfolio_gates(load_yaml(args.config))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
