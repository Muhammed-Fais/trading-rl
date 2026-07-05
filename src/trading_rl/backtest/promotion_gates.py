from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from trading_rl.utils.config import load_yaml

DEFAULT_CRITERIA: dict[str, float] = {
    "min_symbols_tested": 2,
    "min_positive_symbols": 2,
    "min_mean_return": 0.03,
    "min_worst_fold_return": -0.15,
    "max_mean_drawdown": 0.15,
    "max_worst_fold_drawdown": 0.18,
    "max_mean_turnover": 0.01,
    "min_win_rate": 0.5,
    "min_robust_score": 0.1,
}


def evaluate_promotion_gates(
    ranking: pd.DataFrame,
    criteria: dict[str, Any] | None = None,
) -> pd.DataFrame:
    thresholds = {**DEFAULT_CRITERIA, **(criteria or {})}
    results = ranking.copy()

    gate_columns = {
        "gate_symbols_tested": results["symbols_tested"]
        >= float(thresholds["min_symbols_tested"]),
        "gate_positive_symbols": results["positive_symbols"]
        >= float(thresholds["min_positive_symbols"]),
        "gate_mean_return": results["mean_return"] >= float(thresholds["min_mean_return"]),
        "gate_worst_fold_return": results["min_return"]
        >= float(thresholds["min_worst_fold_return"]),
        "gate_mean_drawdown": results["max_drawdown"]
        <= float(thresholds["max_mean_drawdown"]),
        "gate_worst_fold_drawdown": results["max_fold_drawdown"]
        <= float(thresholds["max_worst_fold_drawdown"]),
        "gate_mean_turnover": results["mean_turnover"]
        <= float(thresholds["max_mean_turnover"]),
        "gate_win_rate": results["win_rate"] >= float(thresholds["min_win_rate"]),
        "gate_robust_score": results["robust_score"] >= float(thresholds["min_robust_score"]),
    }
    for column, values in gate_columns.items():
        results[column] = values

    gate_names = list(gate_columns)
    results["gates_passed"] = results[gate_names].sum(axis=1)
    results["gates_total"] = len(gate_names)
    results["eligible_for_next_stage"] = results[gate_names].all(axis=1)
    return results.sort_values(
        ["eligible_for_next_stage", "gates_passed", "robust_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_promotion_gates(config: dict[str, Any]) -> pd.DataFrame:
    ranking_path = Path(config["ranking_path"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(ranking_path)
    results = evaluate_promotion_gates(ranking, config.get("criteria"))
    results.to_csv(output_dir / "promotion_gate_results.csv", index=False)
    _write_html_report(
        results,
        config.get("criteria", {}),
        output_dir / "promotion_gate_results.html",
    )
    return results


def _write_html_report(
    results: pd.DataFrame,
    criteria: dict[str, Any],
    output_path: Path,
) -> Path:
    display_cols = [
        "policy",
        "eligible_for_next_stage",
        "gates_passed",
        "gates_total",
        "mean_return",
        "min_return",
        "max_drawdown",
        "max_fold_drawdown",
        "mean_turnover",
        "win_rate",
        "robust_score",
    ]
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Promotion Gates</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;}",
        "table{border-collapse:collapse;width:100%;font-size:14px;}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:right;}",
        "th:first-child,td:first-child{text-align:left;}",
        "th{background:#f5f5f5;}",
        ".pass{color:#087f5b;font-weight:700}.fail{color:#c92a2a;font-weight:700}",
        "</style></head><body>",
        "<h1>Promotion Gates</h1>",
        "<p>Eligibility requires every configured gate to pass.</p>",
        "<h2>Criteria</h2>",
        pd.DataFrame([criteria]).to_html(index=False, escape=True),
        "<h2>Results</h2>",
        results[display_cols].to_html(index=False, escape=False, formatters=_formatters()),
        "</body></html>",
    ]
    output_path.write_text("\n".join(html), encoding="utf-8")
    return output_path


def _formatters() -> dict[str, Any]:
    percent_cols = {
        "mean_return",
        "min_return",
        "max_drawdown",
        "max_fold_drawdown",
        "mean_turnover",
        "win_rate",
        "robust_score",
    }
    formatters: dict[str, Any] = {column: (lambda value: f"{value:.2%}") for column in percent_cols}
    formatters["eligible_for_next_stage"] = lambda value: (
        "<span class='pass'>PASS</span>" if value else "<span class='fail'>FAIL</span>"
    )
    return formatters


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate strategy promotion gates.")
    parser.add_argument("--config", default="configs/sweeps/promotion_gates.yaml")
    args = parser.parse_args()

    results = run_promotion_gates(load_yaml(args.config))
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
