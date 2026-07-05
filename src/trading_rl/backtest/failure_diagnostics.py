from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_rl.backtest.metrics import add_performance_columns
from trading_rl.utils.config import load_yaml


def run_failure_diagnostics(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    history_root = Path(config["history_root"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = str(config.get("policy", "selected_trend_risk"))
    benchmark_policy = str(config.get("benchmark_policy", "buy_and_hold"))
    trend_window = int(config.get("trend_window", 168))
    vol_window = int(config.get("vol_window", 168))

    histories = _load_histories(history_root, policy, benchmark_policy)
    enriched = [
        _prepare_history(symbol, policy_name, history, trend_window, vol_window)
        for symbol, policy_name, history in histories
    ]
    combined = pd.concat(enriched, ignore_index=True)
    policy_rows = combined[combined["policy"] == policy].copy()

    symbol_summary = _symbol_summary(policy_rows)
    regime_summary = _regime_summary(policy_rows)
    monthly_summary = _monthly_summary(policy_rows)
    failure_events = _failure_events(policy_rows)

    combined.to_parquet(output_dir / "diagnostic_history.parquet", index=False)
    symbol_summary.to_csv(output_dir / "symbol_summary.csv", index=False)
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)
    monthly_summary.to_csv(output_dir / "monthly_summary.csv", index=False)
    failure_events.to_csv(output_dir / "failure_events.csv", index=False)
    _write_html_report(
        output_dir / "failure_diagnostics.html",
        symbol_summary,
        regime_summary,
        monthly_summary,
        failure_events,
    )
    return {
        "symbol_summary": symbol_summary,
        "regime_summary": regime_summary,
        "monthly_summary": monthly_summary,
        "failure_events": failure_events,
    }


def _load_histories(
    history_root: Path,
    policy: str,
    benchmark_policy: str,
) -> list[tuple[str, str, pd.DataFrame]]:
    histories: list[tuple[str, str, pd.DataFrame]] = []
    for symbol_dir in sorted(path for path in history_root.iterdir() if path.is_dir()):
        symbol = symbol_dir.name.upper()
        for policy_name in [policy, benchmark_policy]:
            path = symbol_dir / f"{policy_name}.parquet"
            if path.exists():
                histories.append((symbol, policy_name, pd.read_parquet(path)))
    if not histories:
        raise ValueError(f"No policy histories found under {history_root}")
    return histories


def _prepare_history(
    symbol: str,
    policy: str,
    history: pd.DataFrame,
    trend_window: int,
    vol_window: int,
) -> pd.DataFrame:
    out = add_performance_columns(history)
    out.insert(0, "policy", policy)
    out.insert(0, "symbol", symbol)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["benchmark_momentum"] = out["benchmark_value"].pct_change(trend_window)
    out["benchmark_volatility"] = out["benchmark_return"].rolling(vol_window, min_periods=24).std()
    vol_median = float(out["benchmark_volatility"].median())
    out["trend_regime"] = np.where(out["benchmark_momentum"] >= 0.0, "trend_up", "trend_down")
    out["vol_regime"] = np.where(
        out["benchmark_volatility"] >= vol_median,
        "high_vol",
        "low_vol",
    )
    out["market_regime"] = out["trend_regime"] + "_" + out["vol_regime"]
    out["month"] = out["timestamp"].dt.strftime("%Y-%m")
    return out


def _symbol_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("symbol", sort=True)[_SUMMARY_COLUMNS]
    summary = grouped.apply(_return_summary).reset_index()
    summary["missed_benchmark_return"] = summary["benchmark_return"] - summary["portfolio_return"]
    return summary.sort_values("portfolio_return", ascending=True).reset_index(drop=True)


def _regime_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["symbol", "market_regime"], sort=True)[_SUMMARY_COLUMNS]
    return grouped.apply(_return_summary).reset_index()


def _monthly_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["symbol", "month"], sort=True)[_SUMMARY_COLUMNS]
    out = grouped.apply(_return_summary).reset_index()
    return out.sort_values(["portfolio_return", "symbol"], ascending=[True, True])


def _return_summary(rows: pd.DataFrame) -> pd.Series:
    portfolio_return = _compound_return(rows["portfolio_return"])
    benchmark_return = _compound_return(rows["benchmark_return"])
    return pd.Series(
        {
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "excess_return": portfolio_return - benchmark_return,
            "max_drawdown": float(abs(rows["drawdown"].min())),
            "average_exposure": float(rows["position_fraction"].abs().mean()),
            "average_turnover": float(rows["turnover"].mean()),
            "bars": int(len(rows)),
        }
    )


_SUMMARY_COLUMNS = [
    "portfolio_return",
    "benchmark_return",
    "drawdown",
    "position_fraction",
    "turnover",
]


def _compound_return(returns: pd.Series) -> float:
    clean = returns.fillna(0.0).to_numpy(dtype=float)
    return float(np.prod(1.0 + clean) - 1.0)


def _failure_events(rows: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    monthly = _monthly_summary(rows)
    failures = monthly[monthly["portfolio_return"] < 0.0].copy()
    failures["missed_benchmark_return"] = (
        failures["benchmark_return"] - failures["portfolio_return"]
    )
    return failures.sort_values("portfolio_return").head(top_n).reset_index(drop=True)


def _write_html_report(
    output_path: Path,
    symbol_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    failure_events: pd.DataFrame,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Failure Diagnostics</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;}",
        "table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:28px;}",
        "th,td{border:1px solid #ddd;padding:7px;text-align:right;}",
        "th:first-child,td:first-child{text-align:left;}",
        "th{background:#f5f5f5;}",
        "</style></head><body>",
        "<h1>Failure Diagnostics</h1>",
        "<h2>Symbol Summary</h2>",
        symbol_summary.to_html(index=False, escape=True, formatters=_formatters()),
        "<h2>Regime Summary</h2>",
        regime_summary.to_html(index=False, escape=True, formatters=_formatters()),
        "<h2>Worst Monthly Failures</h2>",
        failure_events.to_html(index=False, escape=True, formatters=_formatters()),
        "<h2>Monthly Summary</h2>",
        monthly_summary.to_html(index=False, escape=True, formatters=_formatters()),
        "</body></html>",
    ]
    output_path.write_text("\n".join(html), encoding="utf-8")
    return output_path


def _formatters() -> dict[str, Any]:
    percent_cols = {
        "portfolio_return",
        "benchmark_return",
        "excess_return",
        "max_drawdown",
        "average_exposure",
        "average_turnover",
        "missed_benchmark_return",
    }
    return {column: (lambda value: f"{value:.2%}") for column in percent_cols}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose holdout strategy failures.")
    parser.add_argument("--config", default="configs/sweeps/failure_diagnostics_crypto5.yaml")
    args = parser.parse_args()

    results = run_failure_diagnostics(load_yaml(args.config))
    print(results["symbol_summary"].to_string(index=False))
    print(results["failure_events"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
