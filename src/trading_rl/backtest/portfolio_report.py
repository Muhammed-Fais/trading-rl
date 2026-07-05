from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_rl.backtest.metrics import PerformanceMetrics, calculate_metrics
from trading_rl.utils.config import load_yaml


def run_portfolio_report(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    history_root = Path(config["history_root"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = str(config.get("policy", "selected_trend_risk"))
    symbols = [str(symbol).upper() for symbol in config["symbols"]]

    histories = _load_symbol_histories(history_root, symbols, policy)
    portfolio = combine_equal_weight_portfolio(histories)
    monthly = monthly_returns(portfolio)
    metrics = calculate_metrics(portfolio)

    portfolio.to_parquet(output_dir / "portfolio_history.parquet", index=False)
    monthly.to_csv(output_dir / "monthly_returns.csv", index=False)
    pd.DataFrame([metrics.as_dict()]).to_csv(output_dir / "portfolio_metrics.csv", index=False)
    report_path = write_portfolio_html_report(
        portfolio,
        monthly,
        metrics,
        output_dir / "portfolio_report.html",
        str(config.get("title", "Portfolio Report")),
    )
    return portfolio, monthly, report_path


def combine_equal_weight_portfolio(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not histories:
        raise ValueError("No histories provided")

    frames: list[pd.DataFrame] = []
    for symbol, history in histories.items():
        required = {
            "timestamp",
            "portfolio_value",
            "benchmark_value",
            "position_fraction",
            "turnover",
        }
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"{symbol} history missing required columns: {sorted(missing)}")
        frame = history[list(required)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.rename(
            columns={
                "portfolio_value": f"{symbol}_portfolio_value",
                "benchmark_value": f"{symbol}_benchmark_value",
                "position_fraction": f"{symbol}_position_fraction",
                "turnover": f"{symbol}_turnover",
            }
        )
        frames.append(frame)

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="timestamp", how="inner")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    portfolio_cols = [f"{symbol}_portfolio_value" for symbol in histories]
    benchmark_cols = [f"{symbol}_benchmark_value" for symbol in histories]
    exposure_cols = [f"{symbol}_position_fraction" for symbol in histories]
    turnover_cols = [f"{symbol}_turnover" for symbol in histories]

    combined["portfolio_value"] = combined[portfolio_cols].sum(axis=1)
    combined["benchmark_value"] = combined[benchmark_cols].sum(axis=1)
    combined["position_fraction"] = combined[exposure_cols].mean(axis=1)
    combined["turnover"] = combined[turnover_cols].mean(axis=1)
    combined["portfolio_return"] = combined["portfolio_value"].pct_change().fillna(0.0)
    combined["benchmark_return"] = combined["benchmark_value"].pct_change().fillna(0.0)
    combined["portfolio_peak"] = combined["portfolio_value"].cummax()
    combined["benchmark_peak"] = combined["benchmark_value"].cummax()
    combined["drawdown"] = combined["portfolio_value"] / combined["portfolio_peak"] - 1.0
    combined["benchmark_drawdown"] = (
        combined["benchmark_value"] / combined["benchmark_peak"] - 1.0
    )
    return combined


def monthly_returns(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.copy()
    out["month"] = pd.to_datetime(out["timestamp"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for month, group in out.groupby("month", sort=True):
        rows.append(
            {
                "month": month,
                "portfolio_return": _period_return(group["portfolio_value"]),
                "benchmark_return": _period_return(group["benchmark_value"]),
                "max_drawdown": float(abs(group["drawdown"].min())),
                "average_exposure": float(group["position_fraction"].mean()),
                "average_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_portfolio_html_report(
    portfolio: pd.DataFrame,
    monthly: pd.DataFrame,
    metrics: PerformanceMetrics,
    output_path: str | Path,
    title: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    symbols = _symbols_from_portfolio(portfolio)
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.30, 0.22, 0.18, 0.18, 0.12],
        subplot_titles=(
            "Combined Equity",
            "Underwater Drawdown",
            "Per-Symbol Equity Contribution",
            "Exposure by Symbol",
            "Monthly Returns",
        ),
    )
    x = portfolio["timestamp"]
    fig.add_trace(go.Scatter(x=x, y=portfolio["portfolio_value"], name="Portfolio"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=portfolio["benchmark_value"], name="Buy & Hold"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=portfolio["drawdown"] * 100.0,
            name="Portfolio DD %",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=portfolio["benchmark_drawdown"] * 100.0, name="Benchmark DD %"),
        row=2,
        col=1,
    )
    for symbol in symbols:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=portfolio[f"{symbol}_portfolio_value"],
                name=f"{symbol} strategy",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=portfolio[f"{symbol}_position_fraction"],
                name=f"{symbol} exposure",
            ),
            row=4,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["portfolio_return"] * 100.0,
            name="Portfolio monthly %",
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["benchmark_return"] * 100.0,
            name="Benchmark monthly %",
        ),
        row=5,
        col=1,
    )
    fig.update_layout(
        title=_title_with_metrics(title, metrics),
        template="plotly_white",
        height=1350,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        barmode="group",
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=3, col=1)
    fig.update_yaxes(title_text="Exposure", row=4, col=1)
    fig.update_yaxes(title_text="Return %", row=5, col=1)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _load_symbol_histories(
    history_root: Path,
    symbols: list[str],
    policy: str,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = history_root / symbol.lower() / f"{policy}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        histories[symbol] = pd.read_parquet(path)
    return histories


def _period_return(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    if first <= 0.0:
        return 0.0
    return last / first - 1.0


def _symbols_from_portfolio(portfolio: pd.DataFrame) -> list[str]:
    suffix = "_portfolio_value"
    return sorted(col.removesuffix(suffix) for col in portfolio.columns if col.endswith(suffix))


def _title_with_metrics(title: str, metrics: PerformanceMetrics) -> str:
    return (
        f"{title}<br><sup>"
        f"Return {metrics.total_return:.2%} | "
        f"Benchmark {metrics.benchmark_return:.2%} | "
        f"Excess {metrics.excess_return:.2%} | "
        f"Max DD {metrics.max_drawdown:.2%} | "
        f"Sharpe {metrics.sharpe:.2f} | "
        f"Sortino {metrics.sortino:.2f}"
        f"</sup>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a combined portfolio HTML report.")
    parser.add_argument("--config", default="configs/reports/core_exposure_crypto3_portfolio.yaml")
    args = parser.parse_args()

    _, _, report_path = run_portfolio_report(load_yaml(args.config))
    print(report_path)


if __name__ == "__main__":
    main()
