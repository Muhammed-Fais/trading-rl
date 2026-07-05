from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_rl.backtest.metrics import PerformanceMetrics, add_performance_columns


def write_html_report(
    history: pd.DataFrame,
    metrics: PerformanceMetrics,
    output_path: str | Path,
    title: str = "Trading RL Evaluation",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched = add_performance_columns(history)
    x = enriched["timestamp"] if "timestamp" in enriched.columns else enriched.index

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.36, 0.24, 0.18, 0.22],
        subplot_titles=(
            "Portfolio vs Benchmark",
            "Underwater Drawdown",
            "Exposure and Actions",
            "Reward Components",
        ),
    )
    fig.add_trace(
        go.Scatter(x=x, y=enriched["portfolio_value"], name="Portfolio", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=enriched["benchmark_value"], name="Buy & Hold", mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=enriched["drawdown"] * 100.0,
            name="Portfolio DD %",
            mode="lines",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=enriched["benchmark_drawdown"] * 100.0,
            name="Benchmark DD %",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    if "position_fraction" in enriched.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=enriched["position_fraction"],
                name="Position Fraction",
                mode="lines",
            ),
            row=3,
            col=1,
        )
    for column, name in (
        ("base_target_fraction", "Base Risk Cap"),
        ("overlay_multiplier", "Overlay Multiplier"),
        ("effective_target_fraction", "Effective Target"),
    ):
        if column in enriched.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=enriched[column],
                    name=name,
                    mode="lines",
                ),
                row=3,
                col=1,
            )
    if "action" in enriched.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=enriched["action"],
                name="Action",
                mode="markers",
                marker={"size": 5},
            ),
            row=3,
            col=1,
        )
    reward_cols = [col for col in enriched.columns if col.startswith("reward_")]
    for col in reward_cols:
        fig.add_trace(go.Scatter(x=x, y=enriched[col], name=col, mode="lines"), row=4, col=1)

    fig.update_layout(
        title=_title_with_metrics(title, metrics),
        template="plotly_white",
        height=1150,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Exposure / Action", row=3, col=1)
    fig.update_yaxes(title_text="Reward", row=4, col=1)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


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
