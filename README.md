# Trading RL

A production-minded reinforcement learning research platform for crypto trading.

The first target is spot BTC/USDT or ETH/USDT on Binance historical klines. The
project is intentionally layered so data, simulation, training, backtesting, risk,
and live/paper execution can evolve independently.

## Current Scope

- Binance spot kline ingestion with local Parquet cache.
- Deterministic feature generation with chronological splits.
- Gymnasium-compatible single-asset spot trading environment.
- Portfolio accounting with fees, slippage, and turnover tracking.
- Pluggable reward functions.
- RLlib PPO training entrypoint scaffold.
- Unit tests for the accounting and environment contract.

## Install

```bash
uv sync --extra dev
```

RLlib is optional because distributed RL dependencies can lag the newest Python
versions:

```bash
uv sync --extra dev --extra rllib
```

Common workflows are also available through `make`:

```bash
make check
make mlflow
make sweep
make multi-sweep
make trend-grid-fast
make promotion-gates
make calendar-holdout
make tune-test
make tune-test-participation
make tune-test-core-exposure
make tune-test-core-exposure-crypto3
make failure-diagnostics
make portfolio-report
```

## Download Data

```bash
uv run trading-rl-download \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2021-01-01 \
  --end 2024-12-31 \
  --output data/raw/binance/BTCUSDT_1h.parquet
```

## Train

```bash
uv run trading-rl-train-rllib --config configs/train/ppo.yaml
```

## Experiment Dashboard and Reports

Training logs to MLflow by default:

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:////Users/fais/Desktop/My_projects/RL/mlflow.db \
  --host 127.0.0.1 \
  --port 5001
```

Then open `http://127.0.0.1:5001`.

Use the absolute SQLite URI so the dashboard reads this repo's `mlflow.db`.
Port `5000` may be occupied by macOS services on some machines.

Each run logs parameters, training metrics, checkpoints, evaluation metrics, and
an HTML trading report artifact with:

- portfolio vs buy-and-hold equity
- underwater drawdown
- exposure and actions
- reward component traces

The default PPO config now trains on the chronological training split with
randomized episode starts and evaluates on the held-out test split. Reports also
log cash, buy-and-hold, and random baselines for comparison.

The environment also supports continuous target-allocation actions with a
no-trade band, which is closer to real portfolio management than repeatedly
choosing coarse buy/sell buckets.

Run baseline walk-forward validation:

```bash
uv run trading-rl-walk-forward \
  --config configs/train/ppo.yaml \
  --policies cash buy_and_hold trend vol_target trend_risk_slow trend_risk_defensive \
  --output-dir artifacts/walk_forward/btcusdt
```

Run a risk-aware strategy sweep:

```bash
uv run trading-rl-strategy-sweep --config configs/train/ppo.yaml
```

In the first BTCUSDT 1h sweep, `trend_risk_slow` beat buy-and-hold on the
risk-aware score by accepting lower mean return for much lower drawdown and
turnover.

Run the same strategy sweep across BTC and ETH without per-symbol retuning:

```bash
uv run trading-rl-multi-symbol-sweep --config configs/sweeps/btc_eth_1h.yaml
```

This is a robustness check, not a guarantee. A candidate is more interesting
when it keeps positive mean walk-forward return across symbols using the same
parameters.

Run a BTC + ETH trend-risk parameter grid:

```bash
make trend-grid-fast
```

The fast grid is the normal iteration loop. Use `make trend-grid` for the wider
search once the fast grid points to a promising neighborhood.

Evaluate objective promotion gates:

```bash
make promotion-gates
```

Promotion gates convert a ranking CSV into pass/fail evidence for the next
research stage. They check cross-symbol coverage, positive-symbol count, mean
return, worst fold, drawdown, turnover, win rate, and robust score.

For wider robustness checks, download the five-asset universe and run the same
process without changing strategy parameters per symbol:

```bash
make download-crypto5
make crypto5-sweep
make trend-grid-fast-crypto5
make trend-grid-adaptive-crypto5
make calendar-holdout
make tune-test
make tune-test-participation
make tune-test-core-exposure
make tune-test-core-exposure-crypto3
make failure-diagnostics
make portfolio-report
```

See `docs/results.md` for the current BTC/ETH sweep results and research
conclusions. See `docs/research_notes.md` for research principles and source
references behind the validation workflow.

You can also generate a baseline report without training:

```bash
uv run trading-rl-evaluate \
  --config configs/train/ppo.yaml \
  --policy buy_and_hold \
  --output artifacts/reports/buy_and_hold.html
```

## Research Guardrails

Financial RL is unusually easy to fool. This project treats these as first-class
risks:

- No shuffled time-series splits.
- No features computed from future candles.
- Transaction costs and slippage enabled by default.
- Evaluation on unseen chronological windows.
- Baselines required before promoting any agent.
- Paper trading and risk gates before live execution.

This is research software, not financial advice.
