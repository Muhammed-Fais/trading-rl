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
uv run mlflow ui --backend-store-uri mlruns
```

Then open `http://127.0.0.1:5000`.

Each run logs parameters, training metrics, checkpoints, evaluation metrics, and
an HTML trading report artifact with:

- portfolio vs buy-and-hold equity
- underwater drawdown
- exposure and actions
- reward component traces

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
