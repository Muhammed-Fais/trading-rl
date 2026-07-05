# Current Research Results

This project is still in research mode. The current best candidate is not an RL
policy. It is the `trend_risk_slow` rule-based strategy, which is useful as a
robust baseline and possible future alpha/risk layer.

## Data

- Symbols: `BTCUSDT`, `ETHUSDT`
- Venue data: Binance spot klines
- Interval: `1h`
- Date range: `2021-01-01` to `2024-12-31`
- Fees/slippage in simulator:
  - fee: `0.10%`
  - slippage: `0.05%`

## Best Current Candidate

`trend_risk_slow` uses:

- slow trend filter
- volatility-targeted exposure
- drawdown guard
- trailing stop
- cooldown after risk exits
- long/cash only exposure

## BTC + ETH Walk-Forward Result

From `artifacts/strategy_sweeps/btc_eth/combined_strategy_ranking.csv`:

| Policy | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Mean Turnover | Win Rate | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trend_risk_slow | 4.71% | -12.12% | 12.09% | 13.27% | 0.62% | 58.33% | 2 / 2 |
| buy_and_hold | 11.54% | -69.96% | 27.86% | 74.49% | 0.05% | 50.00% | 2 / 2 |
| trend | 6.12% | -38.86% | 19.27% | 42.70% | 0.86% | 45.83% | 2 / 2 |

Interpretation: buy-and-hold still wins on mean return, but `trend_risk_slow`
has much smaller drawdowns and a far less severe worst fold. It is the current
best risk-adjusted candidate.

## Trend-Risk Parameter Grid

From `artifacts/strategy_sweeps/trend_risk_grid_fast_btc_eth/trend_risk_grid_ranking.csv`:

| Policy | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Mean Turnover | Win Rate | Robust Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trend_risk_grid_015 | 4.71% | -12.12% | 12.09% | 13.27% | 0.62% | 58.33% | 14.43% |
| trend_risk_grid_007 | 5.12% | -12.29% | 12.13% | 13.45% | 0.61% | 54.17% | 14.41% |
| trend_risk_grid_005 | 4.87% | -12.43% | 12.17% | 13.45% | 0.60% | 54.17% | 14.14% |

The top grid member matches the existing `trend_risk_slow` parameters:

- short window: `48`
- long window: `336`
- realized volatility window: `120`
- target hourly volatility: `0.008`
- max portfolio drawdown guard: `0.12`
- trailing stop: `0.15`
- cooldown: `48`
- max exposure: `1.0`

Interpretation: the initial fast grid did not find a materially better setting.
That is useful because the current candidate remains strongest when compared
against nearby alternatives using the same parameters across BTC and ETH.

## PPO Status

PPO experiments are useful infrastructure, but current PPO policies are not
profitable candidates. They underperform the trend-risk baseline and still trade
too much.

Current direction:

1. Keep `trend_risk_slow` as the benchmark candidate.
2. Run wider parameter grids only after the fast grid shows a promising region.
3. Improve robustness across more assets and time windows.
4. Use RL later as a sizing/risk overlay only after the baseline remains stable.

## Overfit Controls

We are intentionally avoiding paper trading until backtests improve.

Current guardrails:

- no shuffled time-series splits
- walk-forward evaluation
- same parameters across BTC and ETH
- separate per-symbol and combined rankings
- turnover, drawdown, and worst-fold metrics in ranking
- no per-symbol retuning

## Reproduce

Start MLflow:

```bash
make mlflow
```

Run tests:

```bash
make check
```

Run BTC-only strategy sweep:

```bash
make sweep
```

Run BTC + ETH robustness sweep:

```bash
make multi-sweep
```

Run BTC + ETH trend-risk parameter grid:

```bash
make trend-grid-fast
```
