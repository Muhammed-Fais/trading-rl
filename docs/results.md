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

## Five-Symbol Robustness Check

The first wider universe uses:

- `BTCUSDT`
- `ETHUSDT`
- `BNBUSDT`
- `SOLUSDT`
- `XRPUSDT`

From `artifacts/strategy_sweeps/crypto5/combined_strategy_ranking.csv`:

| Policy | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Mean Turnover | Win Rate | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| buy_and_hold | 18.94% | -75.87% | 32.39% | 81.49% | 0.05% | 53.33% | 5 / 5 |
| trend | 10.94% | -45.06% | 22.60% | 56.53% | 0.87% | 51.67% | 5 / 5 |
| trend_risk_slow | 3.26% | -12.98% | 12.25% | 13.49% | 0.56% | 50.00% | 3 / 5 |

Interpretation: buy-and-hold and trend benefit from the broad crypto bull/bear
sample, but their drawdowns are too large for the risk profile we want. The
trend-risk family keeps drawdown controlled, but the named `trend_risk_slow`
setting is only positive on `3 / 5` symbols.

From `artifacts/strategy_sweeps/trend_risk_grid_fast_crypto5/trend_risk_grid_ranking.csv`:

| Policy | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Mean Turnover | Win Rate | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trend_risk_grid_007 | 3.29% | -12.76% | 12.27% | 13.59% | 0.57% | 48.33% | 4 / 5 |
| trend_risk_grid_005 | 3.06% | -12.97% | 12.39% | 17.59% | 0.58% | 46.67% | 4 / 5 |

The best five-symbol grid setting uses:

- short window: `24`
- long window: `336`
- realized volatility window: `120`
- target hourly volatility: `0.008`
- max portfolio drawdown guard: `0.12`
- trailing stop: `0.15`
- cooldown: `48`
- max exposure: `1.0`

This is the current best candidate for the next research stage. It passes the
five-symbol promotion gates, but it is not live-ready because one of five symbols
still has negative mean walk-forward return.

## Adaptive Trailing Stop Test

The trend-risk policy now supports two trailing-stop modes:

- `percent`: fixed percent drawdown from the highest price since entry
- `atr`: volatility-adaptive stop based on recent true range

From `artifacts/strategy_sweeps/trend_risk_grid_adaptive_crypto5/trend_risk_grid_ranking.csv`,
the focused adaptive grid did not improve the current best result. The top rows
were still fixed `percent` trailing stops:

| Policy | Stop Mode | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Positive Symbols |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| trend_risk_grid_012 | percent | 3.29% | -12.76% | 12.27% | 13.59% | 4 / 5 |
| trend_risk_grid_004 | percent | 3.06% | -12.97% | 12.39% | 17.59% | 4 / 5 |
| trend_risk_grid_002 | atr | 2.38% | -13.09% | 12.07% | 13.85% | 4 / 5 |

Interpretation: ATR-style trailing exits are now available, but they are not the
default candidate because the fixed `15%` trailing stop remains stronger in this
test.

## Calendar Holdout Diagnostic

From `artifacts/holdout/calendar_crypto5/calendar_holdout_ranking.csv`, using
calendar year `2024`:

| Policy | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_and_hold | 125.37% | 48.95% | 42.78% | 46.66% | 5 / 5 |
| trend_risk_crypto5_best | 23.77% | -5.86% | 12.31% | 12.51% | 3 / 5 |
| cash | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 5 |

This is only a diagnostic because previous grids already used data through
`2024-12-31`. The machinery is now in place, but a true final holdout must use
data that was not used during parameter selection.

## Purified Tune/Test Result

The first date-purified workflow selects trend-risk parameters using only
`2021-01-01` through `2023-12-31`, then evaluates the selected policy on
`2024-01-01` through `2024-12-31`.

From `artifacts/tune_test/crypto5_2021_2024/selection/trend_risk_grid_ranking.csv`,
the selected parameters are:

- short window: `24`
- long window: `336`
- realized volatility window: `120`
- target hourly volatility: `0.008`
- max portfolio drawdown guard: `0.12`
- trailing stop: `0.15`
- cooldown: `48`
- max exposure: `1.0`

Selection-period result:

| Policy | Mean Return | Min Fold Return | Mean Drawdown | Max Fold Drawdown | Win Rate | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trend_risk_grid_007 | 4.31% | -12.76% | 12.18% | 13.59% | 57.50% | 5 / 5 |

Holdout result from
`artifacts/tune_test/crypto5_2021_2024/holdout/calendar_holdout_ranking.csv`:

| Policy | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_and_hold | 125.37% | 48.95% | 42.78% | 46.66% | 5 / 5 |
| selected_trend_risk | 23.77% | -5.86% | 12.31% | 12.51% | 3 / 5 |
| cash | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 5 |

Interpretation: the selected parameters are stable enough to be chosen without
looking at `2024`, but the holdout still weakens from `5 / 5` positive symbols
in selection to `3 / 5` in `2024`. That blocks paper trading and points to
regime and per-symbol failure diagnostics as the next research step.

## PPO Status

PPO experiments are useful infrastructure, but current PPO policies are not
profitable candidates. They underperform the trend-risk baseline and still trade
too much.

Current direction:

1. Keep `trend_risk_slow` as the benchmark candidate.
2. Promote `trend_risk_crypto5_best` as the current research candidate.
3. Run wider parameter grids only after the fast grid shows a promising region.
4. Improve robustness across more assets and time windows.
5. Use RL later as a sizing/risk overlay only after the baseline remains stable.

## Overfit Controls

We are intentionally avoiding paper trading until backtests improve.

Current guardrails:

- no shuffled time-series splits
- walk-forward evaluation
- same parameters across BTC and ETH
- separate per-symbol and combined rankings
- turnover, drawdown, and worst-fold metrics in ranking
- no per-symbol retuning
- objective promotion gates before any next-stage work

## Promotion Gates

From `configs/sweeps/promotion_gates.yaml`, a candidate must pass:

- at least `2` symbols tested
- at least `2` positive symbols
- mean return of at least `3%`
- worst fold no worse than `-15%`
- mean drawdown no higher than `15%`
- worst fold drawdown no higher than `18%`
- mean turnover no higher than `1%`
- win rate of at least `50%`
- robust score of at least `10%`

These gates are not a live-trading approval. They are only a research promotion
filter for deciding what deserves wider assets, longer history, and stricter
out-of-sample tests.

Current gate outcomes:

- BTC/ETH fast grid: top candidate passes `9 / 9` gates.
- Five-symbol fast grid: top candidate passes `9 / 9` gates.
- Paper trading remains blocked until stricter out-of-sample tests pass.

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

Evaluate promotion gates:

```bash
make promotion-gates
```

Run five-symbol robustness checks after downloading the extra local data:

```bash
make download-crypto5
make crypto5-sweep
make trend-grid-fast-crypto5
```

Run adaptive trailing-stop grid:

```bash
make trend-grid-adaptive-crypto5
```

Run calendar holdout diagnostic:

```bash
make calendar-holdout
```

Run purified tune/test:

```bash
make tune-test
```
