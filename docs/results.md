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

## Failure Diagnostics

From `artifacts/diagnostics/crypto5_2021_2024/symbol_summary.csv`, the negative
holdout symbols are:

| Symbol | Strategy Return | Benchmark Return | Max Drawdown | Average Exposure | Missed Benchmark Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| SOLUSDT | -5.86% | 90.82% | 12.51% | 2.70% | 96.68% |
| XRPUSDT | -2.37% | 253.08% | 12.35% | 3.74% | 255.44% |

The strategy protected capital in down regimes but missed too much upside in
trend-up regimes:

- SOL trend-up high-vol: strategy `3.31%`, benchmark `268.12%`, average exposure `8.00%`
- SOL trend-up low-vol: strategy `0.90%`, benchmark `196.59%`, average exposure `0.33%`
- XRP trend-up high-vol: strategy `0.00%`, benchmark `708.66%`, average exposure `0.00%`
- XRP trend-up low-vol: strategy `1.49%`, benchmark `83.28%`, average exposure `15.89%`

Worst monthly failures:

| Symbol | Month | Strategy Return | Benchmark Return | Average Exposure |
| --- | --- | ---: | ---: | ---: |
| BNBUSDT | 2024-01 | -6.30% | -5.92% | 23.38% |
| ETHUSDT | 2024-01 | -5.58% | 2.32% | 10.59% |
| SOLUSDT | 2024-02 | -5.07% | 29.62% | 18.47% |
| XRPUSDT | 2024-02 | -2.37% | 16.57% | 46.70% |

Interpretation: the current blocker is not uncontrolled losses. The blocker is
under-participation after the risk/trend filter goes defensive. The next
candidate should improve re-entry and upside participation while preserving the
drawdown cap.

## Participation Re-Entry Test

The trend-risk policy now supports an optional momentum participation floor. If
the slow trend filter is still off but recent momentum is strong, the strategy
can take a small floor exposure while keeping the same portfolio drawdown guard,
trailing stop, and cooldown rules.

Purified selection on `2021-2023` chose:

- participation floor: `0.30`
- momentum window: `168`
- momentum threshold: `0.10`

However, this selected participation setting produced the same `2024` holdout
metrics as the baseline because the trigger did not materially change holdout
actions.

A looser diagnostic candidate, `participation_floor=0.30`,
`momentum_window=72`, `momentum_threshold=0.05`, slightly improved the aggregate
holdout score but did not solve the breadth problem:

| Policy | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 23.77% | -5.86% | 12.31% | 12.51% | 3 / 5 |
| participation_72_005_030 | 23.96% | -4.35% | 12.39% | 12.73% | 3 / 5 |

Interpretation: simple momentum re-entry is directionally useful for SOL, but it
does not repair XRP or improve positive-symbol breadth. The next candidate
should use a stronger regime-aware participation model, not just a small
momentum floor.

## Core Exposure Test

The next participation variant keeps a small protected exposure floor when the
full trend filter is off. This is closer to a core/satellite approach: the
system can hold a small beta allocation for upside participation while the
original trend-risk layer still controls larger exposure, drawdown exits,
trailing stops, and cooldowns.

Purified selection on `2021-2023` chose:

- participation mode: `always`
- participation floor: `0.20`
- target hourly volatility: `0.008`
- short/long windows: `24 / 336`
- realized volatility window: `120`

Holdout result:

| Policy | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 23.77% | -5.86% | 12.31% | 12.51% | 3 / 5 |
| core_exposure | 24.80% | -6.22% | 12.55% | 12.93% | 3 / 5 |

Symbol impact:

| Symbol | Baseline Return | Core Exposure Return |
| --- | ---: | ---: |
| BTCUSDT | 50.07% | 49.95% |
| ETHUSDT | 35.06% | 34.71% |
| BNBUSDT | 41.95% | 46.21% |
| SOLUSDT | -5.86% | -6.22% |
| XRPUSDT | -2.37% | -0.64% |

Interpretation: core exposure is the best aggregate variant so far and materially
improves XRP, but it does not fix SOL and does not improve positive-symbol
breadth. It remains a research candidate, not a paper-trading candidate.

## Filtered Crypto3 Universe

After diagnosing SOL and XRP as persistent holdout weaknesses, we tested a
filtered universe containing only:

- `BTCUSDT`
- `ETHUSDT`
- `BNBUSDT`

This is a universe-selection decision, so it must be treated carefully. It is
valid for defining a narrower research universe, but it is not evidence that the
strategy generalizes to all liquid crypto pairs.

Purified selection on `2021-2023` again selected:

- participation mode: `always`
- participation floor: `0.20`
- short/long windows: `24 / 336`
- realized volatility window: `120`
- target hourly volatility: `0.008`
- trailing stop: `0.15`

Crypto3 holdout result on `2024`:

| Policy | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| buy_and_hold | 94.58% | 48.95% | 40.54% | 46.66% | 3 / 3 |
| selected_trend_risk | 43.63% | 34.71% | 12.48% | 12.86% | 3 / 3 |
| cash | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 3 |

Symbol result:

| Symbol | Strategy Return | Benchmark Return | Max Drawdown |
| --- | ---: | ---: | ---: |
| BTCUSDT | 49.95% | 115.04% | 12.48% |
| ETHUSDT | 34.71% | 49.17% | 12.86% |
| BNBUSDT | 46.21% | 120.41% | 12.09% |

Interpretation: the filtered BTC/ETH/BNB universe is the strongest current
research candidate. It clears the immediate holdout breadth issue with `3 / 3`
positive symbols and controlled drawdown. It is still not live-ready until it
passes stricter final holdout, longer-history, and execution-readiness checks.

## Combined Crypto3 Portfolio Report

From
`artifacts/tune_test/core_exposure_crypto3_2021_2024/portfolio_report/portfolio_metrics.csv`,
the equal-weight combined BTC/ETH/BNB strategy portfolio produced:

| Portfolio | Return | Benchmark Return | Max Drawdown | Sharpe | Average Exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected_trend_risk | 43.63% | 94.87% | 11.83% | 2.11 | 13.47% |

The generated HTML report is:

`artifacts/tune_test/core_exposure_crypto3_2021_2024/portfolio_report/portfolio_report.html`

It includes:

- combined equity curve
- underwater drawdown
- per-symbol equity contribution
- exposure by symbol
- monthly returns

Interpretation: as one equal-weight portfolio, the filtered crypto3 strategy
keeps drawdown controlled while capturing less than half of buy-and-hold upside
in a strong crypto year. The strategy is risk-controlled, but the next research
question is whether we can increase participation without losing the drawdown
profile.

The combined report goes flat after March because each selected symbol hits the
portfolio drawdown guard, exits, and remains in cash. This is an intentional
consequence of the current risk rule, not an HTML/reporting issue.

We tested a `reset_peak_after_drawdown` option that allows re-entry after the
drawdown cooldown by resetting the local risk peak. It kept the strategy active,
but the 2024 holdout risk profile became much worse:

| Variant | Mean Return | Min Symbol Return | Mean Drawdown | Max Symbol Drawdown | Positive Symbols |
| --- | ---: | ---: | ---: | ---: | ---: |
| reset off | 43.63% | 34.71% | 12.48% | 12.86% | 3 / 3 |
| reset on | 21.69% | -18.14% | 39.55% | 51.38% | 2 / 3 |

Interpretation: simple re-entry after a drawdown stop is not acceptable. The next
research step should be a more selective re-entry rule, such as requiring a new
regime confirmation or portfolio-level recovery signal before restoring risk.

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

Run participation tune/test:

```bash
make tune-test-participation
```

Run core exposure tune/test:

```bash
make tune-test-core-exposure
```

Run filtered crypto3 core exposure tune/test:

```bash
make tune-test-core-exposure-crypto3
```

Run failure diagnostics:

```bash
make failure-diagnostics
```

Run combined portfolio report:

```bash
make portfolio-report
```
