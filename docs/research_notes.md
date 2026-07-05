# Research Notes

This project treats trading RL as a research system first, not as a shortcut to
deployment.

## What Other Serious Systems Emphasize

- Backtest overfitting is a primary failure mode. Bailey, Borwein, López de
  Prado, and Zhu propose measuring the probability of backtest overfitting
  instead of trusting the best-looking simulation:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Financial cross-validation needs time-series-specific controls. Purging,
  embargoing, and combinatorial purged cross-validation are designed to reduce
  leakage and winner-picking risk:
  https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/
- RL trading systems need market frictions and risk controls inside the
  environment. FinRL's design explicitly includes transaction costs, liquidity,
  risk aversion, baselines, and reproducible backtesting:
  https://arxiv.org/abs/2011.09607
- Older direct reinforcement trading work also stresses path dependence,
  transaction costs, and risk-adjusted objectives such as differential Sharpe:
  https://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf

## How We Apply This

- Keep walk-forward validation as a baseline research loop.
- Use cross-symbol robustness instead of per-symbol retuning.
- Penalize drawdown, turnover, and worst-fold behavior instead of ranking only
  by PnL.
- Maintain promotion gates before moving to any next stage.
- Add calendar holdout diagnostics, and reserve true final holdout for data not
  used during parameter selection.
- Treat RL as a future sizing/risk overlay unless it beats strong non-RL
  baselines after costs and risk controls.

## Trailing Stop Status

The current best candidate uses a fixed percent trailing stop. The policy tracks
the highest price since entry and exits to cash if price falls more than the
configured trailing threshold, followed by a cooldown.

An ATR-style adaptive trailing stop is now supported, but the first crypto5
adaptive grid did not beat the fixed `15%` trailing stop. We keep the feature,
but do not promote it as the current default.
