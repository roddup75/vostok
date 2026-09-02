# vostok

This repository is centered on [`crypto_rebalance_strategy`](./crypto_rebalance_strategy), a research and backtesting project for cross-sectional crypto trading.

## What Is In The Repo

- `crypto_rebalance_strategy/`: the strategy package, CLI, model benchmarks, reporting code, and sample datasets.

## Strategy Overview

The project builds a rolling long/short crypto strategy that:

- forecasts forward returns with competing models including Elastic Net, XGBoost, and a DDPM-based regressor,
- ranks assets cross-sectionally at each rebalance date,
- forms long and short portfolios from the forecast distribution,
- evaluates both prediction metrics and portfolio performance metrics,
- generates LaTeX/PDF reports with tables and charts.

## Entry Point

Run the strategy from:

```bash
python -m crypto_rebalance_strategy.cli --data-dir crypto_rebalance_strategy/test_data_5y
```

For implementation details, configuration, and outputs, see [`crypto_rebalance_strategy/README.md`](./crypto_rebalance_strategy/README.md).
