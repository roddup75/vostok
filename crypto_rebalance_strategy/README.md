# Crypto Rebalance Strategy

This directory contains a standalone workflow for a cross-sectional long/short crypto strategy over a configurable universe of large-cap and liquid crypto assets.

The workflow supports:

- Rebalance every `3D` or `7D`
- Refit the elastic-net monthly while trading more frequently
- Predict forward returns at `1`, `3`, or `5` day horizons
- Use a rolling estimation window in months, for example `6` months
- Start with a broader default universe of 30 crypto assets, while remaining fully configurable
- Construct a default `5` long / `5` short market-neutral book from the forecast ranks
- Build a large technical-indicator feature matrix across multiple windows
- Add shared BTC and ETH benchmark regressors, including lagged daily and weekly returns
- Transform asset-specific features cross-sectionally by date using rank-to-uniform then inverse-normal scaling
- Predict beta-adjusted forward residual returns versus BTC and ETH
- Size positions by inverse volatility and neutralize them to cash, BTC beta, and ETH beta
- Train and compare rolling Elastic Net and XGBoost models with hyperparameter search
- Construct long/short portfolios from top and bottom forecasts
- Produce prediction and portfolio metrics, including information coefficient, for both models
- Generate charts plus a LaTeX report and optional PDF

## Quick Start

1. Install project dependencies from `requirements.txt` in this directory.
2. Run:

```bash
python -m crypto_rebalance_strategy.cli \
  --start-date 2021-09-01 \
  --end-date 2026-09-01 \
  --rebalance-frequency 7D \
  --horizon 5 \
  --train-window-months 6 \
  --model-refit-frequency M \
  --output-dir crypto_rebalance_strategy/output
```

If `yfinance` is unavailable or network access is restricted, provide local OHLCV csv files:

```bash
python -m crypto_rebalance_strategy.cli \
  --data-dir /path/to/csvs \
  --start-date 2021-09-01 \
  --end-date 2026-09-01 \
  --rebalance-frequency 3D \
  --horizon 3
```

CSV files should be named like `BTC-USD.csv` and contain:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

## Output

The run writes into the chosen output directory:

- `predictions.csv`
- `portfolio_returns.csv`
- `summary_metrics.json`
- `plots/`
- `report/report.tex`
- `report/report.pdf` if `pdflatex` is installed
