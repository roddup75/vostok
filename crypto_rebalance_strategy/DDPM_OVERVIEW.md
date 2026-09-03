# DDPM Overview

This note explains how the denoising diffusion probabilistic model (DDPM) is used inside `crypto_rebalance_strategy`.

## Goal

The DDPM benchmark is used as a competing forecasting model for cross-sectional crypto returns. At each rebalance date, the model takes the feature matrix `X` and produces a forecast for the forward target return `y`.

In this project, the DDPM is not used for image generation. It is adapted as a conditional tabular regressor for:

- nonlinear return forecasting,
- comparison against Elastic Net and XGBoost,
- walk-forward backtesting with monthly refits.

## Where It Lives

The implementation is in [`ddpm.py`](./ddpm.py).

The model is wired into the benchmark pipeline through:

- [`model.py`](./model.py)
- [`backtest.py`](./backtest.py)
- [`cli.py`](./cli.py)
- [`methodology_report.py`](./methodology_report.py)

## High-Level Idea

A standard DDPM defines a forward process that gradually adds Gaussian noise to a signal, and a reverse process that learns to remove that noise.

Here, the signal is not an image. It is the return target for each asset-date observation.

The workflow is:

1. build technical and market-context predictors `X`,
2. define the forward target `y`,
3. fit a simple linear baseline first,
4. let the DDPM model the residual nonlinear component,
5. sample the reverse diffusion process to obtain a forecast.

## Why Use A Residual DDPM

Using diffusion directly on the full return target is possible, but inefficient for this problem. Returns are noisy, small in magnitude, and the linear component often explains part of the structure already.

So the implementation uses:

`y = y_linear + y_residual`

where:

- `y_linear` comes from a ridge-style linear fit on `X`,
- `y_residual` is modeled by the DDPM.

This has two advantages:

- the diffusion model focuses on nonlinear structure rather than relearning the entire level of returns,
- training is more stable and more efficient in the rolling walk-forward setting.

## Forward Process

Let `r0` denote the scaled residual target.

The diffusion process adds Gaussian noise over a fixed number of timesteps:

`q(rt | r(t-1)) = N(sqrt(1 - beta_t) * r(t-1), beta_t I)`

Over time, the residual is transformed into noise. The model learns how to invert that process conditional on the predictor vector `X`.

## Reverse Process

The reverse model is a compact multilayer perceptron that receives:

- the predictor vector `X`,
- the noisy residual state `rt`,
- the diffusion timestep `t`.

The timestep is encoded with sinusoidal embeddings. The network is trained to predict the noise injected into the residual. During inference, the algorithm iteratively denoises from Gaussian noise back to an estimate of the residual signal.

Multiple reverse-diffusion paths are sampled and averaged to reduce Monte Carlo noise.

## Architecture In This Repo

The implementation is intentionally small because the strategy uses repeated monthly refits inside a rolling backtest.

Main components:

- linear beta schedule,
- sinusoidal timestep embedding,
- MLP denoiser,
- Monte Carlo sampling for prediction,
- gradient clipping for stability,
- feature-importance proxy from first-layer weights.

The model is exposed as `ConditionalTabularDDPMRegressor`, so it behaves like a scikit-learn regressor and can be used inside the same search and backtest workflow as the other benchmarks.

## Training Setup

For each scheduled refit:

1. the rolling training sample is selected,
2. missing features are imputed,
3. features are standardized,
4. the linear baseline is estimated,
5. residuals are scaled,
6. the DDPM is trained on those residuals,
7. hyperparameters are tuned with the same model-selection framework used elsewhere in the project.

The hyperparameters include:

- hidden dimension,
- network depth,
- diffusion timesteps,
- number of training epochs,
- learning rate,
- weight decay,
- ridge penalty,
- number of prediction samples.

## How Prediction Works

For a test cross section:

1. compute the linear baseline forecast,
2. initialize residual paths from Gaussian noise,
3. run the reverse diffusion steps,
4. average the sampled residual forecasts,
5. add the residual forecast back to the linear baseline.

That final output is the model forecast used for ranking assets in the strategy.

## How It Fits The Backtest

The DDPM forecast is treated exactly like the Elastic Net and XGBoost forecasts:

- rank assets by predicted return,
- go long the top names and short the bottom names,
- scale by inverse volatility,
- neutralize to cash, BTC beta, and ETH beta,
- evaluate both prediction and portfolio metrics.

This makes the DDPM a true competing model rather than a separate experiment.

## Current Interpretation

So far, the DDPM implementation is a useful nonlinear benchmark, but it has not yet outperformed XGBoost in the reduced tests run in this repository.

That does not mean the DDPM approach is invalid. It means:

- the current architecture is intentionally compact,
- the training objective is still standard squared-error style forecasting,
- the data is still based on the local sample dataset rather than full production-grade exchange data.

## Reasonable Next Improvements

- train on real exchange-sourced OHLCV data,
- add market-state conditioning features,
- test longer training windows,
- move from point forecast loss to ranking-aware or portfolio-aware objectives,
- evaluate probabilistic outputs from the diffusion samples instead of only the posterior mean,
- compare regime-specific performance across bullish and bearish subperiods.
