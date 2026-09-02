from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import StrategyConfig


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _feature_columns(config: StrategyConfig) -> List[str]:
    cols: List[str] = []
    for window in config.feature_windows:
        cols.extend(
            [
                f"ret_{window}",
                f"mom_z_{window}",
                f"range_pos_{window}",
                f"vol_z_{window}",
            ]
        )
    for window in config.ewma_windows:
        cols.extend([f"ewma_gap_{window}", f"price_ewma_ratio_{window}"])
    for window in config.rsi_windows:
        cols.append(f"rsi_{window}")
    for window in config.breakout_windows:
        cols.extend([f"breakout_up_{window}", f"breakout_down_{window}"])
    for window in config.volume_windows:
        cols.extend([f"abn_volume_{window}", f"dollar_volume_z_{window}"])
    for window in config.volatility_windows:
        cols.extend([f"vol_{window}", f"atr_ratio_{window}"])
    cols.extend(["ewo", "intraday_reversal", "overnight_gap", "high_low_spread"])
    cols.extend(
        [
            "btc_ret_lag1",
            "btc_ret_lag2",
            "btc_ret_7d_lag1",
            "btc_ret_7d_lag2",
            "eth_ret_lag1",
            "eth_ret_lag2",
            "eth_ret_7d_lag1",
            "eth_ret_7d_lag2",
            "beta_btc",
            "beta_eth",
            "asset_vol",
        ]
    )
    return cols


def _build_benchmark_regressors(market_data: pd.DataFrame) -> pd.DataFrame:
    benchmark_frames = []
    benchmark_map = {
        "BTC-USD": "btc",
        "ETH-USD": "eth",
    }
    for asset, prefix in benchmark_map.items():
        asset_df = market_data.loc[market_data["asset"] == asset, ["date", "close"]].sort_values("date").copy()
        daily_return = asset_df["close"].pct_change()
        weekly_return = asset_df["close"].pct_change(7)
        benchmark_frames.append(
            pd.DataFrame(
                {
                    "date": asset_df["date"].to_numpy(),
                    f"{prefix}_ret_lag1": daily_return.shift(1).to_numpy(),
                    f"{prefix}_ret_lag2": daily_return.shift(2).to_numpy(),
                    f"{prefix}_ret_7d_lag1": weekly_return.shift(1).to_numpy(),
                    f"{prefix}_ret_7d_lag2": weekly_return.shift(2).to_numpy(),
                }
            )
        )

    benchmark_df = benchmark_frames[0]
    for extra in benchmark_frames[1:]:
        benchmark_df = benchmark_df.merge(extra, on="date", how="inner")
    return benchmark_df


def _cross_sectional_rank_gaussianize(feature_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    def _transform(group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        out = group.copy()
        if n <= 1:
            return out
        for col in columns:
            rank = out[col].rank(method="average", pct=False)
            u = (rank - 0.5) / n
            u = u.clip(lower=1e-6, upper=1 - 1e-6)
            out[col] = norm.ppf(u)
        return out

    return feature_df.groupby("date", group_keys=False).apply(_transform)


def build_feature_frame(market_data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    benchmark_map = {"BTC-USD": "btc", "ETH-USD": "eth"}
    frames = []
    for asset, asset_df in market_data.groupby("asset", sort=True):
        df = asset_df.sort_values("date").copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"]
        log_close = np.log(close.replace(0.0, np.nan))
        daily_ret = close.pct_change()
        dollar_volume = close * volume
        asset_vol = daily_ret.rolling(config.position_vol_window).std()

        for window in config.feature_windows:
            df[f"ret_{window}"] = close.pct_change(window)
            df[f"mom_z_{window}"] = _zscore(daily_ret, window)
            rolling_high = high.rolling(window).max()
            rolling_low = low.rolling(window).min()
            df[f"range_pos_{window}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0.0, np.nan)
            df[f"vol_z_{window}"] = _zscore(daily_ret.rolling(window).std(), window)

        for window in config.ewma_windows:
            ewma = close.ewm(span=window, adjust=False, min_periods=window).mean()
            df[f"ewma_gap_{window}"] = (close - ewma) / ewma.replace(0.0, np.nan)
            df[f"price_ewma_ratio_{window}"] = close / ewma.replace(0.0, np.nan)

        for window in config.rsi_windows:
            df[f"rsi_{window}"] = _rsi(close, window)

        for window in config.breakout_windows:
            prev_high = high.rolling(window).max().shift(1)
            prev_low = low.rolling(window).min().shift(1)
            df[f"breakout_up_{window}"] = (close - prev_high) / prev_high.replace(0.0, np.nan)
            df[f"breakout_down_{window}"] = (close - prev_low) / prev_low.replace(0.0, np.nan)

        for window in config.volume_windows:
            rolling_vol = volume.rolling(window).mean()
            df[f"abn_volume_{window}"] = volume / rolling_vol.replace(0.0, np.nan)
            df[f"dollar_volume_z_{window}"] = _zscore(dollar_volume, window)

        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for window in config.volatility_windows:
            df[f"vol_{window}"] = daily_ret.rolling(window).std()
            atr = true_range.rolling(window).mean()
            df[f"atr_ratio_{window}"] = atr / close.replace(0.0, np.nan)

        ema_fast = close.ewm(span=5, adjust=False, min_periods=5).mean()
        ema_slow = close.ewm(span=35, adjust=False, min_periods=35).mean()
        df["ewo"] = (ema_fast - ema_slow) / close.replace(0.0, np.nan)
        df["intraday_reversal"] = (close - open_) / open_.replace(0.0, np.nan)
        df["overnight_gap"] = (open_ - close.shift(1)) / close.shift(1).replace(0.0, np.nan)
        df["high_low_spread"] = (high - low) / close.replace(0.0, np.nan)
        df["target_return_raw"] = close.shift(-config.horizon) / close - 1.0
        df["next_period_close"] = close.shift(-config.horizon)
        df["log_close"] = log_close
        df["daily_return"] = daily_ret
        df["asset_vol"] = asset_vol
        frames.append(df)

    feature_df = pd.concat(frames, ignore_index=True)
    benchmark_df = _build_benchmark_regressors(market_data)
    feature_df = feature_df.merge(benchmark_df, on="date", how="left")
    benchmark_future = []
    for asset, prefix in benchmark_map.items():
        asset_df = market_data.loc[market_data["asset"] == asset, ["date", "close"]].sort_values("date").copy()
        benchmark_future.append(
            pd.DataFrame(
                {
                    "date": asset_df["date"].to_numpy(),
                    f"{prefix}_forward_return": asset_df["close"].shift(-config.horizon).div(asset_df["close"]).sub(1.0).to_numpy(),
                }
            )
        )
    future_df = benchmark_future[0]
    for extra in benchmark_future[1:]:
        future_df = future_df.merge(extra, on="date", how="inner")
    feature_df = feature_df.merge(future_df, on="date", how="left")

    feature_df["beta_btc"] = np.nan
    feature_df["beta_eth"] = np.nan
    for asset, asset_df in feature_df.groupby("asset", sort=False):
        idx = asset_df.index
        asset_ret = asset_df["daily_return"]
        btc_var = asset_df["btc_ret_lag1"].rolling(config.beta_window).var()
        eth_var = asset_df["eth_ret_lag1"].rolling(config.beta_window).var()
        feature_df.loc[idx, "beta_btc"] = asset_ret.rolling(config.beta_window).cov(asset_df["btc_ret_lag1"]).div(
            btc_var.replace(0.0, np.nan)
        ).to_numpy()
        feature_df.loc[idx, "beta_eth"] = asset_ret.rolling(config.beta_window).cov(asset_df["eth_ret_lag1"]).div(
            eth_var.replace(0.0, np.nan)
        ).to_numpy()

    feature_df["target_return"] = feature_df["target_return_raw"] - 0.5 * (
        feature_df["beta_btc"] * feature_df["btc_forward_return"] +
        feature_df["beta_eth"] * feature_df["eth_forward_return"]
    )
    feature_cols = _feature_columns(config)
    cross_sectional_cols = [col for col in feature_cols if not col.startswith(("btc_", "eth_", "beta_", "asset_vol"))]
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = _cross_sectional_rank_gaussianize(feature_df, cross_sectional_cols)
    feature_df = feature_df.dropna(subset=feature_cols + ["target_return"]).reset_index(drop=True)
    return feature_df


def get_feature_columns(config: StrategyConfig) -> List[str]:
    return _feature_columns(config)
