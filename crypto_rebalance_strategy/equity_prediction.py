from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from .config import StrategyConfig


EQUITY_FORWARD_RETURN_COLUMNS = {
    1: "forward_return_1m",
    3: "forward_return_3m",
}


def _rank_gaussianize_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    def _transform(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        count = len(out)
        if count <= 1:
            return out
        for column in columns:
            rank = out[column].rank(method="average", pct=False)
            uniform = ((rank - 0.5) / count).clip(lower=1e-6, upper=1 - 1e-6)
            out[column] = norm.ppf(uniform)
        return out

    return frame.groupby("date", group_keys=False).apply(_transform)


def _panel_files(path: Path) -> list[Path]:
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            return parquet_files
        text_files = sorted(path.glob("*.txt")) + sorted(path.glob("*.tsv"))
        if text_files:
            return text_files
        raise FileNotFoundError(f"No parquet, txt, or tsv files found in {path}")
    return [path]


def _list_panel_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema.names
    return pd.read_csv(path, sep="\t", nrows=0).columns.tolist()


def _read_panel_file(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(path, sep="\t", parse_dates=["date"], usecols=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_equity_panel(path: str | Path, universe_size: int) -> pd.DataFrame:
    files = _panel_files(Path(path))
    available_columns = _list_panel_columns(files[0])
    observed_columns = [column for column in available_columns if column.startswith("observed_factor_")]
    required_columns = [
        "date",
        "asset_id",
        "stock_return_1m",
        "forward_return_1m",
        "forward_return_3m",
        "close",
        "elastic_net_alpha",
        "price_adv_20",
        "spread_bps",
        "impact_coefficient",
        "commission_bps",
    ]
    use_columns = [column for column in required_columns + observed_columns if column in available_columns]
    panel = pd.concat([_read_panel_file(file, use_columns) for file in files], ignore_index=True)
    asset_ids = sorted(panel["asset_id"].drop_duplicates().tolist())[:universe_size]
    return panel.loc[panel["asset_id"].isin(asset_ids)].copy()


def _build_asset_volatility(panel: pd.DataFrame) -> pd.Series:
    ordered = panel.sort_values(["asset", "date"])
    volatility = ordered.groupby("asset")["stock_return_1m"].rolling(12, min_periods=3).std()
    return volatility.reset_index(level=0, drop=True).reindex(panel.index)


def build_equity_prediction_frame(config: StrategyConfig) -> pd.DataFrame:
    forward_column = EQUITY_FORWARD_RETURN_COLUMNS[config.horizon]
    panel = _load_equity_panel(config.equity_data_path, config.equity_universe_size)
    panel = panel.rename(columns={"asset_id": "asset"}).copy()
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)

    mask = (panel["date"] >= pd.Timestamp(config.start_date)) & (panel["date"] <= pd.Timestamp(config.end_date))
    panel = panel.loc[mask].copy()
    market_forward_return = panel.groupby("date")[forward_column].transform("mean")
    panel["target_return"] = panel[forward_column] - market_forward_return
    panel["market_forward_return"] = market_forward_return
    panel["asset_vol"] = _build_asset_volatility(panel)
    panel["close"] = panel["close"].astype(float)

    feature_cols = get_equity_feature_columns(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = _rank_gaussianize_by_date(panel, feature_cols)
    panel = panel.dropna(subset=feature_cols + ["target_return", "asset_vol", "close"]).reset_index(drop=True)
    return panel[["date", "asset", "target_return", "close", "asset_vol", "market_forward_return"] + feature_cols]


def get_equity_feature_columns(panel: pd.DataFrame | None = None) -> list[str]:
    if panel is None:
        return [f"observed_factor_{index + 1:03d}" for index in range(50)] + [
            "elastic_net_alpha",
            "price_adv_20",
            "spread_bps",
            "impact_coefficient",
            "commission_bps",
        ]
    observed = sorted(column for column in panel.columns if column.startswith("observed_factor_"))
    optional = [
        "elastic_net_alpha",
        "price_adv_20",
        "spread_bps",
        "impact_coefficient",
        "commission_bps",
    ]
    return observed + [column for column in optional if column in panel.columns]
