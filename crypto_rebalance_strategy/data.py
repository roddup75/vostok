from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_frame(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    rename_map = {col: col.lower().strip() for col in frame.columns}
    frame = frame.rename(columns=rename_map).copy()
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame["date"] = frame["datetime"]
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{asset} is missing required columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame[REQUIRED_COLUMNS].dropna().sort_values("date").drop_duplicates("date")
    frame["asset"] = asset
    return frame.reset_index(drop=True)


def load_local_ohlcv(data_dir: str | Path, universe: Iterable[str]) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for asset in universe:
        csv_path = data_dir / f"{asset}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {csv_path}")
        frames.append(_normalize_frame(pd.read_csv(csv_path), asset))
    return pd.concat(frames, ignore_index=True)


def load_yfinance_ohlcv(
    universe: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for remote downloads") from exc

    frames = []
    for asset in universe:
        history = yf.download(
            tickers=asset,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            interval="1d",
        )
        if history.empty:
            raise ValueError(f"No history returned for {asset}")
        history = history.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frames.append(_normalize_frame(history, asset))
    return pd.concat(frames, ignore_index=True)


def load_market_data(
    universe: Iterable[str],
    start_date: str,
    end_date: str,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    if data_dir:
        data = load_local_ohlcv(data_dir=data_dir, universe=universe)
    else:
        data = load_yfinance_ohlcv(universe=universe, start_date=start_date, end_date=end_date)
    mask = (data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))
    return data.loc[mask].reset_index(drop=True)
