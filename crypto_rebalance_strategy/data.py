from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_INTERVAL = "1d"
BINANCE_LIMIT = 1000
BINANCE_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "BNB-USD": "BNBUSDT",
    "XRP-USD": "XRPUSDT",
    "SOL-USD": "SOLUSDT",
    "ADA-USD": "ADAUSDT",
    "DOGE-USD": "DOGEUSDT",
    "TRX-USD": "TRXUSDT",
    "AVAX-USD": "AVAXUSDT",
    "SHIB-USD": "SHIBUSDT",
    "DOT-USD": "DOTUSDT",
    "LINK-USD": "LINKUSDT",
    "BCH-USD": "BCHUSDT",
    "LTC-USD": "LTCUSDT",
    "XLM-USD": "XLMUSDT",
    "TON11419-USD": "TONUSDT",
    "HBAR-USD": "HBARUSDT",
    "SUI20947-USD": "SUIUSDT",
    "APT21794-USD": "APTUSDT",
    "NEAR-USD": "NEARUSDT",
    "ICP-USD": "ICPUSDT",
    "PEPE24478-USD": "PEPEUSDT",
    "AAVE-USD": "AAVEUSDT",
    "ETC-USD": "ETCUSDT",
    "MKR-USD": "MKRUSDT",
    "POL-USD": "POLUSDT",
    "UNI7083-USD": "UNIUSDT",
    "ATOM-USD": "ATOMUSDT",
    "FIL-USD": "FILUSDT",
    "OP-USD": "OPUSDT",
}


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


def _binance_klines_to_frame(rows: list[list[object]], asset: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return _normalize_frame(frame[["date", "open", "high", "low", "close", "volume"]], asset)


def _fetch_binance_klines(symbol: str, start_ms: int, end_ms: int) -> list[list[object]]:
    all_rows: list[list[object]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": symbol,
                "interval": BINANCE_INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": BINANCE_LIMIT,
            }
        )
        req = urllib.request.Request(f"{BINANCE_BASE_URL}?{params}", headers={"user-agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            rows = json.load(response)
        if not rows:
            break
        all_rows.extend(rows)
        next_open_time = int(rows[-1][0]) + 24 * 60 * 60 * 1000
        if next_open_time <= cursor:
            break
        cursor = next_open_time
        time.sleep(0.15)
    return all_rows


def load_binance_ohlcv(
    universe: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    frames = []
    for asset in universe:
        symbol = BINANCE_SYMBOL_MAP.get(asset)
        if symbol is None:
            raise KeyError(f"No Binance symbol mapping configured for {asset}")
        rows = _fetch_binance_klines(symbol=symbol, start_ms=start_ms, end_ms=end_ms)
        if not rows:
            raise ValueError(f"No Binance OHLCV returned for {asset} ({symbol})")
        frames.append(_binance_klines_to_frame(rows, asset))
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
    provider: str = "binance",
) -> pd.DataFrame:
    if data_dir:
        data = load_local_ohlcv(data_dir=data_dir, universe=universe)
    elif provider == "binance":
        data = load_binance_ohlcv(universe=universe, start_date=start_date, end_date=end_date)
    elif provider == "yfinance":
        data = load_yfinance_ohlcv(universe=universe, start_date=start_date, end_date=end_date)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    mask = (data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))
    return data.loc[mask].reset_index(drop=True)
