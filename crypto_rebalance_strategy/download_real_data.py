from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_UNIVERSE
from .data import load_market_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache real crypto OHLCV data.")
    parser.add_argument("--provider", default="binance", choices=["binance", "yfinance"])
    parser.add_argument("--start-date", default="2021-09-01")
    parser.add_argument("--end-date", default="2026-09-03")
    parser.add_argument("--output-dir", default="crypto_rebalance_strategy/real_data_binance")
    parser.add_argument(
        "--universe",
        nargs="+",
        default=list(DEFAULT_UNIVERSE),
        help="Universe tickers, for example BTC-USD ETH-USD",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_market_data(
        universe=args.universe,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=None,
        provider=args.provider,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for asset, asset_df in data.groupby("asset", sort=True):
        asset_df.drop(columns=["asset"]).to_csv(output_dir / f"{asset}.csv", index=False)
    print(f"Wrote {data['asset'].nunique()} assets to {output_dir}")


if __name__ == "__main__":
    main()
