from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .data import load_market_data
from .features import build_feature_frame


def build_crypto_prediction_frame(config: StrategyConfig, data_dir: str | None, provider: str) -> pd.DataFrame:
    market_data = load_market_data(
        universe=config.universe_list,
        start_date=config.start_date,
        end_date=config.end_date,
        data_dir=data_dir,
        provider=provider,
    )
    return build_feature_frame(market_data=market_data, config=config)
