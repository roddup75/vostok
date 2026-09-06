from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_EQUITY_DATA_PATH = str(
    PACKAGE_ROOT
    / "equity_data"
    / "synthetic_factor_1000x360_20260905"
    / "equity_model_panel_parts"
)

DEFAULT_UNIVERSE = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "SOL-USD",
    "ADA-USD",
    "DOGE-USD",
    "TRX-USD",
    "AVAX-USD",
    "SHIB-USD",
    "DOT-USD",
    "LINK-USD",
    "BCH-USD",
    "LTC-USD",
    "XLM-USD",
    "TON11419-USD",
    "HBAR-USD",
    "SUI20947-USD",
    "APT21794-USD",
    "NEAR-USD",
    "ICP-USD",
    "PEPE24478-USD",
    "AAVE-USD",
    "ETC-USD",
    "MKR-USD",
    "POL-USD",
    "UNI7083-USD",
    "ATOM-USD",
    "FIL-USD",
    "OP-USD",
]


@dataclass
class StrategyConfig:
    asset_class: str = "crypto"
    universe: Sequence[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    equity_data_path: str = DEFAULT_EQUITY_DATA_PATH
    equity_universe_size: int = 1000
    equity_long_short_quantile: float = 0.20
    start_date: str = "2021-09-01"
    end_date: str = "2026-09-01"
    horizon: int = 5
    rebalance_frequency: str = "7D"
    train_window_months: int = 6
    model_refit_frequency: str = "M"
    min_train_samples: int = 500
    top_n: int = 5
    bottom_n: int = 5
    gross_leverage: float = 1.0
    transaction_cost_bps: float = 10.0
    feature_windows: Sequence[int] = field(default_factory=lambda: [3, 5, 7, 10, 14, 20, 30, 50])
    breakout_windows: Sequence[int] = field(default_factory=lambda: [10, 20, 50])
    ewma_windows: Sequence[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volatility_windows: Sequence[int] = field(default_factory=lambda: [5, 10, 20, 30])
    volume_windows: Sequence[int] = field(default_factory=lambda: [5, 10, 20, 30])
    rsi_windows: Sequence[int] = field(default_factory=lambda: [7, 14, 28])
    beta_window: int = 60
    position_vol_window: int = 20
    elastic_net_max_iter: int = 10000
    elastic_net_alpha: float = 1e-4
    elastic_net_l1_ratio: float = 0.50
    elastic_net_feature_pruning: bool = True
    elastic_net_max_features: int = 35
    elastic_net_min_features: int = 15
    elastic_net_min_feature_dates: int = 20
    elastic_net_feature_corr_threshold: float = 0.95
    cv_splits: int = 4
    bayes_iter: int = 25
    random_search_iter: int = 25
    random_state: int = 7
    ddpm_hidden_dim: int = 64
    ddpm_depth: int = 2
    ddpm_timesteps: int = 16
    ddpm_epochs: int = 20
    ddpm_batch_size: int = 256
    ddpm_learning_rate: float = 7.5e-4
    ddpm_weight_decay: float = 5e-5
    ddpm_ridge_alpha: float = 1e-3
    ddpm_gradient_clip: float = 1.0
    ddpm_predict_samples: int = 24

    def validate(self) -> None:
        if self.asset_class not in {"crypto", "equity"}:
            raise ValueError("asset_class must be one of 'crypto' or 'equity'")
        if self.asset_class == "crypto" and self.horizon not in {1, 3, 5}:
            raise ValueError("crypto horizon must be one of 1, 3, or 5")
        if self.asset_class == "equity" and self.horizon not in {1, 3}:
            raise ValueError("equity horizon must be one of 1 or 3 monthly periods")
        if self.top_n <= 0 or self.bottom_n <= 0:
            raise ValueError("top_n and bottom_n must be positive")
        if self.train_window_months <= 0:
            raise ValueError("train_window_months must be positive")
        if self.model_refit_frequency not in {"M", "Q"}:
            raise ValueError("model_refit_frequency must be one of 'M' or 'Q'")
        if self.asset_class == "crypto" and self.model_refit_frequency != "M":
            raise ValueError("crypto model_refit_frequency currently supports monthly refits only: 'M'")
        if self.asset_class == "equity" and self.rebalance_frequency not in {"M", "ME"}:
            raise ValueError("equity rebalance_frequency must be monthly: 'M' or 'ME'")
        if self.equity_universe_size <= 1:
            raise ValueError("equity_universe_size must be greater than 1")
        if not 0.0 < self.equity_long_short_quantile < 0.5:
            raise ValueError("equity_long_short_quantile must be in (0, 0.5)")
        if self.min_train_samples <= 0:
            raise ValueError("min_train_samples must be positive")
        if self.beta_window <= 1:
            raise ValueError("beta_window must be greater than 1")
        if self.position_vol_window <= 1:
            raise ValueError("position_vol_window must be greater than 1")
        if self.elastic_net_max_features <= 0 or self.elastic_net_min_features <= 0:
            raise ValueError("elastic-net feature limits must be positive")
        if self.elastic_net_alpha <= 0:
            raise ValueError("elastic_net_alpha must be positive")
        if not 0.0 <= self.elastic_net_l1_ratio <= 1.0:
            raise ValueError("elastic_net_l1_ratio must be in [0, 1]")
        if self.elastic_net_min_features > self.elastic_net_max_features:
            raise ValueError("elastic_net_min_features cannot exceed elastic_net_max_features")
        if self.elastic_net_min_feature_dates <= 0:
            raise ValueError("elastic_net_min_feature_dates must be positive")
        if not 0.0 < self.elastic_net_feature_corr_threshold <= 1.0:
            raise ValueError("elastic_net_feature_corr_threshold must be in (0, 1]")
        if self.bayes_iter < 0 or self.random_search_iter < 0:
            raise ValueError("bayes_iter and random_search_iter must be non-negative")
        if self.ddpm_timesteps <= 1:
            raise ValueError("ddpm_timesteps must be greater than 1")
        if self.ddpm_epochs <= 0 or self.ddpm_batch_size <= 0:
            raise ValueError("ddpm_epochs and ddpm_batch_size must be positive")
        if self.ddpm_ridge_alpha < 0 or self.ddpm_gradient_clip < 0:
            raise ValueError("ddpm_ridge_alpha and ddpm_gradient_clip must be non-negative")
        if self.asset_class == "crypto" and (not self.universe or len(self.universe) < self.top_n + self.bottom_n):
            raise ValueError("universe is too small for the chosen long/short construction")

    @property
    def universe_list(self) -> List[str]:
        return list(self.universe)
