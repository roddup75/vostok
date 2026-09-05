from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_backtest, save_artifacts
from .config import DEFAULT_EQUITY_DATA_PATH, DEFAULT_UNIVERSE, StrategyConfig
from .crypto_prediction import build_crypto_prediction_frame
from .report import compile_pdf, create_plots, write_latex_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crypto or equity model-comparison rebalance backtests.")
    parser.add_argument("--asset-class", default="crypto", choices=["crypto", "equity"])
    parser.add_argument("--data-dir", default=None, help="Optional directory of per-asset csv files.")
    parser.add_argument("--provider", default="binance", choices=["binance", "yfinance"])
    parser.add_argument("--equity-data-path", default=DEFAULT_EQUITY_DATA_PATH)
    parser.add_argument("--equity-universe-size", type=int, default=1000)
    parser.add_argument("--equity-long-short-quantile", type=float, default=0.20)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--rebalance-frequency", default=None, choices=["3D", "7D", "M", "ME"])
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 3, 5])
    parser.add_argument("--train-window-months", type=int, default=None)
    parser.add_argument("--min-train-samples", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--bottom-n", type=int, default=5)
    parser.add_argument("--gross-leverage", type=float, default=1.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--position-vol-window", type=int, default=20)
    parser.add_argument("--no-elastic-net-feature-pruning", action="store_true")
    parser.add_argument("--elastic-net-max-features", type=int, default=35)
    parser.add_argument("--elastic-net-min-features", type=int, default=15)
    parser.add_argument("--elastic-net-min-feature-dates", type=int, default=20)
    parser.add_argument("--elastic-net-feature-corr-threshold", type=float, default=0.95)
    parser.add_argument("--model-refit-frequency", default=None, choices=["M", "Q"])
    parser.add_argument("--bayes-iter", type=int, default=25)
    parser.add_argument("--random-search-iter", type=int, default=25)
    parser.add_argument("--ddpm-hidden-dim", type=int, default=64)
    parser.add_argument("--ddpm-depth", type=int, default=2)
    parser.add_argument("--ddpm-timesteps", type=int, default=16)
    parser.add_argument("--ddpm-epochs", type=int, default=20)
    parser.add_argument("--ddpm-batch-size", type=int, default=256)
    parser.add_argument("--ddpm-learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--ddpm-weight-decay", type=float, default=5e-5)
    parser.add_argument("--ddpm-ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--ddpm-gradient-clip", type=float, default=1.0)
    parser.add_argument("--ddpm-predict-samples", type=int, default=24)
    parser.add_argument("--output-dir", default="crypto_rebalance_strategy/output")
    parser.add_argument(
        "--universe",
        nargs="+",
        default=list(DEFAULT_UNIVERSE),
        help="Universe tickers, for example BTC-USD ETH-USD",
    )
    return parser.parse_args()


def _resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.asset_class == "crypto":
        args.start_date = args.start_date or "2021-09-01"
        args.end_date = args.end_date or "2026-09-01"
        args.rebalance_frequency = args.rebalance_frequency or "7D"
        args.horizon = args.horizon or 5
        args.train_window_months = args.train_window_months or 6
        args.min_train_samples = args.min_train_samples or 500
        args.model_refit_frequency = args.model_refit_frequency or "M"
    else:
        args.start_date = args.start_date or "1996-01-31"
        args.end_date = args.end_date or "2025-12-31"
        args.rebalance_frequency = args.rebalance_frequency or "M"
        args.horizon = args.horizon or 1
        args.train_window_months = args.train_window_months or 60
        args.min_train_samples = args.min_train_samples or 5_000
        args.model_refit_frequency = args.model_refit_frequency or "Q"
    return args


def main() -> None:
    args = _resolve_defaults(parse_args())
    config = StrategyConfig(
        asset_class=args.asset_class,
        universe=args.universe,
        equity_data_path=args.equity_data_path,
        equity_universe_size=args.equity_universe_size,
        equity_long_short_quantile=args.equity_long_short_quantile,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        rebalance_frequency=args.rebalance_frequency,
        train_window_months=args.train_window_months,
        min_train_samples=args.min_train_samples,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        gross_leverage=args.gross_leverage,
        transaction_cost_bps=args.transaction_cost_bps,
        beta_window=args.beta_window,
        position_vol_window=args.position_vol_window,
        elastic_net_feature_pruning=not args.no_elastic_net_feature_pruning,
        elastic_net_max_features=args.elastic_net_max_features,
        elastic_net_min_features=args.elastic_net_min_features,
        elastic_net_min_feature_dates=args.elastic_net_min_feature_dates,
        elastic_net_feature_corr_threshold=args.elastic_net_feature_corr_threshold,
        model_refit_frequency=args.model_refit_frequency,
        bayes_iter=args.bayes_iter,
        random_search_iter=args.random_search_iter,
        ddpm_hidden_dim=args.ddpm_hidden_dim,
        ddpm_depth=args.ddpm_depth,
        ddpm_timesteps=args.ddpm_timesteps,
        ddpm_epochs=args.ddpm_epochs,
        ddpm_batch_size=args.ddpm_batch_size,
        ddpm_learning_rate=args.ddpm_learning_rate,
        ddpm_weight_decay=args.ddpm_weight_decay,
        ddpm_ridge_alpha=args.ddpm_ridge_alpha,
        ddpm_gradient_clip=args.ddpm_gradient_clip,
        ddpm_predict_samples=args.ddpm_predict_samples,
    )
    config.validate()

    if config.asset_class == "crypto":
        feature_df = build_crypto_prediction_frame(config=config, data_dir=args.data_dir, provider=args.provider)
    else:
        from .equity_prediction import build_equity_prediction_frame

        feature_df = build_equity_prediction_frame(config=config)
    artifacts = run_backtest(feature_df=feature_df, config=config)
    output_dir = Path(args.output_dir)
    save_artifacts(artifacts=artifacts, output_dir=output_dir)
    create_plots(
        predictions=artifacts.predictions,
        portfolio_returns=artifacts.portfolio_returns,
        cross_sectional_diagnostics=artifacts.cross_sectional_diagnostics,
        parameter_history=artifacts.parameter_history,
        output_dir=output_dir,
    )
    tex_path = write_latex_report(
        summary_metrics=artifacts.summary_metrics,
        predictions=artifacts.predictions,
        output_dir=output_dir,
    )
    pdf_path = compile_pdf(tex_path)
    print(f"Wrote outputs to {output_dir}")
    print(f"LaTeX report: {tex_path}")
    if pdf_path is not None:
        print(f"PDF report: {pdf_path}")
    else:
        print("PDF report was not generated because pdflatex is unavailable.")


if __name__ == "__main__":
    main()
