from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_backtest, save_artifacts
from .config import DEFAULT_UNIVERSE, StrategyConfig
from .data import load_market_data
from .features import build_feature_frame
from .report import compile_pdf, create_plots, write_latex_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a crypto elastic-net rebalance backtest.")
    parser.add_argument("--data-dir", default=None, help="Optional directory of per-asset csv files.")
    parser.add_argument("--start-date", default="2021-09-01")
    parser.add_argument("--end-date", default="2026-09-01")
    parser.add_argument("--rebalance-frequency", default="7D", choices=["3D", "7D"])
    parser.add_argument("--horizon", type=int, default=5, choices=[1, 3, 5])
    parser.add_argument("--train-window-months", type=int, default=6)
    parser.add_argument("--min-train-samples", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--bottom-n", type=int, default=5)
    parser.add_argument("--gross-leverage", type=float, default=1.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--beta-window", type=int, default=60)
    parser.add_argument("--position-vol-window", type=int, default=20)
    parser.add_argument("--model-refit-frequency", default="M", choices=["M"])
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


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        universe=args.universe,
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

    market_data = load_market_data(
        universe=config.universe_list,
        start_date=config.start_date,
        end_date=config.end_date,
        data_dir=args.data_dir,
    )
    feature_df = build_feature_frame(market_data=market_data, config=config)
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
