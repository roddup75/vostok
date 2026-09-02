from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from shutil import which
from subprocess import run
from typing import Dict

from jinja2 import Template


METHODOLOGY_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{amsmath}
\title{Crypto Cross-Sectional Forecasting Methodology}
\author{}
\date{ {{ report_date }} }
\begin{document}
\maketitle
\section*{Objective}
Build a cross-sectional long/short crypto strategy that forecasts forward returns for a liquid multi-asset universe and compares three model classes: Elastic Net, XGBoost, and a conditional denoising diffusion probabilistic model (DDPM). The workflow is fully walk-forward: features are measured at date $t$, the target is a forward return over the selected horizon, and the portfolio is formed only from the information available at the rebalance date.

\section*{Data, Universe, and Labels}
\begin{itemize}
\item Universe size: {{ universe_count }} large-cap crypto assets.
\item Sample window: {{ start_date }} to {{ end_date }}.
\item Rebalance frequency: every {{ rebalance_frequency }}.
\item Forecast horizon: {{ horizon }} trading days ahead.
\item Estimation window: rolling {{ train_window_months }}-month training sample with monthly model refits.
\item Target definition: asset forward return minus a BTC/ETH beta-adjusted benchmark component, so the model focuses more on idiosyncratic cross-sectional variation.
\end{itemize}

\section*{Feature Engineering}
For each asset-date pair, the feature matrix $X_t$ combines technical signals across multiple windows:
\begin{itemize}
\item returns and momentum z-scores,
\item range position, breakout scores, EWMAs, RSI, and Elliott-wave-style oscillator proxy,
\item volatility, ATR-style range scaling, abnormal volume, and dollar-volume z-scores,
\item lagged BTC and ETH daily and weekly returns,
\item rolling BTC and ETH betas and an asset volatility estimate used later for sizing.
\end{itemize}
To stabilize cross-sectional learning, asset-specific predictors are transformed date by date using rank normalization followed by the inverse normal map. This preserves ordering while reducing the influence of outliers and heterogeneous scales across coins.

\section*{Walk-Forward Backtest}
At each rebalance date:
\begin{enumerate}
\item build the rolling training set from the prior {{ train_window_months }} months;
\item refit each model only on scheduled monthly update dates;
\item generate one-step cross-sectional forecasts for all assets in the universe;
\item rank forecasts and trade the top {{ top_n }} longs versus bottom {{ bottom_n }} shorts;
\item scale positions by forecast strength divided by recent asset volatility;
\item project the portfolio to be neutral to cash, BTC beta, and ETH beta.
\end{enumerate}
Performance is measured with prediction statistics (MSE, RMSE, MAE, bias, correlation, information coefficient, and $R^2$) and trading statistics (annualized return, volatility, Sharpe, Sortino, hit ratio, drawdown, turnover, and trading cost).

\section*{Competing Models}
\textbf{Elastic Net.} A linear benchmark with L1/L2 shrinkage. It is useful when the signal is diffuse across many correlated predictors and when coefficient stability matters.

\textbf{XGBoost.} A nonlinear tree ensemble benchmark. It captures threshold effects, interactions, and asymmetric response patterns in the predictor set.

\textbf{Conditional DDPM.} The diffusion model is used as a conditional regressor for $y \mid X$, not as an unconditional generator. The implementation proceeds in two layers:
\begin{enumerate}
\item A ridge-style linear baseline first maps $X$ into a coarse forecast $\hat{y}^{\text{lin}}$.
\item The DDPM models the residual $r = y - \hat{y}^{\text{lin}}$ by gradually adding Gaussian noise in a forward process and learning a denoiser that removes that noise conditional on $X$ and the diffusion step index.
\end{enumerate}
The denoiser is a compact MLP with sinusoidal timestep embeddings. At inference time, multiple reverse-diffusion paths are sampled and averaged, then added back to the linear baseline. This design makes the diffusion model focus on nonlinear residual structure instead of relearning the entire level of the target.

In notation, if $r_0$ is the scaled residual, the forward process is
\[
q(r_t \mid r_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}r_{t-1}, \beta_t I),
\]
and the network is trained to predict the injected noise $\epsilon$ from $(X, r_t, t)$. The reverse process then iteratively reconstructs an estimate of $r_0$, whose Monte Carlo mean becomes the residual forecast.

\section*{Reduced Benchmark Snapshot}
\begin{center}
\begin{tabular}{lrrr}
\toprule
Metric & Elastic Net & XGBoost & DDPM \\
\midrule
MSE & {{ mse_elastic }} & {{ mse_xgboost }} & {{ mse_ddpm }} \\
IC (Pearson) & {{ ic_elastic }} & {{ ic_xgboost }} & {{ ic_ddpm }} \\
$R^2$ & {{ r2_elastic }} & {{ r2_xgboost }} & {{ r2_ddpm }} \\
Ann.\ Return & {{ ret_elastic }} & {{ ret_xgboost }} & {{ ret_ddpm }} \\
Sharpe & {{ sharpe_elastic }} & {{ sharpe_xgboost }} & {{ sharpe_ddpm }} \\
Max Drawdown & {{ dd_elastic }} & {{ dd_xgboost }} & {{ dd_ddpm }} \\
\bottomrule
\end{tabular}
\end{center}

On this reduced walk-forward run, XGBoost remains the strongest benchmark. The DDPM improves on Elastic Net in portfolio return and drawdown, but it still trails XGBoost and does not yet outperform the simpler benchmarks consistently on forecast quality. The result is still informative: the diffusion model is viable in this framework, and the residual-DDPM formulation is a more appropriate starting point than a raw unconditional generative setup.

\section*{Interpretation and Next Steps}
\begin{itemize}
\item Negative $R^2$ is possible and means the model underperforms a simple mean benchmark on the evaluation sample.
\item The current DDPM is intentionally compact to keep monthly refits tractable inside a walk-forward backtest.
\item The most promising next upgrades are real exchange-sourced OHLCV data, richer market-state conditioning, and direct training on ranking or portfolio-aware objectives rather than pure squared error.
\end{itemize}

\end{document}
"""


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def write_methodology_report(summary_metrics: Dict[str, object], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    models = summary_metrics["models"]
    template = Template(METHODOLOGY_TEMPLATE)
    tex = template.render(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe_count=len(summary_metrics["config"]["universe"]),
        start_date=summary_metrics["config"]["start_date"],
        end_date=summary_metrics["config"]["end_date"],
        rebalance_frequency=summary_metrics["config"]["rebalance_frequency"],
        horizon=summary_metrics["config"]["horizon"],
        train_window_months=summary_metrics["config"]["train_window_months"],
        top_n=summary_metrics["config"]["top_n"],
        bottom_n=summary_metrics["config"]["bottom_n"],
        mse_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["mse"]),
        mse_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["mse"]),
        mse_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["mse"]),
        ic_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["information_coefficient_pearson"]),
        ic_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["information_coefficient_pearson"]),
        ic_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["information_coefficient_pearson"]),
        r2_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["r2"]),
        r2_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["r2"]),
        r2_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["r2"]),
        ret_elastic=_fmt(models["elastic_net"]["portfolio_metrics"]["annualized_return"]),
        ret_xgboost=_fmt(models["xgboost"]["portfolio_metrics"]["annualized_return"]),
        ret_ddpm=_fmt(models["ddpm"]["portfolio_metrics"]["annualized_return"]),
        sharpe_elastic=_fmt(models["elastic_net"]["portfolio_metrics"]["sharpe_ratio"]),
        sharpe_xgboost=_fmt(models["xgboost"]["portfolio_metrics"]["sharpe_ratio"]),
        sharpe_ddpm=_fmt(models["ddpm"]["portfolio_metrics"]["sharpe_ratio"]),
        dd_elastic=_fmt(models["elastic_net"]["portfolio_metrics"]["max_drawdown"]),
        dd_xgboost=_fmt(models["xgboost"]["portfolio_metrics"]["max_drawdown"]),
        dd_ddpm=_fmt(models["ddpm"]["portfolio_metrics"]["max_drawdown"]),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_path = report_dir / f"methodology_report_{timestamp}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def compile_methodology_pdf(tex_path: str | Path) -> Path | None:
    tex_path = Path(tex_path)
    latex_engine = which("xelatex") or which("lualatex") or which("pdflatex")
    if latex_engine is None:
        return None
    run(
        [latex_engine, "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def build_methodology_report_from_json(summary_json_path: str | Path) -> Path | None:
    summary_json_path = Path(summary_json_path)
    summary_metrics = json.loads(summary_json_path.read_text(encoding="utf-8"))
    tex_path = write_methodology_report(summary_metrics=summary_metrics, output_dir=summary_json_path.parent)
    return compile_methodology_pdf(tex_path)
