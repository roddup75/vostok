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
\usepackage[margin=0.82in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\title{Crypto Cross-Sectional Forecasting Methodology}
\author{}
\date{ {{ report_date }} }
\begin{document}
\maketitle

\begin{abstract}
This report documents a walk-forward cross-sectional crypto forecasting system using real daily OHLCV data, technical predictors, beta-adjusted residual return labels, and three forecasting models: Elastic Net, XGBoost, and a conditional denoising diffusion probabilistic model (DDPM). The design objective is not point prediction alone; it is ranking quality that can support a market-neutral long/short portfolio under transaction costs and beta constraints.
\end{abstract}

\section*{1. Research Design}
At each rebalance date $t$, the system estimates a function $f_t(X_{i,t})$ for each asset $i$ and trades on the cross-sectional ranking of forecasts. The backtest is strictly walk-forward: model fitting uses only observations available before $t$, refits occur monthly, and portfolio weights are formed from contemporaneous features and out-of-sample forecasts.

\section*{2. Data and Label Construction}
\begin{itemize}
\item Data source: cached Binance daily spot OHLCV files loaded through the local real-data path.
\item Universe: {{ universe_count }} liquid crypto assets.
\item Sample window: {{ start_date }} to {{ end_date }}.
\item Forecast horizon: {{ horizon }} days; rebalance frequency: {{ rebalance_frequency }}.
\item Training window: rolling {{ train_window_months }} months, with a minimum of {{ min_train_samples }} samples.
\end{itemize}
The raw label is the future asset return over the forecast horizon. To reduce common market exposure, the implemented target subtracts a BTC/ETH beta-adjusted benchmark component:
\[
y_{i,t} = R_{i,t:t+h} - \frac{1}{2}\left(\beta^{BTC}_{i,t}R^{BTC}_{t:t+h} + \beta^{ETH}_{i,t}R^{ETH}_{t:t+h}\right).
\]
This makes the supervised problem closer to idiosyncratic cross-sectional return forecasting than broad market direction forecasting.

\section*{3. Technical Feature Set}
The feature matrix combines trend, reversal, volatility, volume, and market-state information across multiple horizons. Asset-specific predictors include simple returns, momentum z-scores, range position, breakout distance, EWMA gaps, price-to-EWMA ratios, RSI, abnormal volume, dollar-volume z-scores, realized volatility, ATR-style range measures, intraday reversal, overnight gap, high-low spread, and an Elliott-wave-style oscillator proxy. Shared market features include lagged BTC and ETH daily and weekly returns, rolling BTC/ETH betas, and recent asset volatility.

Before modelling, most asset-specific predictors are transformed date by date using cross-sectional rank normalization followed by an inverse-normal map. This preserves the ordering information needed for ranking while limiting the impact of heterogeneous scales, listing histories, and crypto-specific outliers.

\section*{4. Model Selection Objective}
The current model-selection objective is cross-sectional Spearman information coefficient (IC), not mean-squared error. For each validation fold, forecasts are grouped by date and scored by their same-date rank correlation with realized target returns. The fold score is the average valid date-level IC. This better matches the portfolio construction problem, because the strategy trades ranks rather than calibrated return magnitudes.

Hyperparameter search uses Bayesian optimization when \texttt{scikit-optimize} is installed and otherwise falls back to randomized search. The current run used {{ search_type }}.

\section*{5. Model Specifications}
\textbf{Elastic Net.} Elastic Net is the linear benchmark, combining L1 and L2 shrinkage. Because the technical feature set is broad and correlated, Elastic Net now includes train-window-only feature pruning before model search. Each feature is ranked by historical cross-sectional Spearman IC, then redundant predictors are removed using a Spearman correlation threshold of {{ elastic_corr_threshold }}. The default selected set is capped at {{ elastic_max_features }} features, with a floor of {{ elastic_min_features }}.

\textbf{XGBoost.} XGBoost provides the nonlinear tree benchmark. It is included to capture threshold effects, interactions among technical signals, and nonlinear responses that a linear shrinkage model cannot express.

\textbf{Conditional DDPM.} The DDPM is implemented as a conditional residual regressor. A ridge-style linear head first estimates a baseline forecast $\hat{y}^{lin}$, then a compact denoising MLP models the residual $r = y - \hat{y}^{lin}$. The forward diffusion process adds Gaussian noise to scaled residuals, and the network learns to predict that noise conditional on $(X, r_t, t)$. At inference, multiple reverse-diffusion paths are averaged and added back to the linear baseline. The ridge head now uses adaptive regularization and a least-squares fallback to avoid singular-matrix failures in small or collinear training folds.

\section*{6. Portfolio Construction and Diagnostics}
At each rebalance date, forecasts are ranked cross-sectionally. The strategy buys the top {{ top_n }} assets and shorts the bottom {{ bottom_n }} assets. Position size is proportional to forecast strength divided by recent volatility, then projected to be neutral to cash, BTC beta, and ETH beta. Transaction costs are applied through turnover.

Diagnostics include global prediction metrics, date-level cross-sectional $R^2$, Pearson IC, Spearman IC, parameter or feature-importance paths, drawdowns, NAV curves, and pairwise forecast correlations between Elastic Net, XGBoost, and DDPM. The pairwise correlation plot is useful because strong model agreement can indicate redundant signals, while disagreement identifies dates where ensemble or model-selection logic may matter.

\section*{7. Empirical Snapshot}
\begin{center}
\begin{tabular}{lrrr}
\toprule
Metric & Elastic Net & XGBoost & DDPM \\
\midrule
Features Used & {{ features_elastic }} & {{ features_xgboost }} & {{ features_ddpm }} \\
MSE & {{ mse_elastic }} & {{ mse_xgboost }} & {{ mse_ddpm }} \\
Spearman IC & {{ sic_elastic }} & {{ sic_xgboost }} & {{ sic_ddpm }} \\
Pearson IC & {{ ic_elastic }} & {{ ic_xgboost }} & {{ ic_ddpm }} \\
$R^2$ & {{ r2_elastic }} & {{ r2_xgboost }} & {{ r2_ddpm }} \\
Ann.\ Return & {{ ret_elastic }} & {{ ret_xgboost }} & {{ ret_ddpm }} \\
Ann.\ Volatility & {{ vol_elastic }} & {{ vol_xgboost }} & {{ vol_ddpm }} \\
Sharpe & {{ sharpe_elastic }} & {{ sharpe_xgboost }} & {{ sharpe_ddpm }} \\
Max Drawdown & {{ dd_elastic }} & {{ dd_xgboost }} & {{ dd_ddpm }} \\
\bottomrule
\end{tabular}
\end{center}

In the latest real-data run, XGBoost remains the strongest economic benchmark. Elastic Net feature pruning materially improved portfolio return relative to the unpruned IC-optimized run, but its drawdown remains large and its average IC declined. DDPM became numerically more stable and improved versus its earlier MSE-tuned version, but it still does not dominate the simpler alternatives.

\section*{8. Limitations and Next Work}
The results should be read as research evidence, not production trading guidance. The universe has changing listing histories, daily bars omit intraday liquidity and funding information, and transaction costs are simplified. The next defensible additions are turnover-aware portfolio construction, conviction thresholds for weak Elastic Net ranks, feature-stability constraints across refits, and model ensembling based on rolling validation IC.

\end{document}
"""


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _feature_count(model_summary: Dict[str, object], fallback: int) -> str:
    return str(model_summary.get("latest_feature_count") or fallback)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_methodology_report(summary_metrics: Dict[str, object], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    models = summary_metrics["models"]
    config = summary_metrics["config"]
    feature_count = len(config["feature_windows"]) * 4
    feature_count += len(config["ewma_windows"]) * 2
    feature_count += len(config["rsi_windows"])
    feature_count += len(config["breakout_windows"]) * 2
    feature_count += len(config["volume_windows"]) * 2
    feature_count += len(config["volatility_windows"]) * 2
    feature_count += 15
    template = Template(METHODOLOGY_TEMPLATE)
    tex = template.render(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        universe_count=len(config["universe"]),
        start_date=config["start_date"],
        end_date=config["end_date"],
        rebalance_frequency=config["rebalance_frequency"],
        horizon=config["horizon"],
        train_window_months=config["train_window_months"],
        min_train_samples=config["min_train_samples"],
        top_n=config["top_n"],
        bottom_n=config["bottom_n"],
        elastic_max_features=config.get("elastic_net_max_features", "n/a"),
        elastic_min_features=config.get("elastic_net_min_features", "n/a"),
        elastic_corr_threshold=_fmt(config.get("elastic_net_feature_corr_threshold", "n/a")),
        search_type=_latex_escape(", ".join(models["elastic_net"].get("search_types_used", [])) or "n/a"),
        features_elastic=_feature_count(models["elastic_net"], feature_count),
        features_xgboost=_feature_count(models["xgboost"], feature_count),
        features_ddpm=_feature_count(models["ddpm"], feature_count),
        mse_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["mse"]),
        mse_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["mse"]),
        mse_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["mse"]),
        sic_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["information_coefficient_spearman"]),
        sic_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["information_coefficient_spearman"]),
        sic_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["information_coefficient_spearman"]),
        ic_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["information_coefficient_pearson"]),
        ic_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["information_coefficient_pearson"]),
        ic_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["information_coefficient_pearson"]),
        r2_elastic=_fmt(models["elastic_net"]["prediction_metrics"]["r2"]),
        r2_xgboost=_fmt(models["xgboost"]["prediction_metrics"]["r2"]),
        r2_ddpm=_fmt(models["ddpm"]["prediction_metrics"]["r2"]),
        ret_elastic=_fmt(models["elastic_net"]["portfolio_metrics"]["annualized_return"]),
        ret_xgboost=_fmt(models["xgboost"]["portfolio_metrics"]["annualized_return"]),
        ret_ddpm=_fmt(models["ddpm"]["portfolio_metrics"]["annualized_return"]),
        vol_elastic=_fmt(models["elastic_net"]["portfolio_metrics"]["annualized_volatility"]),
        vol_xgboost=_fmt(models["xgboost"]["portfolio_metrics"]["annualized_volatility"]),
        vol_ddpm=_fmt(models["ddpm"]["portfolio_metrics"]["annualized_volatility"]),
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
