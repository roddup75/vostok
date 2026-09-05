from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import which
from subprocess import run
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template


LATEX_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{amsmath}
\title{Cross-Asset Strategy Model Comparison Report}
\date{}
\begin{document}
\maketitle

\section*{Configuration}
\begin{longtable}{ll}
\toprule
Parameter & Value \\
\midrule
{% for key, value in config_rows %}
{{ key }} & {{ value }} \\
{% endfor %}
\textbf{universe\_count} & {{ universe_count }} \\
\bottomrule
\end{longtable}

\section*{Model Comparison}
{% for table in comparison_tables %}
\subsection*{ {{ table.title }} }
\begin{tabular}{l{% for _ in table.model_headers %}r{% endfor %}}
\toprule
Metric {% for header in table.model_headers %}& {{ header }} {% endfor %}\\
\midrule
{% for row in table.rows %}
{{ row.metric }} {% for value in row.cells %}& {{ value }} {% endfor %}\\
{% endfor %}
\bottomrule
\end{tabular}
{% endfor %}

{% for model in model_sections %}
\section*{ {{ model.model_label }} }
\subsection*{Prediction Metrics}
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
{% for key, value in model.prediction_rows %}
{{ key }} & {{ value }} \\
{% endfor %}
\bottomrule
\end{tabular}

\subsection*{Portfolio Metrics}
\begin{tabular}{lr}
\toprule
Metric & Value \\
\midrule
{% for key, value in model.portfolio_rows %}
{{ key }} & {{ value }} \\
{% endfor %}
\bottomrule
\end{tabular}
{% endfor %}

\section*{Figures}
{% for figure in figures %}
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../plots/{{ figure.filename }}}
\caption{ {{ figure.caption }} }
\end{figure}
{% endfor %}

\appendix
\section*{Universe}
\begin{longtable}{l}
\toprule
Asset \\
\midrule
{% for asset in universe_assets %}
{{ asset }} \\
{% endfor %}
\bottomrule
\end{longtable}

\end{document}
"""


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_table_rows(mapping: Dict[str, object]) -> Iterable[Tuple[str, str]]:
    rows = []
    for key, value in mapping.items():
        if key == "universe":
            continue
        rows.append((_latex_escape(key), _latex_escape(_format_scalar(value))))
    return rows


def _comparison_table(title: str, comparison_mapping: Dict[str, Dict[str, float]], model_names: list[str]) -> Dict[str, object]:
    rows = []
    for metric, values in comparison_mapping.items():
        rows.append(
            {
                "metric": _latex_escape(metric),
                "cells": [_latex_escape(_format_scalar(values[model_name])) for model_name in model_names],
            }
        )
    return {
        "title": _latex_escape(title),
        "model_headers": [_latex_escape(model_name) for model_name in model_names],
        "rows": rows,
    }


def _select_top_features(parameter_history: pd.DataFrame, model_name: str, top_n: int = 12) -> list[str]:
    subset = parameter_history.loc[parameter_history["model_name"] == model_name].copy()
    if subset.empty:
        return []
    ranking = subset.groupby("feature")["value"].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
    return ranking.head(top_n).index.tolist()


def _figure_filename(model_name: str) -> str:
    return f"{model_name}_parameter_paths.png"


def create_plots(
    predictions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    cross_sectional_diagnostics: pd.DataFrame,
    parameter_history: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    nav_frame = portfolio_returns.pivot(index="date", columns="model_label", values="nav")
    drawdown_frame = nav_frame.divide(nav_frame.cummax()).subtract(1.0)

    plt.figure(figsize=(10, 5))
    for column in nav_frame.columns:
        plt.plot(nav_frame.index, nav_frame[column], linewidth=2, label=column)
    plt.title("NAV Curve")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "nav_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for column in drawdown_frame.columns:
        plt.plot(drawdown_frame.index, drawdown_frame[column], linewidth=2, label=column)
    plt.title("Drawdown")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "drawdown.png", dpi=180)
    plt.close()

    model_labels = predictions["model_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(model_labels), figsize=(7 * len(model_labels), 6), squeeze=False)
    for ax, model_label in zip(axes[0], model_labels):
        subset = predictions.loc[predictions["model_label"] == model_label]
        ax.scatter(subset["prediction"], subset["target_return"], alpha=0.35, s=16)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(model_label)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Realized forward return")
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_label, subset in cross_sectional_diagnostics.groupby("model_label"):
        plt.plot(subset["date"], subset["cross_sectional_r2"], linewidth=2, label=model_label)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Cross-Sectional R2 by Rebalance Date")
    plt.ylabel("R2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "cross_sectional_r2.png", dpi=180)
    plt.close()

    forecast_matrix = predictions.pivot_table(
        index=["date", "asset"],
        columns="model_label",
        values="prediction",
        aggfunc="first",
    )
    model_pairs = [
        ("Elastic Net", "XGBoost"),
        ("Elastic Net", "DDPM"),
        ("DDPM", "XGBoost"),
    ]
    pairwise_rows = []
    for date, date_matrix in forecast_matrix.groupby(level="date"):
        date_matrix = date_matrix.droplevel("date")
        for left_model, right_model in model_pairs:
            model_pair = f"{left_model} vs {right_model}"
            if left_model not in date_matrix.columns or right_model not in date_matrix.columns:
                pairwise_rows.append({"date": date, "model_pair": model_pair, "correlation": float("nan")})
                continue
            pair = date_matrix[[left_model, right_model]].dropna()
            if len(pair) < 2 or pair[left_model].nunique() < 2 or pair[right_model].nunique() < 2:
                pairwise_rows.append({"date": date, "model_pair": model_pair, "correlation": float("nan")})
                continue
            pairwise_rows.append(
                {
                    "date": date,
                    "model_pair": model_pair,
                    "correlation": pair[left_model].corr(pair[right_model]),
                }
            )
    pairwise_correlations = pd.DataFrame(pairwise_rows)
    if not pairwise_correlations.empty:
        pairwise_correlation_frame = pairwise_correlations.pivot(
            index="date",
            columns="model_pair",
            values="correlation",
        ).sort_index()
        plt.figure(figsize=(10, 5))
        for left_model, right_model in model_pairs:
            model_pair = f"{left_model} vs {right_model}"
            if model_pair in pairwise_correlation_frame.columns:
                plt.plot(
                    pairwise_correlation_frame.index,
                    pairwise_correlation_frame[model_pair],
                    linewidth=1.8,
                    label=model_pair,
                )
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title("Cross-Sectional Correlation Between Model Forecasts")
        plt.ylabel("Forecast Correlation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "cross_sectional_forecast_correlation.png", dpi=180)
        plt.close()

    for model_name in parameter_history["model_name"].drop_duplicates().tolist():
        top_features = _select_top_features(parameter_history, model_name=model_name, top_n=12)
        if not top_features:
            continue
        subset = parameter_history.loc[
            (parameter_history["model_name"] == model_name) & (parameter_history["feature"].isin(top_features))
        ]
        if subset.empty:
            continue
        plt.figure(figsize=(11, 6))
        for feature, feature_df in subset.groupby("feature"):
            plt.plot(feature_df["date"], feature_df["value"], linewidth=1.8, label=feature)
        if model_name == "elastic_net":
            plt.axhline(0.0, color="black", linewidth=1)
        plt.title(f"{subset['model_label'].iloc[0]} Parameter Paths")
        plt.ylabel(subset["parameter_name"].iloc[0].replace("_", " ").title())
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_dir / _figure_filename(model_name), dpi=180, bbox_inches="tight")
        plt.close()

    return plots_dir


def write_latex_report(
    summary_metrics: Dict[str, object],
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    universe_assets = [_latex_escape(asset) for asset in summary_metrics["config"]["universe"]]
    models = summary_metrics["models"]
    model_names = list(models.keys())

    model_sections = []
    figures = [
        {"filename": "nav_curve.png", "caption": "NAV comparison"},
        {"filename": "drawdown.png", "caption": "Drawdown comparison"},
        {"filename": "prediction_scatter.png", "caption": "Predicted versus realized forward returns by model"},
        {"filename": "cross_sectional_r2.png", "caption": "Cross-sectional R2 stability over time"},
        {
            "filename": "cross_sectional_forecast_correlation.png",
            "caption": "Pairwise cross-sectional correlation between model forecasts over time",
        },
    ]
    for model_name in model_names:
        model_sections.append(
            {
                "model_label": _latex_escape(models[model_name]["model_label"]),
                "prediction_rows": _format_table_rows(models[model_name]["prediction_metrics"]),
                "portfolio_rows": _format_table_rows(models[model_name]["portfolio_metrics"]),
            }
        )
        figures.append(
            {
                "filename": _figure_filename(model_name),
                "caption": f"{models[model_name]['model_label']} parameter stability",
            }
        )

    template = Template(LATEX_TEMPLATE)
    tex = template.render(
        config_rows=_format_table_rows(summary_metrics["config"]),
        universe_count=len(summary_metrics["config"]["universe"]),
        universe_assets=universe_assets,
        comparison_tables=[
            _comparison_table("Prediction Metrics", summary_metrics["comparison"]["prediction_metrics"], model_names),
            _comparison_table("Portfolio Metrics", summary_metrics["comparison"]["portfolio_metrics"], model_names),
        ],
        model_sections=model_sections,
        figures=figures,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_path = report_dir / f"report_{timestamp}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def compile_pdf(tex_path: str | Path) -> Path | None:
    tex_path = Path(tex_path)
    if which("pdflatex") is None:
        return None
    run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None
