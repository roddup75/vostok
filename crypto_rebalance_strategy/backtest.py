from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd

from .config import StrategyConfig
from .features import get_feature_columns
from .metrics import summarize_portfolio
from .model import MODEL_LABELS, extract_parameter_vector, fit_model, predict_cross_section, score_predictions


@dataclass
class BacktestArtifacts:
    predictions: pd.DataFrame
    portfolio_returns: pd.DataFrame
    cross_sectional_diagnostics: pd.DataFrame
    parameter_history: pd.DataFrame
    summary_metrics: Dict[str, object]


def _select_rebalance_dates(feature_df: pd.DataFrame, frequency: str) -> List[pd.Timestamp]:
    by_date = feature_df[["date"]].drop_duplicates().sort_values("date").set_index("date")
    picks = by_date.resample(frequency).last().dropna().index.tolist()
    return [pd.Timestamp(x) for x in picks]


def _build_weights(test_slice: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    ranked = test_slice.sort_values("prediction", ascending=False)
    longs = ranked.head(config.top_n).copy()
    shorts = ranked.tail(config.bottom_n).copy()
    weights = pd.Series(0.0, index=test_slice["asset"])

    def _scaled_side(df: pd.DataFrame, sign: float) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        vol = df["asset_vol"].replace(0.0, np.nan).fillna(df["asset_vol"].median())
        strength = df["prediction"].abs().clip(lower=1e-8)
        raw = strength.div(vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=df.index)
        scaled = sign * (config.gross_leverage / 2.0) * raw / raw.sum()
        return pd.Series(scaled.to_numpy(), index=df["asset"].to_numpy())

    long_weights = _scaled_side(longs, sign=1.0)
    short_weights = _scaled_side(shorts, sign=-1.0)
    if not long_weights.empty:
        weights.loc[long_weights.index] = long_weights
    if not short_weights.empty:
        weights.loc[short_weights.index] = short_weights

    selected_assets = weights[weights != 0.0].index.tolist()
    if selected_assets:
        beta_df = test_slice.set_index("asset").loc[selected_assets, ["beta_btc", "beta_eth"]].fillna(0.0)
        basis = np.column_stack(
            [
                np.ones(len(selected_assets)),
                beta_df["beta_btc"].to_numpy(),
                beta_df["beta_eth"].to_numpy(),
            ]
        )
        original = weights.loc[selected_assets].to_numpy()
        projection = basis @ np.linalg.pinv(basis) @ original
        neutralized = original - projection
        if np.abs(neutralized).sum() > 0:
            neutralized = config.gross_leverage * neutralized / np.abs(neutralized).sum()
        weights.loc[selected_assets] = neutralized
    return weights


def _select_refit_dates(feature_df: pd.DataFrame) -> set[pd.Timestamp]:
    by_date = feature_df[["date"]].drop_duplicates().sort_values("date").set_index("date")
    return set(pd.Timestamp(x) for x in by_date.resample("M").last().dropna().index.tolist())


def _information_coefficient(predictions: pd.DataFrame, method: str) -> float:
    per_date = []
    for _, group in predictions.groupby("date"):
        if group["prediction"].nunique() < 2 or group["target_return"].nunique() < 2:
            continue
        value = group["prediction"].corr(group["target_return"], method=method)
        if pd.notna(value):
            per_date.append(float(value))
    if not per_date:
        return float("nan")
    return float(pd.Series(per_date).mean())


def _cross_sectional_r2(actual: pd.Series, predicted: pd.Series) -> float:
    denom = np.square(actual - actual.mean()).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.square(actual - predicted).sum() / denom)


def _make_cross_sectional_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = predictions.groupby(["model_name", "model_label", "date"])
    for (model_name, model_label, date), group in grouped:
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "date": date,
                "cross_sectional_r2": _cross_sectional_r2(group["target_return"], group["prediction"]),
                "cross_sectional_ic_pearson": group["prediction"].corr(group["target_return"], method="pearson"),
                "cross_sectional_ic_spearman": group["prediction"].corr(group["target_return"], method="spearman"),
                "cross_sectional_mae": float((group["target_return"] - group["prediction"]).abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "date"]).reset_index(drop=True)


def _run_single_model_backtest(
    feature_df: pd.DataFrame,
    config: StrategyConfig,
    model_name: str,
) -> BacktestArtifacts:
    feature_cols = get_feature_columns(config)
    rebalance_dates = _select_rebalance_dates(feature_df, config.rebalance_frequency)
    refit_dates = _select_refit_dates(feature_df)
    prediction_rows: List[pd.DataFrame] = []
    portfolio_rows: List[Dict[str, object]] = []
    parameter_rows: List[pd.DataFrame] = []
    turnover_prev = pd.Series(0.0, index=config.universe_list)
    search_types: List[str] = []
    params_history: List[Dict[str, float]] = []
    fit = None

    for rebalance_date in rebalance_dates:
        test_mask = feature_df["date"] == rebalance_date
        test_df = feature_df.loc[test_mask].copy()
        if test_df.empty:
            continue

        should_refit = fit is None or rebalance_date in refit_dates
        if should_refit:
            train_start = rebalance_date - pd.DateOffset(months=config.train_window_months)
            train_mask = (feature_df["date"] < rebalance_date) & (feature_df["date"] >= train_start)
            train_df = feature_df.loc[train_mask].copy()
            if len(train_df) < config.min_train_samples:
                continue
            fit = fit_model(model_name=model_name, train_df=train_df, X_cols=feature_cols, config=config)
            search_types.append(fit.search_type)
            params_history.append({k: float(v) for k, v in fit.best_params.items()})
            parameter_df = extract_parameter_vector(fit.best_estimator, model_name=model_name, feature_names=feature_cols)
            parameter_df["date"] = rebalance_date
            parameter_df["model_name"] = model_name
            parameter_df["model_label"] = MODEL_LABELS[model_name]
            parameter_df["search_type"] = fit.search_type
            parameter_df["cv_score"] = fit.cv_score
            parameter_rows.append(parameter_df)

        if fit is None:
            continue

        _, pred_df = predict_cross_section(fit.best_estimator, test_df, feature_cols)
        weights = _build_weights(pred_df, config)
        pred_df["model_name"] = model_name
        pred_df["model_label"] = MODEL_LABELS[model_name]
        pred_df["weight"] = pred_df["asset"].map(weights).fillna(0.0)
        pred_df["gross_weight"] = pred_df["weight"].abs()
        pred_df["model_search_type"] = fit.search_type
        prediction_rows.append(pred_df)

        realized_gross = float((pred_df["weight"] * pred_df["target_return"]).sum())
        current_weights = weights.reindex(config.universe_list).fillna(0.0)
        turnover = float((current_weights - turnover_prev).abs().sum())
        trading_cost = turnover * (config.transaction_cost_bps / 10000.0)
        net_return = realized_gross - trading_cost
        portfolio_rows.append(
            {
                "date": rebalance_date,
                "model_name": model_name,
                "model_label": MODEL_LABELS[model_name],
                "gross_return": realized_gross,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "net_return": net_return,
                "n_longs": int((current_weights > 0).sum()),
                "n_shorts": int((current_weights < 0).sum()),
                "model_search_type": fit.search_type,
            }
        )
        turnover_prev = current_weights

    if not prediction_rows or not portfolio_rows:
        raise ValueError(f"Backtest produced no rebalance observations for {model_name}.")

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(
        ["date", "prediction"], ascending=[True, False]
    )
    portfolio_returns = pd.DataFrame(portfolio_rows).sort_values("date").reset_index(drop=True)
    portfolio_returns["nav"] = (1.0 + portfolio_returns["net_return"]).cumprod()
    cross_sectional_diagnostics = _make_cross_sectional_diagnostics(predictions)
    parameter_history = pd.concat(parameter_rows, ignore_index=True).sort_values(["model_name", "date", "feature"])
    pred_metrics = score_predictions(predictions["target_return"], predictions["prediction"])
    pred_metrics["information_coefficient_pearson"] = _information_coefficient(predictions, method="pearson")
    pred_metrics["information_coefficient_spearman"] = _information_coefficient(predictions, method="spearman")
    portfolio_metrics = summarize_portfolio(portfolio_returns.set_index("date")["net_return"])

    summary = {
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
        "prediction_metrics": pred_metrics,
        "portfolio_metrics": portfolio_metrics,
        "backtest_observations": int(len(portfolio_returns)),
        "prediction_observations": int(len(predictions)),
        "model_refits": int(len(params_history)),
        "avg_turnover": float(portfolio_returns["turnover"].mean()),
        "avg_trading_cost": float(portfolio_returns["trading_cost"].mean()),
        "avg_cross_sectional_r2": float(cross_sectional_diagnostics["cross_sectional_r2"].mean()),
        "search_types_used": sorted(set(search_types)),
        "latest_best_params": params_history[-1],
    }
    return BacktestArtifacts(
        predictions=predictions,
        portfolio_returns=portfolio_returns,
        cross_sectional_diagnostics=cross_sectional_diagnostics,
        parameter_history=parameter_history,
        summary_metrics=summary,
    )


def _build_comparison(models_summary: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    prediction_metric_keys = sorted(next(iter(models_summary.values()))["prediction_metrics"].keys())
    portfolio_metric_keys = sorted(next(iter(models_summary.values()))["portfolio_metrics"].keys())
    return {
        "prediction_metrics": {
            key: {model_name: models_summary[model_name]["prediction_metrics"][key] for model_name in models_summary}
            for key in prediction_metric_keys
        },
        "portfolio_metrics": {
            key: {model_name: models_summary[model_name]["portfolio_metrics"][key] for model_name in models_summary}
            for key in portfolio_metric_keys
        },
    }


def run_backtest(feature_df: pd.DataFrame, config: StrategyConfig) -> BacktestArtifacts:
    model_names = ["elastic_net", "xgboost", "ddpm"]
    model_artifacts = {
        model_name: _run_single_model_backtest(feature_df=feature_df, config=config, model_name=model_name)
        for model_name in model_names
    }
    predictions = pd.concat([artifact.predictions for artifact in model_artifacts.values()], ignore_index=True)
    portfolio_returns = pd.concat(
        [artifact.portfolio_returns for artifact in model_artifacts.values()],
        ignore_index=True,
    )
    cross_sectional_diagnostics = pd.concat(
        [artifact.cross_sectional_diagnostics for artifact in model_artifacts.values()],
        ignore_index=True,
    )
    parameter_history = pd.concat(
        [artifact.parameter_history for artifact in model_artifacts.values()],
        ignore_index=True,
    )
    models_summary = {model_name: artifact.summary_metrics for model_name, artifact in model_artifacts.items()}
    summary = {
        "config": asdict(config),
        "models": models_summary,
        "comparison": _build_comparison(models_summary),
    }
    return BacktestArtifacts(
        predictions=predictions,
        portfolio_returns=portfolio_returns,
        cross_sectional_diagnostics=cross_sectional_diagnostics,
        parameter_history=parameter_history,
        summary_metrics=summary,
    )


def save_artifacts(artifacts: BacktestArtifacts, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.predictions.to_csv(output_dir / "predictions.csv", index=False)
    artifacts.portfolio_returns.to_csv(output_dir / "portfolio_returns.csv", index=False)
    artifacts.cross_sectional_diagnostics.to_csv(output_dir / "cross_sectional_diagnostics.csv", index=False)
    artifacts.parameter_history.to_csv(output_dir / "parameter_history.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(artifacts.summary_metrics, handle, indent=2, sort_keys=True)
