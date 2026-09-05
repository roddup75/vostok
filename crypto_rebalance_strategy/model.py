from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import StrategyConfig
from .ddpm import ConditionalTabularDDPMRegressor


MODEL_LABELS = {
    "elastic_net": "Elastic Net",
    "xgboost": "XGBoost",
    "ddpm": "DDPM",
}


@dataclass
class ModelFitResult:
    best_estimator: Any
    best_params: Dict[str, float]
    search_type: str
    cv_score: float
    model_name: str
    feature_names: list[str]


def _make_elastic_pipeline(config: StrategyConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=config.elastic_net_max_iter,
                    fit_intercept=True,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _make_xgboost_pipeline(config: StrategyConfig) -> Pipeline:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is required to run the XGBoost benchmark") from exc

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    random_state=config.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _make_ddpm_pipeline(config: StrategyConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ConditionalTabularDDPMRegressor(
                    hidden_dim=config.ddpm_hidden_dim,
                    depth=config.ddpm_depth,
                    timesteps=config.ddpm_timesteps,
                    epochs=config.ddpm_epochs,
                    batch_size=config.ddpm_batch_size,
                    learning_rate=config.ddpm_learning_rate,
                    weight_decay=config.ddpm_weight_decay,
                    ridge_alpha=config.ddpm_ridge_alpha,
                    gradient_clip=config.ddpm_gradient_clip,
                    predict_samples=config.ddpm_predict_samples,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _make_estimator(model_name: str, config: StrategyConfig) -> Any:
    if model_name == "elastic_net":
        return _make_elastic_pipeline(config)
    if model_name == "xgboost":
        return _make_xgboost_pipeline(config)
    if model_name == "ddpm":
        return _make_ddpm_pipeline(config)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _search_space_skopt(model_name: str) -> Dict[str, Any]:
    from skopt.space import Integer, Real

    if model_name == "elastic_net":
        return {
            "model__alpha": Real(1e-5, 10.0, prior="log-uniform"),
            "model__l1_ratio": Real(0.01, 0.99),
        }
    if model_name == "xgboost":
        return {
            "model__n_estimators": Integer(100, 600),
            "model__max_depth": Integer(2, 8),
            "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "model__subsample": Real(0.5, 1.0),
            "model__colsample_bytree": Real(0.5, 1.0),
            "model__reg_alpha": Real(1e-6, 10.0, prior="log-uniform"),
            "model__reg_lambda": Real(1e-3, 20.0, prior="log-uniform"),
            "model__min_child_weight": Real(1.0, 10.0),
        }
    if model_name == "ddpm":
        return {
            "model__hidden_dim": Integer(32, 96),
            "model__depth": Integer(2, 3),
            "model__timesteps": Integer(8, 16),
            "model__epochs": Integer(12, 24),
            "model__learning_rate": Real(3e-4, 2e-3, prior="log-uniform"),
            "model__weight_decay": Real(1e-6, 2e-4, prior="log-uniform"),
            "model__ridge_alpha": Real(1e-5, 1e-2, prior="log-uniform"),
            "model__predict_samples": Integer(12, 24),
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def _search_space_randomized(model_name: str) -> Dict[str, Any]:
    if model_name == "elastic_net":
        return {
            "model__alpha": loguniform(1e-5, 10.0),
            "model__l1_ratio": uniform(0.01, 0.98),
        }
    if model_name == "xgboost":
        return {
            "model__n_estimators": randint(100, 601),
            "model__max_depth": randint(2, 9),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__subsample": uniform(0.5, 0.5),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_alpha": loguniform(1e-6, 10.0),
            "model__reg_lambda": loguniform(1e-3, 20.0),
            "model__min_child_weight": uniform(1.0, 9.0),
        }
    if model_name == "ddpm":
        return {
            "model__hidden_dim": randint(32, 97),
            "model__depth": randint(2, 4),
            "model__timesteps": randint(8, 17),
            "model__epochs": randint(12, 25),
            "model__learning_rate": loguniform(3e-4, 2e-3),
            "model__weight_decay": loguniform(1e-6, 2e-4),
            "model__ridge_alpha": loguniform(1e-5, 1e-2),
            "model__predict_samples": randint(12, 25),
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def _make_cross_sectional_ic_scorer(train_df: pd.DataFrame) -> Callable[[Any, pd.DataFrame, pd.Series], float]:
    dates_by_index = train_df["date"]

    def _score(estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        predictions = estimator.predict(X)
        if hasattr(X, "index"):
            dates = dates_by_index.reindex(X.index).to_numpy()
        else:
            dates = dates_by_index.iloc[: len(y)].to_numpy()
        score_frame = pd.DataFrame(
            {
                "date": dates,
                "actual": np.asarray(y, dtype=float),
                "prediction": np.asarray(predictions, dtype=float),
            }
        ).dropna()
        per_date_scores = []
        for _, group in score_frame.groupby("date"):
            if group["actual"].nunique() < 2 or group["prediction"].nunique() < 2:
                continue
            value = group["prediction"].corr(group["actual"], method="spearman")
            if pd.notna(value):
                per_date_scores.append(float(value))
        if not per_date_scores:
            return -1.0
        return float(np.mean(per_date_scores))

    return _score


def _feature_cross_sectional_ic(train_df: pd.DataFrame, feature_name: str) -> tuple[float, int]:
    per_date_scores = []
    for _, group in train_df[["date", feature_name, "target_return"]].dropna().groupby("date"):
        if group[feature_name].nunique() < 2 or group["target_return"].nunique() < 2:
            continue
        value = group[feature_name].corr(group["target_return"], method="spearman")
        if pd.notna(value):
            per_date_scores.append(float(value))
    if not per_date_scores:
        return 0.0, 0
    return float(np.mean(per_date_scores)), len(per_date_scores)


def _select_elastic_net_features(train_df: pd.DataFrame, X_cols: list[str], config: StrategyConfig) -> list[str]:
    if not config.elastic_net_feature_pruning or len(X_cols) <= config.elastic_net_max_features:
        return X_cols

    scored_features = []
    for feature_name in X_cols:
        mean_ic, valid_dates = _feature_cross_sectional_ic(train_df, feature_name)
        scored_features.append(
            {
                "feature": feature_name,
                "abs_mean_ic": abs(mean_ic),
                "valid_dates": valid_dates,
            }
        )
    score_frame = pd.DataFrame(scored_features).sort_values(
        ["valid_dates", "abs_mean_ic"],
        ascending=[False, False],
    )
    eligible = score_frame.loc[score_frame["valid_dates"] >= config.elastic_net_min_feature_dates].copy()
    if len(eligible) < config.elastic_net_min_features:
        eligible = score_frame.copy()
    ranked_features = eligible.sort_values("abs_mean_ic", ascending=False)["feature"].tolist()

    selected_features: list[str] = []
    corr_frame = train_df[X_cols].corr(method="spearman").abs()
    for feature_name in ranked_features:
        if len(selected_features) >= config.elastic_net_max_features:
            break
        if not selected_features:
            selected_features.append(feature_name)
            continue
        max_corr = corr_frame.loc[feature_name, selected_features].max()
        if pd.isna(max_corr) or max_corr < config.elastic_net_feature_corr_threshold:
            selected_features.append(feature_name)

    if len(selected_features) < config.elastic_net_min_features:
        for feature_name in ranked_features:
            if len(selected_features) >= config.elastic_net_min_features:
                break
            if feature_name not in selected_features:
                selected_features.append(feature_name)

    return selected_features


def _run_skopt_search(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: StrategyConfig,
    cv: TimeSeriesSplit,
    scoring: Callable[[Any, pd.DataFrame, pd.Series], float],
) -> ModelFitResult:
    from skopt import BayesSearchCV

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=_search_space_skopt(model_name),
        n_iter=config.bayes_iter,
        cv=cv,
        n_jobs=1,
        scoring=scoring,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(X, y)
    return ModelFitResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        search_type="bayesian",
        cv_score=float(search.best_score_),
        model_name=model_name,
        feature_names=list(X.columns),
    )


def _run_random_search(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: StrategyConfig,
    cv: TimeSeriesSplit,
    scoring: Callable[[Any, pd.DataFrame, pd.Series], float],
) -> ModelFitResult:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=_search_space_randomized(model_name),
        n_iter=config.random_search_iter,
        cv=cv,
        n_jobs=1,
        scoring=scoring,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(X, y)
    return ModelFitResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        search_type="randomized_fallback",
        cv_score=float(search.best_score_),
        model_name=model_name,
        feature_names=list(X.columns),
    )


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    X_cols: list[str],
    config: StrategyConfig,
) -> ModelFitResult:
    selected_cols = _select_elastic_net_features(train_df, X_cols, config) if model_name == "elastic_net" else X_cols
    X = train_df[selected_cols]
    y = train_df["target_return"]
    cv = TimeSeriesSplit(n_splits=config.cv_splits)
    estimator = _make_estimator(model_name, config)
    scoring = _make_cross_sectional_ic_scorer(train_df)
    try:
        return _run_skopt_search(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            config=config,
            cv=cv,
            scoring=scoring,
        )
    except ImportError:
        return _run_random_search(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            config=config,
            cv=cv,
            scoring=scoring,
        )


def score_predictions(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    residual = actual - predicted
    denom = np.square(actual - actual.mean()).sum()
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "bias": float(residual.mean()),
        "r2": float(1.0 - np.square(residual).sum() / denom) if denom > 0 else float("nan"),
        "corr": float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else float("nan"),
    }


def predict_cross_section(
    estimator: Any,
    test_df: pd.DataFrame,
    X_cols: list[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    preds = estimator.predict(test_df[X_cols])
    output_columns = ["date", "asset", "target_return", "close", "asset_vol"]
    for optional_column in ["beta_btc", "beta_eth", "market_forward_return"]:
        if optional_column in test_df.columns:
            output_columns.append(optional_column)
    out = test_df[output_columns].copy()
    out["prediction"] = preds
    return preds, out


def extract_parameter_vector(estimator: Any, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    if model_name == "elastic_net":
        values = estimator.named_steps["model"].coef_
        parameter_name = "coefficient"
    elif model_name == "xgboost":
        values = estimator.named_steps["model"].feature_importances_
        parameter_name = "feature_importance"
    elif model_name == "ddpm":
        values = estimator.named_steps["model"].feature_importances_
        parameter_name = "feature_importance"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "value": values,
            "parameter_name": parameter_name,
        }
    )
