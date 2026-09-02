from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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
from diffusion_tf.tabular_ddpm import ConditionalTabularDDPMRegressor


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


def _run_skopt_search(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: StrategyConfig,
    cv: TimeSeriesSplit,
) -> ModelFitResult:
    from skopt import BayesSearchCV

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=_search_space_skopt(model_name),
        n_iter=config.bayes_iter,
        cv=cv,
        n_jobs=1,
        scoring="neg_mean_squared_error",
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
    )


def _run_random_search(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: StrategyConfig,
    cv: TimeSeriesSplit,
) -> ModelFitResult:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=_search_space_randomized(model_name),
        n_iter=config.random_search_iter,
        cv=cv,
        n_jobs=1,
        scoring="neg_mean_squared_error",
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
    )


def fit_model(
    model_name: str,
    train_df: pd.DataFrame,
    X_cols: list[str],
    config: StrategyConfig,
) -> ModelFitResult:
    X = train_df[X_cols]
    y = train_df["target_return"]
    cv = TimeSeriesSplit(n_splits=config.cv_splits)
    estimator = _make_estimator(model_name, config)
    try:
        return _run_skopt_search(model_name=model_name, estimator=estimator, X=X, y=y, config=config, cv=cv)
    except ImportError:
        return _run_random_search(model_name=model_name, estimator=estimator, X=X, y=y, config=config, cv=cv)


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
    out = test_df[["date", "asset", "target_return", "close", "asset_vol", "beta_btc", "beta_eth"]].copy()
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
