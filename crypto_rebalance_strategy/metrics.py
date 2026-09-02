from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _annualization_factor(index: pd.Series) -> float:
    inferred = pd.infer_freq(index)
    if inferred in {"D", "B"}:
        return 365.0
    return 365.0


def compute_drawdown(nav: pd.Series) -> pd.Series:
    running_max = nav.cummax()
    return nav / running_max - 1.0


def summarize_portfolio(returns: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("No portfolio returns to summarize")
    ann = _annualization_factor(returns.index)
    nav = (1.0 + returns).cumprod()
    drawdown = compute_drawdown(nav)
    mean_ret = returns.mean()
    vol = returns.std(ddof=0)
    downside = returns[returns < 0].std(ddof=0)
    total_return = nav.iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (ann / len(returns)) - 1.0
    sharpe = np.sqrt(ann) * mean_ret / vol if vol > 0 else np.nan
    sortino = np.sqrt(ann) * mean_ret / downside if downside and not np.isnan(downside) and downside > 0 else np.nan
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    hit_ratio = float((returns > 0).mean())
    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(vol * np.sqrt(ann)),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar_ratio": float(calmar),
        "hit_ratio": hit_ratio,
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }
