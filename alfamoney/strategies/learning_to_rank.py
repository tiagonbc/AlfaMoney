"""Building Cross-Sectional Systematic Strategies by Learning to Rank
Poh, Lim, Zohren & Roberts (2021), J. Financial Data Science 3(2). arXiv:2012.07149.
Quantitativo post: "Learning to Rank".

Paper: CRSP NYSE stocks 1980-2019; 21 momentum features (see ml_common.ltr_features);
label = next-month return decile; LambdaMART (XGBoost); monthly long top decile /
short bottom decile, each position vol-scaled to 15% annualised; retrain every
5 years on an expanding window.  Headline: Sharpe ≈ 2.16 (vs 0.70 best heuristic).

Quantitativo variant (default here): Russell 3000 current+past members (Norgate),
equal weight, 150% gross, yearly retrain on a sliding 15-year window,
learning rate 0.1, 200 trees.  Reported ≈18%/yr market-neutral, 13% max DD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult  # noqa: F401  (return type)
from .common import monthly, tradable_mask
from .ml_common import long_short_monthly, ltr_features, monthly_result, predict_walk_forward, stack_features


def _fit_predict_factory(n_estimators=200, learning_rate=0.1, max_depth=6, objective="rank:ndcg"):
    def fit_predict(Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        Xtr = Xtr.sort_index()
        ytr = ytr.reindex(Xtr.index)
        groups = Xtr.groupby(level=0).size().values
        model = xgb.XGBRanker(objective=objective, n_estimators=n_estimators,
                              learning_rate=learning_rate, max_depth=max_depth,
                              tree_method="hist", lambdarank_pair_method="topk", n_jobs=-1)
        model.fit(Xtr.values, ytr.values.astype(int), group=groups)
        return model.predict(Xte.values)
    return fit_predict


def run(panel: dict[str, pd.DataFrame], first_test_year: int | None = None, variant: str = "quantitativo",
        min_price: float = 5.0, cost_bps: float = 0.0) -> BacktestResult:
    close = panel["Close"]
    mask_m = monthly(tradable_mask(panel, min_price=min_price).astype(float)).astype(bool)
    feats = ltr_features(close)
    X = stack_features(feats, mask_m)
    ret_next = monthly(close).pct_change(fill_method=None).shift(-1)
    # relevance label: next-month return decile 0..9
    lab = np.floor(ret_next.where(mask_m).rank(axis=1, pct=True).mul(10).clip(upper=9.999))
    y = lab.stack().reindex(X.index)
    if variant == "paper":
        train_years, every, fp, gross = None, 5, _fit_predict_factory(100, 0.3), 2.0
    else:
        train_years, every, fp, gross = 15, 1, _fit_predict_factory(200, 0.1), 1.5
    dates = X.index.get_level_values(0).unique()
    if first_test_year is None:
        first_test_year = dates[0].year + (train_years or 10)
    pred = predict_walk_forward(X, y, fp, train_years, every, first_test_year)
    score = pred.unstack()
    vol_scale = None
    if variant == "paper":
        sig = close.pct_change(fill_method=None).ewm(span=63, min_periods=40).std() * np.sqrt(252)
        vol_scale = 0.15 / monthly(sig)
    r, w = long_short_monthly(score, ret_next, 10, gross=gross, vol_scale=vol_scale)
    return monthly_result(r, w, ret_next.index, cost_bps, {"score": score})
