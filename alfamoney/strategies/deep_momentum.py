"""Deep-learning momentum classifiers.

1) Takeuchi & Lee (2013), "Applying Deep Learning to Enhance Momentum Trading
   Strategies in Stocks" (Stanford CS229).  Quantitativo post: "The Bitter Lesson".
   * 33 inputs (12 monthly cumulative returns, 20 daily cumulative returns, January
     dummy), z-scored cross-sectionally (ml_common.takeuchi_lee_features).
   * Label: next-month return above the cross-sectional median (binary).
   * Network: paper pretrains a 33-40-4 stacked RBM autoencoder then a 4-50-2
     classifier; Quantitativo (and this code) trains the 33-40-4-50-2 network
     end-to-end without RBM pretraining.
   * Monthly long top decile / short bottom decile of predicted probability.
   Paper: ≈45.9% annualised 1990-2009, ≈53% accuracy. Quantitativo: ≈52% accuracy,
   "a far cry" from the paper.

2) Han (2022), "Bimodal Characteristic Returns and Predictability Enhancement via
   Machine Learning", Management Science 68(10).  Quantitativo post: "Uncertainty"
   (also covers Han & Qin 2025, "Bimodality Everywhere", international).
   * Same style of momentum inputs + size-decile dummies.
   * Multi-class target: next-month return decile (10 classes).
   * Rank by EXPECTED return Σ_k p_k·μ_k (μ_k = mean training return of class k),
     not by the most likely class — winners and losers both have bimodal outcomes.
   * Monthly decile long-short (equal- or value-weighted).
   Paper: VW 35%/yr Sharpe 1.66; EW ≈50%/yr Sharpe > 2.8. Quantitativo: "well below".

The MLP is sklearn's (no GPU needed). Swap ``_mlp`` for a torch model for speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import monthly, tradable_mask
from .ml_common import (long_short_monthly, monthly_result, predict_walk_forward, stack_features,
                        takeuchi_lee_features)


def _mlp(hidden=(40, 4, 50), seed=0, max_iter=200):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4,
                         batch_size=512, learning_rate_init=1e-3, max_iter=max_iter,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                         random_state=seed)


def _prep(panel, min_price):
    close = panel["Close"]
    mask_m = monthly(tradable_mask(panel, min_price=min_price).astype(float)).astype(bool)
    ret_next = monthly(close).pct_change(fill_method=None).shift(-1)
    return close, mask_m, ret_next


def run_takeuchi_lee(panel: dict[str, pd.DataFrame], first_test_year: int, train_years: int = 15,
                     retrain_every_years: int = 1, min_price: float = 5.0, cost_bps: float = 0.0,
                     hidden=(40, 4, 50), n_seeds: int = 1):
    close, mask_m, ret_next = _prep(panel, min_price)
    X = stack_features(takeuchi_lee_features(close), mask_m)
    above = ret_next.where(mask_m).gt(ret_next.where(mask_m).median(axis=1), axis=0).astype(float)
    above = above.where(ret_next.where(mask_m).notna())
    y = above.stack().reindex(X.index)

    def fit_predict(Xtr, ytr, Xte):
        ps = []
        for s in range(n_seeds):
            m = _mlp(hidden, seed=s).fit(Xtr.values, ytr.values.astype(int))
            ps.append(m.predict_proba(Xte.values)[:, 1])
        return np.mean(ps, axis=0)

    prob = predict_walk_forward(X, y, fit_predict, train_years, retrain_every_years, first_test_year)
    score = prob.unstack()
    r, w = long_short_monthly(score, ret_next, 10, gross=2.0)
    acc = ((prob > 0.5).astype(float) == y.reindex(prob.index)).mean()
    return monthly_result(r, w, ret_next.index, cost_bps, {"score": score, "accuracy": acc})


def size_dummies(panel: dict[str, pd.DataFrame], mask_m: pd.DataFrame, mktcap: pd.DataFrame | None = None,
                 n: int = 10) -> dict[str, pd.DataFrame]:
    """Size-decile dummies. Uses market cap (e.g. from P123 'MktCap') when given,
    otherwise 63-day average dollar volume as a proxy."""
    if mktcap is None:
        dv = (panel.get("UnadjClose", panel["Close"]) * panel["Volume"]).rolling(63, min_periods=20).mean()
        mktcap = monthly(dv)
    dec = np.floor(mktcap.where(mask_m).rank(axis=1, pct=True).mul(n).clip(upper=n - 1e-9))
    return {f"size{k}": (dec == k).astype(float).where(dec.notna()) for k in range(n)}


def run_bimodal(panel: dict[str, pd.DataFrame], first_test_year: int, train_years: int | None = None,
                retrain_every_years: int = 1, min_price: float = 5.0, cost_bps: float = 0.0,
                mktcap: pd.DataFrame | None = None, value_weighted: bool = False,
                hidden=(512, 256, 128, 64, 32), rank_by: str = "expected"):
    close, mask_m, ret_next = _prep(panel, min_price)
    feats = takeuchi_lee_features(close)
    feats.update(size_dummies(panel, mask_m, mktcap))
    X = stack_features(feats, mask_m)
    rn = ret_next.where(mask_m)
    cls = np.floor(rn.rank(axis=1, pct=True).mul(10).clip(upper=9.999))
    y = cls.stack().reindex(X.index)
    r_stack = rn.stack().reindex(X.index)

    def fit_predict(Xtr, ytr, Xte):
        m = _mlp(hidden).fit(Xtr.values, ytr.values.astype(int))
        P = m.predict_proba(Xte.values)
        if rank_by == "argmax":
            return m.classes_[P.argmax(axis=1)].astype(float)
        mu = r_stack.reindex(Xtr.index).groupby(ytr.values).mean().reindex(m.classes_).values
        return P @ mu

    pred = predict_walk_forward(X, y, fit_predict, train_years, retrain_every_years, first_test_year)
    score = pred.unstack()
    if value_weighted:
        cap = mktcap if mktcap is not None else monthly(
            (panel.get("UnadjClose", close) * panel["Volume"]).rolling(63, min_periods=20).mean())
        vs = cap.reindex_like(score)
    else:
        vs = None
    r, w = long_short_monthly(score, ret_next, 10, gross=2.0, vol_scale=vs)
    return monthly_result(r, w, ret_next.index, cost_bps, {"score": score})
