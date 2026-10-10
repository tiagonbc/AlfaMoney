"""Modern Statistical Arbitrage — Quantitativo post built on
Du, Walter & Ulrich (2026), "Cross-Market Alpha: Testing Short-Term Trading Factors
in the U.S. Market via Double-Selection LASSO", arXiv:2601.06499.

The paper tests the Guotai Junan "Alpha191" library on S&P 500 stocks (2002–2022)
against 153 US factors with double-selection LASSO; 17 signals survive (mostly
volume-price interaction and short-term mean reversion).

Implemented here:
  * GTJA Alpha#46 exactly as in the post:
        a46 = (MA3 + MA6 + MA12 + MA24) / (4·Close)
    demeaned cross-sectionally each day (index-membership mask), positive weight
    for stocks far below their blended average (short-horizon reversal), daily
    rebalance, unit gross exposure.  Quantitativo: alone Sharpe < 0.5, 55.7% DD.
  * A few other well-documented GTJA short-term alphas, and an equal-risk
    combiner ``combine`` to form a multi-signal book (the post combines 17 signals
    → Sharpe 1.15–1.76 over 6 universes). The exact 17-signal list should be
    taken from the paper's table; add them to ``ALPHAS``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, run_weights
from .common import tradable_mask


def _rank(df):  # cross-sectional pct rank
    return df.rank(axis=1, pct=True)


def alpha46(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
    c = p["Close"]
    return (c.rolling(3).mean() + c.rolling(6).mean() + c.rolling(12).mean() + c.rolling(24).mean()) / (4 * c)


def alpha14(p):   # CLOSE − DELAY(CLOSE,5), sign-flipped as reversal
    c = p["Close"]
    return -(c - c.shift(5)) / c.shift(5)


def alpha18(p):   # CLOSE / DELAY(CLOSE,5), reversal
    c = p["Close"]
    return -(c / c.shift(5))


def alpha31(p):   # (CLOSE − MEAN(CLOSE,12)) / MEAN(CLOSE,12), reversal
    c = p["Close"]
    m = c.rolling(12).mean()
    return -(c - m) / m


def alpha_vol_price_corr(p):  # −CORR(RANK(VOLUME), RANK(CLOSE), 10): volume-price interaction
    rv, rc = _rank(p["Volume"]), _rank(p["Close"])
    return -rv.rolling(10).corr(rc)


ALPHAS: dict[str, Callable] = {"a46": alpha46, "a14": alpha14, "a18": alpha18, "a31": alpha31,
                               "vp_corr": alpha_vol_price_corr}


def signal_weights(sig: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    s = sig.where(mask)
    d = s.sub(s.mean(axis=1), axis=0)
    return d.div(d.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def run_signal(panel: dict[str, pd.DataFrame], name: str = "a46", cost_bps: float = 0.0,
               min_price: float = 5.0) -> BacktestResult:
    mask = tradable_mask(panel, min_price=min_price)
    w = signal_weights(ALPHAS[name](panel), mask)
    return run_weights(w, panel["Close"].pct_change(fill_method=None), cost_bps=cost_bps)


def combine(panel: dict[str, pd.DataFrame], names=None, vol_lookback: int = 126, cost_bps: float = 0.0,
            min_price: float = 5.0) -> BacktestResult:
    """Equal-risk blend of single-signal books (weights netted before trading)."""
    names = list(names or ALPHAS)
    mask = tradable_mask(panel, min_price=min_price)
    ret = panel["Close"].pct_change(fill_method=None)
    books, rets = {}, {}
    for n in names:
        w = signal_weights(ALPHAS[n](panel), mask)
        books[n] = w
        rets[n] = (w.shift(1) * ret).sum(axis=1)
    R = pd.DataFrame(rets)
    inv_vol = 1 / R.rolling(vol_lookback, min_periods=vol_lookback // 2).std()
    k = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1)  # lagged risk weights
    total = sum(books[n].mul(k[n], axis=0) for n in names).fillna(0)
    total = total.div(total.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    res = run_weights(total, ret, cost_bps=cost_bps)
    res.meta = {"component_returns": R}
    return res
