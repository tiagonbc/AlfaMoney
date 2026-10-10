"""The Unintended Consequences of Rebalancing — Harvey, Mazzoleni & Melone (2025),
NBER w33554 / SSRN 5122748.  Quantitativo replication: Sharpe ≈ 0.94.

Mechanics:
  * Reference 60/40 portfolio of E-mini S&P 500 (ES) and 10-year T-note (ZN/TY) futures,
    drifting with returns since its last rebalance.
  * Threshold signal: for each δ on a grid (0.1% … 2.5%), simulate a portfolio that
    rebalances to 60/40 whenever |w_eq − 60%| > δ. When a rebalance is triggered at
    the close of t, the predicted flow is −sign(w_eq − 0.6) in equities over t+1.
    The signal is the average of these triggers across δ.
  * Calendar signal: a 60/40 portfolio rebalanced at month end. During the last
    ``calendar_days`` trading days of the month (paper baseline ≈ 5) the predicted
    flow is −sign(w_eq − 0.6) of the month-to-date drift.
  * Trade: equity-minus-bond spread position = predicted flow (short ES / long ZN when
    equities are overweight), held one day; legs vol-scaled to equal risk.
Paper: ≈ −17 bp next-day equity return when overweight; Sharpe > 1 (1997–2023).

Data: Norgate continuous futures "&ES_CCB" and "&ZN_CCB" (or SPY/IEF total-return ETFs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, ewm_vol

DELTAS = tuple(np.round(np.arange(0.001, 0.0251, 0.001), 4))


def _drift_weights(r_eq: np.ndarray, r_bd: np.ndarray, delta: float | None, month_end: np.ndarray | None,
                   target: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Return (equity weight at each close after drift, rebalance-trigger flag)."""
    n = len(r_eq)
    w = np.empty(n)
    trig = np.zeros(n, dtype=bool)
    we = target
    for t in range(n):
        ve, vb = we * (1 + r_eq[t]), (1 - we) * (1 + r_bd[t])
        we = ve / (ve + vb)
        w[t] = we
        if delta is not None and abs(we - target) > delta:
            trig[t] = True
            we = target
        elif month_end is not None and month_end[t]:
            trig[t] = True
            we = target
    return w, trig


def signals(r_eq: pd.Series, r_bd: pd.Series, deltas=DELTAS, calendar_days: int = 5,
            target: float = 0.6) -> pd.DataFrame:
    idx = r_eq.index
    re, rb = r_eq.fillna(0).values, r_bd.fillna(0).values
    thr = np.zeros(len(idx))
    for d in deltas:
        w, trig = _drift_weights(re, rb, d, None, target)
        thr += np.where(trig, -np.sign(w - target), 0.0)
    thr /= len(deltas)
    s = pd.Series(idx, index=idx)
    me = s.groupby([idx.year, idx.month]).transform("max").values == idx.values
    w_cal, _ = _drift_weights(re, rb, None, me, target)
    # trading days remaining until month end (0 on the last day)
    rank_from_end = s.groupby([idx.year, idx.month]).cumcount(ascending=False).values
    in_window = rank_from_end < calendar_days
    cal = np.where(in_window, -np.sign(w_cal - target), 0.0)
    return pd.DataFrame({"threshold": thr, "calendar": cal, "w_calendar": w_cal}, index=idx)


def run(close_eq: pd.Series, close_bd: pd.Series, combine: str = "average", calendar_days: int = 5,
        vol_span: int = 60, target_vol: float = 0.10, cost_bps: float = 0.0) -> BacktestResult:
    px = pd.concat([close_eq, close_bd], axis=1, keys=["eq", "bd"]).dropna()
    r = px.pct_change(fill_method=None).dropna()
    sig = signals(r["eq"], r["bd"], calendar_days=calendar_days)
    if combine == "threshold":
        s = sig["threshold"]
    elif combine == "calendar":
        s = sig["calendar"]
    else:
        s = (sig["threshold"] + sig["calendar"]) / 2
    vol = ewm_vol(r, vol_span)
    # spread: +1 = long equity / short bonds, each leg sized to target_vol/2
    w = pd.DataFrame({"eq": s * (target_vol / 2) / vol["eq"], "bd": -s * (target_vol / 2) / vol["bd"]})
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0)
    held = w.shift(1).fillna(0)        # signal at close t, held over t+1
    gross = (held * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).shift(1).fillna(0)
    net = gross - turnover * cost_bps / 1e4
    return BacktestResult(net, gross, held, turnover, meta={"signals": sig})
