"""Defense First: A Multi-Asset Tactical Model for Adaptive Downside Protection
Thomas Carlson (2025), SSRN 5334772.  Quantitativo post: "From Defense to Offense".

Rules (monthly, at month end):
  * Defensive sleeve: TLT, GLD, DBC, UUP.
  * Momentum = average of 1-, 3-, 6- and 12-month total returns.
  * Rank the four assets by momentum and assign 40% / 30% / 20% / 10%.
  * Absolute filter: any asset whose momentum is below cash (BIL / 3-month T-bill)
    passes its slot to the fallback asset (SPY).
Paper (1986–2025, with pre-ETF proxies): 10.9%/yr, Sharpe 0.89, max DD ≈ 15%.
Quantitativo: Sharpe 0.95 (vs 0.64 S&P 500).

Data: Norgate ETFs with TOTALRETURN adjustment. For history before the ETFs launched
pass proxy series in ``prices`` (e.g. long-Treasury index for TLT, gold spot for
GLD, a commodity index for DBC, the dollar index for UUP, T-bill index for BIL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from .common import monthly

DEFENSIVE = ("TLT", "GLD", "DBC", "UUP")
FALLBACK, CASH = "SPY", "BIL"
SLOTS = (0.40, 0.30, 0.20, 0.10)


def load_norgate(start: str = "2000-01-01") -> pd.DataFrame:
    from ..data import norgate

    return pd.DataFrame({s: norgate.price_frame(s, start)["Close"] for s in (*DEFENSIVE, FALLBACK, CASH)})


def momentum(me: pd.DataFrame) -> pd.DataFrame:
    return sum(me / me.shift(k) - 1 for k in (1, 3, 6, 12)) / 4


def target_weights(prices: pd.DataFrame, defensive=DEFENSIVE, fallback=FALLBACK, cash=CASH,
                   slots=SLOTS) -> pd.DataFrame:
    me = monthly(prices)
    mom = momentum(me)
    rows = {}
    for t, m in mom.iterrows():
        d = m[list(defensive)].dropna()
        if len(d) < len(defensive) or pd.isna(m.get(cash)):
            continue
        w = pd.Series(0.0, index=prices.columns)
        for slot, asset in zip(slots, d.sort_values(ascending=False).index):
            if d[asset] > m[cash]:
                w[asset] += slot
            else:
                w[fallback] += slot
        rows[t] = w
    return pd.DataFrame(rows).T


def run(prices: pd.DataFrame, cost_bps: float = 0.0, **kw) -> BacktestResult:
    wm = target_weights(prices, **kw)
    ret = prices.pct_change(fill_method=None)
    # rebalance at month-end close; weights drift-free (monthly re-set)
    w = wm.reindex(ret.index).ffill()
    held = w.shift(1).fillna(0)
    gross = (held * ret.fillna(0)).sum(axis=1)
    turnover = wm.diff().abs().sum(axis=1).reindex(ret.index).fillna(0).shift(1).fillna(0)
    net = gross - turnover * cost_bps / 1e4
    first = wm.index.min()
    sl = net.index > first
    return BacktestResult(net[sl], gross[sl], held[sl], turnover[sl], meta={"monthly_weights": wm})
