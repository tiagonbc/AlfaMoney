"""Volume Shocks and Overnight Returns — Cartea, Cucuringu, Jin & Wilson (2025), SSRN 5156605.
Quantitativo post: "Volume Shocks and Overnight Returns" (Norgate Russell 3000).

Rules:
  * Volume shock: VS_t = V_t / EMA(V, half-life 60 days)_{t−1} − 1.
  * High VS predicts a positive close-to-open (overnight) return; no effect intraday.
  * Portfolio: sort on VS at the close of t; buy at the close, sell at the next open.
    Paper sorts into quintiles (long-short); Quantitativo uses deciles, top decile
    long-only, equal weight, MOC orders.
Caveat (as in the paper/post): the full-day volume is only known after the closing
auction. Quantitativo uses cumulative volume up to 15:45; with daily Norgate bars the
backtest below uses the full-day volume, i.e. it is the paper's (non-tradable)
version unless you pass ``volume`` built from 15:45 snapshots.
Quantitativo: performance collapsed after commissions + a 10%-of-ADV cap on a broad
universe; a gap-prone (biotech) universe gave 36.1%/yr, Sharpe 1.52.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from .common import tradable_mask


def volume_shock(volume: pd.DataFrame, halflife: int = 60) -> pd.DataFrame:
    base = volume.ewm(halflife=halflife, min_periods=halflife).mean().shift(1)
    return volume / base - 1


def run(panel: dict[str, pd.DataFrame], n_quantiles: int = 10, long_only: bool = True,
        min_price: float = 5.0, min_dollar_vol: float | None = 1e6, cost_bps: float = 0.0,
        volume: pd.DataFrame | None = None, universe_filter: pd.DataFrame | None = None) -> BacktestResult:
    close, open_ = panel["Close"], panel["Open"]
    vs = volume_shock(volume if volume is not None else panel["Volume"])
    mask = tradable_mask(panel, min_price=min_price, min_dollar_vol=min_dollar_vol)
    if universe_filter is not None:
        mask &= universe_filter.reindex_like(mask).fillna(False).astype(bool)
    s = vs.where(mask)
    pct = s.rank(axis=1, pct=True)
    top = (pct > 1 - 1 / n_quantiles).astype(float)
    lw = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if long_only:
        w = lw
    else:
        bot = (pct <= 1 / n_quantiles).astype(float)
        w = lw - bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    overnight = open_.shift(-1) / close - 1        # close t → open t+1
    gross = (w * overnight.fillna(0)).sum(axis=1)
    # fully in and out every night: turnover = 2 × gross exposure
    turnover = 2 * w.abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    # attribute the return to the day it is realised (t+1)
    net, gross = net.shift(1).dropna(), gross.shift(1).dropna()
    return BacktestResult(net, gross, w.shift(1).loc[net.index], turnover.shift(1).loc[net.index],
                          meta={"vs": vs})
