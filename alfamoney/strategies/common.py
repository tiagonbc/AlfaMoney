"""Shared helpers for the equity strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tradable_mask(panel: dict[str, pd.DataFrame], min_price: float = 5.0,
                  min_dollar_vol: float | None = None, size_pct_cut: float | None = None,
                  adv_window: int = 21) -> pd.DataFrame:
    """Point-in-time investable universe: index member, price >= min_price,
    optional ADV filter and optional removal of the smallest ``size_pct_cut``
    fraction by dollar volume (proxy for size when no market cap is available)."""
    close = panel["Close"]
    unadj = panel.get("UnadjClose", close)
    m = panel.get("member", close.notna())
    m = m.reindex_like(close).fillna(False).astype(bool)
    m &= unadj >= min_price
    m &= close.notna()
    dv = (unadj * panel["Volume"]).rolling(adv_window, min_periods=adv_window // 2).mean()
    if min_dollar_vol is not None:
        m &= dv >= min_dollar_vol
    if size_pct_cut:
        m &= dv.where(m).rank(axis=1, pct=True) > size_pct_cut
    return m


def monthly(df: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    g = df.groupby([df.index.year, df.index.month])
    out = getattr(g, how)()
    out.index = df.index.to_series().groupby([df.index.year, df.index.month]).last().values
    return out


def monthly_returns(close: pd.DataFrame) -> pd.DataFrame:
    return monthly(close).pct_change(fill_method=None)


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey–West t-statistic of the mean."""
    x = x.dropna().values
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n)
