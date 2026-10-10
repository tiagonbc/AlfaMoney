"""Synthetic market data with the same layout as the Norgate panels.

Used only for unit tests / smoke runs where the real data vendors are not
available. Results on synthetic data are meaningless as performance evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_panel(n_assets: int = 200, n_days: int = 1500, seed: int = 0, start: str = "2015-01-02") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    cols = [f"S{i:04d}" for i in range(n_assets)]
    mkt = rng.normal(0.0003, 0.011, n_days)
    beta = rng.uniform(0.5, 1.5, n_assets)
    idio_vol = rng.uniform(0.01, 0.03, n_assets)
    eps = rng.normal(0, 1, (n_days, n_assets)) * idio_vol
    # small negative autocorrelation (short-term reversal) so tests have structure
    eps[1:] -= 0.05 * eps[:-1]
    ret = mkt[:, None] * beta[None, :] + eps
    close = 50 * np.exp(np.cumsum(ret, axis=0))
    gap = rng.normal(0, 0.004, (n_days, n_assets))
    open_ = close / np.exp(ret) * np.exp(gap)
    rng_hl = np.abs(rng.normal(0, 0.01, (n_days, n_assets)))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    vol = rng.lognormal(13, 0.5, (n_days, n_assets))
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)
    member = mk(np.ones((n_days, n_assets), dtype=bool))
    # some assets enter/leave the universe
    for j in rng.choice(n_assets, n_assets // 10, replace=False):
        cut = rng.integers(100, n_days - 100)
        member.iloc[:cut, j] = False
    return {"Open": mk(open_), "High": mk(high), "Low": mk(low), "Close": mk(close),
            "Volume": mk(vol), "member": member, "UnadjClose": mk(close)}


def futures_panel(n_assets: int = 30, n_days: int = 3000, seed: int = 1, start: str = "2008-01-02") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    cols = [f"F{i:02d}" for i in range(n_assets)]
    vol = rng.uniform(0.005, 0.02, n_assets)
    # persistent drift regimes give a little trend structure
    drift = np.zeros((n_days, n_assets))
    state = rng.normal(0, 1, n_assets)
    for t in range(n_days):
        state = 0.995 * state + 0.1 * rng.normal(0, 1, n_assets)
        drift[t] = state * vol * 0.03
    ret = drift + rng.normal(0, 1, (n_days, n_assets)) * vol
    close = 100 + np.cumsum(ret * 100, axis=0)
    close = close - close.min(axis=0) + 50
    open_ = np.vstack([close[:1], close[:-1]]) * (1 + rng.normal(0, 0.002, (n_days, n_assets)))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    mk = lambda a: pd.DataFrame(a, index=dates, columns=cols)
    return {"Open": mk(open_), "High": mk(high), "Low": mk(low), "Close": mk(close),
            "Volume": mk(rng.lognormal(10, 0.3, (n_days, n_assets)))}


def intraday_bars(n_days: int = 500, seed: int = 2, start: str = "2022-01-03", minutes: int = 390) -> pd.DataFrame:
    """1-minute OHLCV bars, US RTH 09:30–16:00 New York, single instrument."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    frames, px = [], 4000.0
    for d in days:
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=31), periods=minutes, freq="1min")
        px *= np.exp(rng.normal(0, 0.006))  # overnight gap
        r = rng.normal(0, 0.0006, minutes)
        c = px * np.exp(np.cumsum(r))
        o = np.concatenate([[px], c[:-1]])
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0002, minutes)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0002, minutes)))
        frames.append(pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c,
                                    "Volume": rng.lognormal(8, 0.4, minutes)}, index=idx))
        px = c[-1]
    return pd.concat(frames)
