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


def write_amibroker_export(folder, n_assets: int = 80, n_days: int = 252 * 6, seed: int = 0) -> None:
    """Write a fake AmiBroker/Norgate export with the exact layout produced by
    amibroker/AlfaMoney_Export_Norgate.afl (for tests and dry runs)."""
    from pathlib import Path

    root = Path(folder)
    eq = equity_panel(n_assets, n_days, seed)
    (root / "equities").mkdir(parents=True, exist_ok=True)
    for sym in eq["Close"].columns:
        m = eq["member"][sym].astype(int)
        df = pd.DataFrame({"Open": eq["Open"][sym], "High": eq["High"][sym], "Low": eq["Low"][sym],
                           "Close": eq["Close"][sym], "Volume": eq["Volume"][sym].round(),
                           "UnadjClose": eq["UnadjClose"][sym],
                           "InSPX": m, "InRUI": m, "InRUA": m, "InRUT": 0, "InNDX": 0})
        df.index.name = "Date"
        df.to_csv(root / "equities" / f"{sym}.csv", float_format="%.6f")
    fut = futures_panel(8, n_days, seed + 1, start="2015-01-02")
    (root / "futures").mkdir(parents=True, exist_ok=True)
    (root / "etfs").mkdir(parents=True, exist_ok=True)
    names = ["ES", "ZN", "CL", "GC", "ZC", "6E", "ZB", "NG"]
    for col, root_sym in zip(fut["Close"].columns, names):
        df = pd.DataFrame({k: fut[k][col] for k in ("Open", "High", "Low", "Close", "Volume")})
        df["OpenInt"] = 1000
        df["UnadjClose"] = df["Close"]
        df.index.name = "Date"
        df.to_csv(root / "futures" / f"&{root_sym}_CCB.csv", float_format="%.6f")
    for k, etf in enumerate(["SPY", "TLT", "GLD", "DBC", "UUP", "BIL"]):
        col = fut["Close"].columns[k]
        df = pd.DataFrame({f: fut[f][col] for f in ("Open", "High", "Low", "Close", "Volume")})
        df["OpenInt"] = 0
        df["UnadjClose"] = df["Close"]
        df.index.name = "Date"
        df.to_csv(root / "etfs" / f"{etf}.csv", float_format="%.6f")
    # dated contracts for one commodity root ("ZC"), quarterly expiries
    rng = np.random.default_rng(seed + 2)
    dates = fut["Close"].index
    spot = fut["Close"][fut["Close"].columns[4]].values
    for e in pd.date_range(dates[0] + pd.Timedelta(days=40), dates[-1] + pd.Timedelta(days=200), freq="QS"):
        code = f"ZC-{e.year}{'FGHJKMNQUVXZ'[e.month - 1]}"
        live = (dates <= e - pd.Timedelta(days=5)) & (dates >= e - pd.Timedelta(days=400))
        if not live.any():
            continue
        tau = (e - dates[live]).days / 365
        px = spot[live] * np.exp(0.03 * tau + rng.normal(0, 0.002, live.sum()))
        df = pd.DataFrame({"Open": px, "High": px * 1.002, "Low": px * 0.998, "Close": px,
                           "Volume": 1000, "OpenInt": 5000, "UnadjClose": px}, index=dates[live])
        df.index.name = "Date"
        df.to_csv(root / "futures" / f"{code}.csv", float_format="%.6f")
