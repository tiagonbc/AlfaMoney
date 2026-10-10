"""Short-Term Basis Reversal — Rossi, Zhang & Zhu (2025), SSRN 5250499.
Quantitativo post: "Short-Term Basis Reversal" (39 Norgate commodity markets).

Rules:
  * Each week, per commodity, pick the two nearest contracts with at least
    ``min_days`` (paper: 30) calendar days to expiry / first notice: front (1) and
    second (2).
  * Basis return over the week: b = r1 − r2 (weekly returns of the two contracts).
  * Basis returns are negatively autocorrelated week to week:
      - time-series version: hold −sign(b_{w−1}) units of the spread (long front /
        short second when last week's basis return was negative) for week w;
      - cross-sectional version: rank markets on b_{w−1}; long the spread in the
        most negative quantile, short the spread in the most positive quantile.
  * Re-select contracts weekly; hold one week; equal weight across markets.
Paper: ≈18%/yr, Sharpe 1.42 (cross-sectional). QuantReturns check: Sharpe 0.92
(30 days), 1.58 with a 60-day rule. Quantitativo: Feeder Cattle alone Sharpe 0.91.

Data: Norgate individual dated contracts (alfamoney.data.norgate.load_contract_strip).
Continuous back-adjusted series cannot be used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult

# Norgate roots for the 39 commodity markets (agri 26, energy 8, metals 5) — edit to taste
COMMODITY_ROOTS = [
    # grains / oilseeds / softs / livestock / other ags
    "ZC", "ZW", "KE", "MWE", "ZS", "ZM", "ZL", "ZO", "ZR", "RS", "CT", "KC", "SB", "CC", "OJ", "LBR",
    "LE", "HE", "GF", "DC", "CSC", "RC", "W", "C", "T", "EBM",
    # energy
    "CL", "HO", "RB", "NG", "BRN", "G", "WBS", "NGLNM",
    # metals
    "GC", "SI", "HG", "PL", "PA",
]


def _week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).last().values)


def weekly_basis_returns(strip: dict, min_days: int = 30, first_notice: pd.Series | None = None) -> pd.DataFrame:
    """Per week w (indexed by the week-end date where the position is OPENED):
    contracts chosen at w, and their next-week returns r1, r2 and basis b = r1 − r2.
    Also returns the basis return realised over the week ending at w (for the signal)."""
    close = strip["Close"].sort_index()
    expiry = strip["expiry"]
    if first_notice is not None:
        expiry = pd.concat([expiry, first_notice.reindex(expiry.index)], axis=1).min(axis=1)
    wk = _week_ends(close.index)
    rows = []
    for i in range(len(wk) - 1):
        t0, t1 = wk[i], wk[i + 1]
        live = expiry[(expiry - t0).dt.days >= min_days].sort_values().index
        live = [c for c in live if c in close.columns and np.isfinite(close.at[t0, c]) and np.isfinite(close.at[t1, c])]
        if len(live) < 2:
            continue
        c1, c2 = live[0], live[1]
        r1 = close.at[t1, c1] / close.at[t0, c1] - 1
        r2 = close.at[t1, c2] / close.at[t0, c2] - 1
        rows.append({"open": t0, "close": t1, "c1": c1, "c2": c2, "r1": r1, "r2": r2, "basis": r1 - r2})
    df = pd.DataFrame(rows).set_index("open")
    # basis return realised over the week ending at t0 (same contract-selection rule applied a week earlier)
    df["signal_basis"] = df["basis"].shift(1)
    # guard: the previous row must be the immediately preceding week
    gap = df.index.to_series().diff().dt.days
    df.loc[gap > 10, "signal_basis"] = np.nan
    return df


def run(strips: dict[str, dict], mode: str = "cross_sectional", min_days: int = 30, n_quantiles: int = 3,
        cost_per_spread: float = 0.0, first_notice: dict[str, pd.Series] | None = None) -> BacktestResult:
    """``strips``: root -> output of norgate.load_contract_strip(root)."""
    basis, sig = {}, {}
    for root, strip in strips.items():
        try:
            df = weekly_basis_returns(strip, min_days, (first_notice or {}).get(root))
        except Exception:
            continue
        if df.empty:
            continue
        idx = df["close"]
        basis[root] = pd.Series(df["basis"].values, index=idx)
        sig[root] = pd.Series(df["signal_basis"].values, index=idx)
    B, S = pd.DataFrame(basis).sort_index(), pd.DataFrame(sig).sort_index()
    if mode == "time_series":
        pos = -np.sign(S)
        pos = pos.div(pos.abs().sum(axis=1).replace(0, np.nan), axis=0)
    else:
        pct = S.rank(axis=1, pct=True)
        lo, hi = pct <= 1 / n_quantiles, pct > 1 - 1 / n_quantiles
        lw = lo.astype(float).div(lo.sum(axis=1).replace(0, np.nan), axis=0)
        sw = hi.astype(float).div(hi.sum(axis=1).replace(0, np.nan), axis=0)
        pos = lw.fillna(0) - sw.fillna(0)  # +1 = long front / short second
    pos = pos.fillna(0)
    gross = (pos * B.fillna(0)).sum(axis=1)
    # each week the contracts are re-selected: charge a full round trip on the spread
    turnover = pos.abs().sum(axis=1) * 2
    net = gross - turnover * cost_per_spread
    return BacktestResult(net, gross, pos, turnover, periods_per_year=52, meta={"basis": B, "signal": S})
