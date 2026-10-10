"""Vectorised portfolio backtesting on wide panels.

Convention: ``weights`` at date t are decided with information up to the close
of t and earn the return from t to t+1 (close-to-close) unless the caller passes
an explicit ``returns`` frame aligned differently (e.g. open-to-close).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    returns: pd.Series                     # net daily (or per-period) returns
    gross: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    periods_per_year: int = TRADING_DAYS
    meta: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod()

    def stats(self) -> pd.Series:
        return performance_stats(self.returns, self.periods_per_year, self.turnover, self.weights)


def run_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    lag: int = 1,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    """Apply target ``weights`` to asset ``returns``.

    lag=1: weights formed at close t earn returns of t+1 (no look-ahead).
    lag=0: the caller already aligned weights with the return they earn.
    cost_bps is charged per unit of one-way turnover.
    """
    returns = returns.reindex_like(weights).astype(float)
    w = weights.fillna(0.0).shift(lag).fillna(0.0) if lag else weights.fillna(0.0)
    gross = (w * returns.fillna(0.0)).sum(axis=1)
    # turnover measured against drifted previous weights
    prev = w.shift(1).fillna(0.0)
    drift = prev * (1 + returns.shift(1).fillna(0.0))
    denom = (1 + (prev * returns.shift(1).fillna(0.0)).sum(axis=1)).replace(0, np.nan)
    drift = drift.div(denom, axis=0).fillna(0.0)
    turnover = (w - drift).abs().sum(axis=1)
    net = gross - turnover * cost_bps / 1e4
    return BacktestResult(net, gross, w, turnover, periods_per_year)


def run_long_only_signals(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    ranking: pd.DataFrame | None,
    returns: pd.DataFrame,
    max_positions: int = 10,
    cost_bps: float = 0.0,
    max_hold: int | None = None,
) -> BacktestResult:
    """Event-driven equal-slot long book (for swing / mean-reversion systems).

    At each close, open positions whose ``exits`` fire (or ``max_hold`` reached)
    are closed; free slots are filled with ``entries`` ranked by ``ranking``
    (higher = better). Each position has weight 1/max_positions; trades execute
    at the close of the signal bar.
    """
    dates, cols = entries.index, entries.columns
    E, X = entries.fillna(False).values, exits.fillna(False).values
    R = ranking.reindex_like(entries).values if ranking is not None else np.zeros(E.shape)
    W = np.zeros(E.shape)
    held: dict[int, int] = {}
    for t in range(len(dates)):
        for j in list(held):
            held[j] += 1
            if X[t, j] or (max_hold is not None and held[j] >= max_hold):
                del held[j]
        free = max_positions - len(held)
        if free > 0:
            cand = [j for j in np.flatnonzero(E[t]) if j not in held and np.isfinite(R[t, j])]
            cand.sort(key=lambda j: -R[t, j])
            for j in cand[:free]:
                held[j] = 0
        for j in held:
            W[t, j] = 1.0 / max_positions
    weights = pd.DataFrame(W, index=dates, columns=cols)
    return run_weights(weights, returns, cost_bps=cost_bps, lag=1)


# --------------------------------------------------------------------------- portfolio construction helpers

def cross_sectional_rank(signal: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    s = signal.where(mask) if mask is not None else signal
    return s.rank(axis=1, pct=True)


def quantile_portfolio(
    signal: pd.DataFrame,
    n_quantiles: int = 10,
    mask: pd.DataFrame | None = None,
    long_short: bool = True,
    weighting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long top quantile, short bottom quantile (equal or ``weighting``-weighted).

    Each leg sums to 1 (so long-short is $1 long / $1 short).
    """
    s = signal.where(mask) if mask is not None else signal
    pct = s.rank(axis=1, pct=True)
    top = pct > 1 - 1 / n_quantiles
    bot = pct <= 1 / n_quantiles
    base = weighting.reindex_like(s).where(s.notna()) if weighting is not None else s.notna().astype(float)
    lw = base.where(top, 0.0)
    lw = lw.div(lw.sum(axis=1).replace(0, np.nan), axis=0)
    if not long_short:
        return lw.fillna(0.0)
    sw = base.where(bot, 0.0)
    sw = sw.div(sw.sum(axis=1).replace(0, np.nan), axis=0)
    return (lw.fillna(0.0) - sw.fillna(0.0))


def hold_rebalance(weights: pd.DataFrame, rebalance_dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    """Keep target weights constant between rebalance dates (no intra-period drift)."""
    w = weights.where(weights.index.isin(rebalance_dates)).ffill()
    return w.fillna(0.0)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).last().values)


def vol_target(returns: pd.Series, target: float = 0.10, lookback: int = 60,
               periods_per_year: int = TRADING_DAYS, max_leverage: float = 5.0) -> pd.Series:
    """Scale a return stream ex-ante to an annualised vol target (lagged by one period)."""
    vol = returns.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(periods_per_year)
    lev = (target / vol).clip(upper=max_leverage).shift(1)
    return returns * lev


def ewm_vol(returns: pd.DataFrame, span: int = 60, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    return returns.ewm(span=span, min_periods=span // 2).std() * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------- statistics

def performance_stats(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    turnover: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    eq = (1 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(periods_per_year)
    sharpe = r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    dd = eq / eq.cummax() - 1
    out = {
        "start": r.index[0].date() if hasattr(r.index[0], "date") else r.index[0],
        "end": r.index[-1].date() if hasattr(r.index[-1], "date") else r.index[-1],
        "CAGR": cagr,
        "AnnReturn(arith)": r.mean() * periods_per_year,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "Sortino": r.mean() * periods_per_year / downside if downside > 0 else np.nan,
        "MaxDD": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "HitRate": (r > 0).mean(),
        "Skew": r.skew(),
    }
    if turnover is not None:
        out["AnnTurnover"] = turnover.mean() * periods_per_year
    if weights is not None:
        out["Exposure"] = (weights.abs().sum(axis=1) > 0).mean()
    return pd.Series(out)


def compare(results: dict[str, BacktestResult | pd.Series], periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    rows = {}
    for k, v in results.items():
        rows[k] = v.stats() if isinstance(v, BacktestResult) else performance_stats(v, periods_per_year)
    return pd.DataFrame(rows).T
