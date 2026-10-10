"""Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)
Zarattini, Aziz & Barbon (2024), SSRN 4824172.  Quantitativo applied it to ES and NQ
("Intraday Momentum for ES and NQ").

Rules:
  * Noise area per minute-of-day m:  σ_m = mean over the previous 14 sessions of
    |Close_m / Open_session − 1|  (move from the session open to minute m).
  * Bounds (gap-adjusted):  UB_m = max(Open_today, Close_yesterday) · (1 + σ_m)
                            LB_m = min(Open_today, Close_yesterday) · (1 − σ_m)
  * Decisions only at :00 and :30 (semi-hourly), first at 10:00.
    Price > UB → long;  price < LB → short.
  * Trailing stop: long exits if price < max(UB_m, VWAP_m); short exits if
    price > min(LB_m, VWAP_m).  Re-entry allowed at later decision times.
  * All positions flat at the close.
  * Sizing: leverage = min(4, 2% / σ_daily), σ_daily = 14-day std of daily returns.
  * Costs: $0.0035/share commission + $0.001/share slippage (SPY);
    for futures use ticks per side (``cost_per_side`` in return units).
Paper (SPY, 2007-2024): 19.6% annualised, Sharpe 1.33, 1,985% total return.

Data: Norgate does not provide intraday bars. Supply 1-minute RTH bars (New York
time, bar timestamped at its close) via :func:`load_minute_csv` — e.g. from
IBKR, Databento, FirstRate Data or Kibot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult


def load_minute_csv(path: str, tz: str | None = "America/New_York") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.capitalize() for c in df.columns]
    if tz and df.index.tz is not None:
        df.index = df.index.tz_convert(tz).tz_localize(None)
    t = df.index.time
    return df[(t > pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("16:00").time())]


def _session_matrix(bars: pd.DataFrame, field: str) -> pd.DataFrame:
    """days × minute-of-day matrix (minute index 1..390)."""
    df = bars[[field]].copy()
    df["day"] = df.index.normalize()
    df["mod"] = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    return df.pivot_table(index="day", columns="mod", values=field, aggfunc="last")


def run(bars: pd.DataFrame, lookback: int = 14, target_vol: float = 0.02, max_lev: float = 4.0,
        cost_per_side: float = 0.0, decision_every: int = 30, first_decision: int = 30,
        use_vwap_stop: bool = True) -> BacktestResult:
    close = _session_matrix(bars, "Close").ffill(axis=1)
    open_px = _session_matrix(bars, "Open").bfill(axis=1).iloc[:, 0]
    vol = _session_matrix(bars, "Volume").fillna(0)
    hlc = (_session_matrix(bars, "High") + _session_matrix(bars, "Low") + close) / 3
    vwap = (hlc * vol).cumsum(axis=1) / vol.cumsum(axis=1).replace(0, np.nan)
    vwap = vwap.ffill(axis=1)

    day_close = close.iloc[:, -1]
    prev_close = day_close.shift(1)
    move = (close.div(open_px, axis=0) - 1).abs()
    sigma = move.rolling(lookback, min_periods=lookback).mean().shift(1)  # previous 14 days only
    ub = (1 + sigma).mul(np.maximum(open_px, prev_close), axis=0)
    lb = (1 - sigma).mul(np.minimum(open_px, prev_close), axis=0)

    daily_ret = day_close.pct_change(fill_method=None)
    sd = daily_ret.rolling(lookback, min_periods=lookback).std().shift(1)
    lev = (target_vol / sd).clip(upper=max_lev)

    minutes = close.columns.values
    decision = (minutes >= first_decision) & (minutes % decision_every == 0)
    C, U, L, V = close.values, ub.values, lb.values, vwap.values
    n_days, n_min = C.shape
    pos = np.zeros((n_days, n_min))
    for d in range(n_days):
        if not np.isfinite(U[d]).any() or not np.isfinite(lev.iloc[d]):
            continue
        p = 0.0
        for m in range(n_min):
            if decision[m]:
                px = C[d, m]
                if p > 0:
                    stop = max(U[d, m], V[d, m]) if use_vwap_stop else U[d, m]
                    if px < stop:
                        p = 0.0
                elif p < 0:
                    stop = min(L[d, m], V[d, m]) if use_vwap_stop else L[d, m]
                    if px > stop:
                        p = 0.0
                if p == 0.0:
                    if px > U[d, m]:
                        p = 1.0
                    elif px < L[d, m]:
                        p = -1.0
            pos[d, m] = p
        pos[d, -1] = 0.0  # flat at the close
    # position held during minute m earns return from m to m+1
    r_min = np.diff(C, axis=1) / C[:, :-1]
    held = pos[:, :-1]
    gross_day = np.nansum(held * r_min, axis=1)
    trades = np.abs(np.diff(np.concatenate([np.zeros((n_days, 1)), pos], axis=1), axis=1)).sum(axis=1)
    levv = lev.fillna(0).values
    net_day = levv * (gross_day - trades * cost_per_side)
    idx = close.index
    res = BacktestResult(pd.Series(net_day, idx), pd.Series(levv * gross_day, idx),
                         pd.DataFrame({"exposure": (np.abs(pos).sum(axis=1) > 0).astype(float)}, idx),
                         pd.Series(trades * levv, idx), periods_per_year=252)
    res.meta = {"positions": pd.DataFrame(pos, idx, minutes), "leverage": lev}
    start = sigma.dropna(how="all").index.min()
    res.returns, res.gross, res.turnover = res.returns.loc[start:], res.gross.loc[start:], res.turnover.loc[start:]
    res.weights = res.weights.loc[start:]
    return res


if __name__ == "__main__":
    from ..data import synthetic

    r = run(synthetic.intraday_bars(250))
    print(r.stats())
