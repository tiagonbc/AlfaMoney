"""The Derivative Payoff Bias / "The Third Friday Price Spike"
Baltussen, Terstegge & Whelan (2023), SSRN 4562800.
Quantitativo post: "The Derivative Payoff Bias".

Finding: on monthly option-expiration Fridays (3rd Friday) the S&P 500 Special
Opening Quotation (SOQ) is on average ≈ 18.5 bp above Thursday's close (2003–2021),
then reverses during the day (dealer charm hedging).

Two daily-data trades (Norgate SPY or ES):
  A) "spike":     long at Thursday close  → exit at Friday open.
  B) "reversal":  short at Friday open    → cover at Friday close (paper: before noon;
                  intraday data required for the noon exit).
Holiday rule: if the 3rd Friday is an exchange holiday, expiration/SOQ moves to the
preceding Thursday (then the trade is Wednesday close → Thursday open).

Notes: SPY's 09:30 opening print is a close proxy for the SOQ. Norgate's ES "Open" is
the Globex open (previous evening), so use SPY or an RTH-open futures series.
"""
from __future__ import annotations

import pandas as pd

from ..backtest.engine import performance_stats


def expiration_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly 3rd-Friday expirations mapped to the trading calendar in ``index``."""
    out = []
    months = pd.period_range(index.min(), index.max(), freq="M")
    tset = set(index)
    for m in months:
        first = m.start_time
        fri = first + pd.offsets.Week(weekday=4) if first.weekday() != 4 else first
        third = fri + pd.Timedelta(weeks=2)
        if third in tset:
            out.append(third)
        else:
            prev = index[index < third]
            if len(prev) and prev[-1] >= third - pd.Timedelta(days=3):
                out.append(prev[-1])
    return pd.DatetimeIndex(out)


def event_returns(ohlc: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    """Per-event returns for trade A (close→open) and B (short open→close)."""
    o, c = ohlc["Open"], ohlc["Close"]
    rows = []
    for d in expiration_days(ohlc.index):
        i = ohlc.index.get_loc(d)
        if i == 0:
            continue
        prev_close = c.iloc[i - 1]
        a = o.iloc[i] / prev_close - 1 - 2 * cost_bps / 1e4
        b = -(c.iloc[i] / o.iloc[i] - 1) - 2 * cost_bps / 1e4
        rows.append({"date": d, "spike": a, "reversal": b, "both": a + b})
    return pd.DataFrame(rows).set_index("date")


def summary(ohlc: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    ev = event_returns(ohlc, cost_bps)
    stats = {}
    for col in ev:
        s = performance_stats(ev[col], periods_per_year=12)
        s["mean_bp"] = ev[col].mean() * 1e4
        s["t_stat"] = ev[col].mean() / ev[col].std() * len(ev) ** 0.5
        s["n_events"] = len(ev)
        stats[col] = s
    return pd.DataFrame(stats).T
