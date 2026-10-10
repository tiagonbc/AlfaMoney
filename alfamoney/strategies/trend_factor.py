"""A Trend Factor — Han, Zhou & Zhu (2016), Journal of Financial Economics 122(2).

Rules (paper, Section 2):
  * Moving averages of daily (adjusted) closes over L ∈ {3,5,10,20,50,100,200,400,600,800,1000}
    days, computed on the last trading day of each month and normalised by that
    day's close:  A_{j,t,L} = MA_{j,t,L} / P_{j,t}.
  * Each month t, cross-sectional OLS of the month-t return on the month-(t-1)
    normalised MAs:  r_{j,t} = β0,t + Σ_L β_{L,t} A_{j,t-1,L} + ε.
  * Expected return for month t+1:  E_t[r_{j,t+1}] = Σ_L E_t[β_L] · A_{j,t,L},
    where E_t[β_L] is the 12-month average of the past β_{L} estimates.
  * Sort stocks into quintiles on E_t[r]; the factor is long Q5 / short Q1,
    equal-weighted, rebalanced monthly.
  * Universe: NYSE/AMEX/NASDAQ common stocks, price ≥ $5, excluding the smallest
    NYSE size decile (here: dollar-volume proxy, or market cap from P123).
Paper headline (1930-2014): ~1.63%/month average return, Sharpe ≈ 0.47 (monthly)
— more than double that of the 12-1 momentum factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, performance_stats, quantile_portfolio
from .common import monthly, nw_tstat, tradable_mask

LAGS = (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000)


def normalised_mas(close: pd.DataFrame, lags=LAGS) -> dict[int, pd.DataFrame]:
    """Month-end A_{j,t,L} = MA_L / P for each lag."""
    me = monthly(close)
    out = {}
    for L in lags:
        ma = close.rolling(L, min_periods=L).mean()
        out[L] = (monthly(ma) / me)
    return out


def trend_signal(close: pd.DataFrame, mask_m: pd.DataFrame, lags=LAGS, beta_window: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expected-return signal, betas) at month ends."""
    A = normalised_mas(close, lags)
    ret_m = monthly(close).pct_change(fill_method=None)
    dates = ret_m.index
    betas = pd.DataFrame(np.nan, index=dates, columns=["const", *lags])
    for i in range(1, len(dates)):
        t, tm1 = dates[i], dates[i - 1]
        y = ret_m.loc[t]
        X = pd.DataFrame({L: A[L].loc[tm1] for L in lags})
        ok = mask_m.loc[tm1] & y.notna() & X.notna().all(axis=1)
        if ok.sum() < len(lags) + 10:
            continue
        Xv = np.column_stack([np.ones(ok.sum()), X[ok].values])
        b, *_ = np.linalg.lstsq(Xv, y[ok].values, rcond=None)
        betas.loc[t] = b
    eb = betas.rolling(beta_window, min_periods=beta_window).mean()
    sig = sum(A[L].mul(eb[L], axis=0) for L in lags)
    return sig.where(mask_m), betas


def run(panel: dict[str, pd.DataFrame], n_quantiles: int = 5, min_price: float = 5.0,
        size_pct_cut: float = 0.10, cost_bps: float = 0.0) -> BacktestResult:
    close = panel["Close"]
    mask = tradable_mask(panel, min_price=min_price, size_pct_cut=size_pct_cut)
    mask_m = monthly(mask.astype(float)).astype(bool)
    sig, betas = trend_signal(close, mask_m)
    w = quantile_portfolio(sig, n_quantiles, mask_m)
    ret_m = monthly(close).pct_change(fill_method=None)
    # weights at end of month t earn month t+1 returns
    w_l = w.shift(1).fillna(0.0)
    gross = (w_l * ret_m.fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1).shift(1).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    start = sig.dropna(how="all").index.min()
    net, gross, turnover = net.loc[start:].iloc[1:], gross.loc[start:].iloc[1:], turnover.loc[start:].iloc[1:]
    res = BacktestResult(net, gross, w_l.loc[net.index], turnover, periods_per_year=12)
    res.meta = {"betas": betas, "nw_t": nw_tstat(net)}
    return res


# ------------------------------------------------------------------ Portfolio123
# P123 cannot run the rolling cross-sectional regression natively. Compute the signal
# here and upload it as a custom stock factor (alfamoney.data.p123.upload_stock_factor),
# then rank on StockFactor("TrendFactor") — see scripts/run_p123.py.


def p123_signal_frame(sig: pd.DataFrame) -> pd.DataFrame:
    long = sig.stack().rename("value").reset_index()
    long.columns = ["date", "ticker", "value"]
    return long


if __name__ == "__main__":
    from ..data import synthetic

    p = synthetic.equity_panel(300, 2600)
    r = run(p)
    print(performance_stats(r.returns, 12))
