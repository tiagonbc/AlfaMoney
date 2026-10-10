"""Feature builders and walk-forward utilities shared by the ML stock strategies
(Learning to Rank, Takeuchi & Lee deep momentum, Han's bimodal returns)."""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .common import monthly


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


def stack_features(feats: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    """dict of wide (date × asset) frames -> long frame indexed by (date, asset)."""
    cols = {k: v.where(mask).stack() for k, v in feats.items()}
    return pd.DataFrame(cols).dropna()


# ----------------------------------------------------------------- Baz et al. / Poh et al. features

def _ewma(x: pd.DataFrame, timescale: int) -> pd.DataFrame:
    # Baz et al. (2015): EWMA with half-life HL = log(0.5)/log(1 - 1/S)
    hl = np.log(0.5) / np.log(1 - 1 / timescale)
    return x.ewm(halflife=hl, min_periods=timescale).mean()


def macd_signal(close: pd.DataFrame, s: int, l: int) -> pd.DataFrame:
    """Baz et al. normalised MACD: q = (EWMA_S − EWMA_L)/std63(price); y = q/std252(q)."""
    q = (_ewma(close, s) - _ewma(close, l)) / close.rolling(63, min_periods=40).std()
    return q / q.rolling(252, min_periods=150).std()


MACD_PAIRS = ((8, 24), (16, 48), (32, 96))


def ltr_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The 21 month-end features of Poh, Lim, Zohren & Roberts (2021):
    3 raw returns (3/6/12m), 3 vol-normalised returns, 3 MACD indicators and
    those MACDs lagged 1, 3, 6 and 12 months (12)."""
    daily_ret = close.pct_change(fill_method=None)
    sigma = daily_ret.ewm(span=63, min_periods=40).std()
    me_close, me_sigma = monthly(close), monthly(sigma)
    feats: dict[str, pd.DataFrame] = {}
    for m, d in ((3, 63), (6, 126), (12, 252)):
        r = me_close / me_close.shift(m) - 1
        feats[f"ret_{m}m"] = r
        feats[f"nret_{m}m"] = r / (me_sigma * np.sqrt(d))
    for s, l in MACD_PAIRS:
        mm = monthly(macd_signal(close, s, l))
        feats[f"macd_{s}_{l}"] = mm
        for lag in (1, 3, 6, 12):
            feats[f"macd_{s}_{l}_l{lag}"] = mm.shift(lag)
    return feats


# ----------------------------------------------------------------- Takeuchi & Lee (2013) features

def takeuchi_lee_features(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """33 inputs at month end t (predicting month t+1):
    * 12 cumulative returns ending at month t-1 over 1..12 months (months t-13 … t-2 window starts),
    * 20 cumulative daily returns within the last 20 trading days (d = 1..20 days to month end),
    * January dummy (next month is January).
    Each return feature is z-scored cross-sectionally, as in the paper."""
    me = monthly(close)
    feats: dict[str, pd.DataFrame] = {}
    for k in range(2, 14):
        feats[f"m{k}"] = zscore_cs(me.shift(1) / me.shift(k) - 1)
    # daily cumulative returns, sampled at month ends
    for d in range(1, 21):
        cr = close / close.shift(d) - 1
        feats[f"d{d}"] = zscore_cs(monthly(cr))
    jan = pd.DataFrame(np.repeat(((me.index.month == 12).astype(float))[:, None], me.shape[1], axis=1),
                       index=me.index, columns=me.columns)
    feats["jan"] = jan
    return feats


# ----------------------------------------------------------------- walk-forward

def walk_forward_dates(dates: pd.DatetimeIndex, train_years: int | None, retrain_every_years: int,
                       first_test_year: int) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_dates, test_dates). Training uses only dates strictly before the
    test block minus one month (the label of the last training row is month t+1)."""
    years = sorted(set(dates.year))
    for y0 in range(first_test_year, years[-1] + 1, retrain_every_years):
        test = dates[(dates.year >= y0) & (dates.year < y0 + retrain_every_years)]
        if len(test) == 0:
            continue
        cutoff = test[0]
        tr = dates[dates < cutoff][:-1]  # drop the last month: its label overlaps the test block
        if train_years:
            tr = tr[tr >= cutoff - pd.DateOffset(years=train_years)]
        if len(tr) == 0:
            continue
        yield tr, test


def predict_walk_forward(X: pd.DataFrame, y: pd.Series, fit_predict: Callable, train_years: int | None,
                         retrain_every_years: int, first_test_year: int) -> pd.Series:
    """X, y indexed by (date, asset). ``fit_predict(Xtr, ytr, Xte) -> np.ndarray``."""
    dates = X.index.get_level_values(0).unique().sort_values()
    preds = []
    for tr, te in walk_forward_dates(dates, train_years, retrain_every_years, first_test_year):
        itr = X.index.get_level_values(0).isin(tr) & y.notna().values
        ite = X.index.get_level_values(0).isin(te)
        if itr.sum() < 100 or ite.sum() == 0:
            continue
        p = fit_predict(X[itr], y[itr], X[ite])
        preds.append(pd.Series(p, index=X.index[ite]))
    return pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)


def long_short_monthly(score: pd.DataFrame, ret_next: pd.DataFrame, n_quantiles: int = 10,
                       gross: float = 2.0, vol_scale: pd.DataFrame | None = None,
                       long_only: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly decile portfolio from scores at month t, earning month t+1 returns.
    ``vol_scale`` (optional) multiplies each position (e.g. 0.15/σ_i for 15% vol-target)."""
    pct = score.rank(axis=1, pct=True)
    top = (pct > 1 - 1 / n_quantiles).astype(float)
    bot = (pct <= 1 / n_quantiles).astype(float)
    if vol_scale is not None:
        top, bot = top * vol_scale.reindex_like(top), bot * vol_scale.reindex_like(bot)
    lw = top.div(top.sum(axis=1).replace(0, np.nan), axis=0)
    sw = bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0)
    if long_only:
        w = lw.fillna(0)
    else:
        w = (lw.fillna(0) - sw.fillna(0)) * gross / 2
    r = (w * ret_next.reindex_like(w).fillna(0)).sum(axis=1)
    return r, w


def to_realisation(series: pd.Series, month_index: pd.DatetimeIndex) -> pd.Series:
    """A series decided at month-end t (earning month t+1) re-indexed to t+1."""
    first = series.index.min()
    out = series.reindex(month_index).shift(1)
    return out[out.index > first].fillna(0.0)


def monthly_result(r: pd.Series, w: pd.DataFrame, month_index: pd.DatetimeIndex, cost_bps: float = 0.0,
                   meta: dict | None = None):
    from ..backtest.engine import BacktestResult

    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    gross = to_realisation(r, month_index)
    to = to_realisation(turnover, month_index)
    net = gross - to * cost_bps / 1e4
    return BacktestResult(net, gross, w, to, periods_per_year=12, meta=meta or {})
