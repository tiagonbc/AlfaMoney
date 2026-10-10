"""Asset Embeddings — Gabaix, Koijen, Richmond & Yogo (2025), NBER w33651.
Quantitativo post: "Asset Embeddings" (Sharpe 2.59, 18.5%/yr, 6.6% vol, DD −5.3%).

The paper learns firm embeddings from institutional holdings (13F): firms held by
similar portfolios are "close". The trading rule is Quantitativo's own design:
  1. Embeddings from the latest quarterly 13F holdings, usable 45 days after quarter end.
  2. K-means into 50 clusters.
  3. Using the previous quarter's data, per cluster regress next-day returns on the
     stock's residual return (stock − cluster mean); keep the 10 clusters with the
     strongest reversal (most negative t-statistic).
  4. Daily, within each selected cluster, long-short quintiles on the
     volatility-adjusted residual return (long the most negative residuals).
     Equal weight across clusters, rebalance daily.

Embeddings:
  * ``holdings_embeddings``: SVD of a (stock × manager) 13F holdings-share matrix
    (download from SEC EDGAR Form 13F data sets; neither Norgate nor P123 provide
    holder-level data).
  * ``return_embeddings``: fallback using PCA of trailing standardised daily
    returns — NOT the paper's method, a proxy when 13F data is unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult
from .common import tradable_mask


def holdings_embeddings(holdings: pd.DataFrame, dim: int = 128) -> pd.DataFrame:
    """holdings: rows = stocks, cols = managers, values = shares held / shares outstanding."""
    X = holdings.fillna(0.0).values
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(dim, len(S))
    return pd.DataFrame(U[:, :k] * S[:k], index=holdings.index)


def return_embeddings(ret: pd.DataFrame, t: pd.Timestamp, window: int = 252, dim: int = 20) -> pd.DataFrame:
    R = ret.loc[:t].iloc[-window:]
    R = R.loc[:, R.notna().mean() > 0.95].fillna(0.0)
    Z = (R - R.mean()) / R.std().replace(0, 1)
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    k = min(dim, len(S))
    return pd.DataFrame(Vt[:k].T * S[:k], index=R.columns)


def cluster(emb: pd.DataFrame, k: int = 50, seed: int = 0) -> pd.Series:
    from sklearn.cluster import KMeans

    k = min(k, len(emb))
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb.values)
    return pd.Series(lab, index=emb.index)


def _residuals(ret: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    cols = labels.index.intersection(ret.columns)
    r = ret[cols]
    cm = r.T.groupby(labels[cols]).transform("mean").T
    return r - cm


def select_clusters(ret: pd.DataFrame, labels: pd.Series, n_keep: int = 10) -> list[int]:
    res = _residuals(ret, labels)
    nxt = ret[res.columns].shift(-1)
    tstats = {}
    for c in labels.unique():
        members = labels.index[(labels == c)].intersection(res.columns)
        if len(members) < 5:
            continue
        x = res[members].iloc[:-1].values.ravel()
        y = nxt[members].iloc[:-1].values.ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 50:
            continue
        x, y = x[ok], y[ok]
        xm = x - x.mean()
        b = (xm @ (y - y.mean())) / (xm @ xm)
        e = y - y.mean() - b * xm
        se = np.sqrt(e @ e / (len(x) - 2) / (xm @ xm))
        tstats[c] = b / se
    return sorted(tstats, key=tstats.get)[:n_keep]


def run(panel: dict[str, pd.DataFrame], embeddings: dict[pd.Timestamp, pd.DataFrame] | None = None,
        n_clusters: int = 50, n_keep: int = 10, n_quantiles: int = 5, vol_window: int = 20,
        cost_bps: float = 0.0, min_price: float = 5.0, lag_days: int = 45) -> BacktestResult:
    """``embeddings``: quarter-end date -> embedding frame (from holdings_embeddings).
    If None, return-based embeddings are recomputed each quarter (proxy)."""
    close = panel["Close"]
    ret = close.pct_change(fill_method=None)
    mask = tradable_mask(panel, min_price=min_price)
    q_ends = close.resample("QE").last().index
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i in range(1, len(q_ends) - 1):
        q, q_next = q_ends[i], q_ends[i + 1]
        avail = q + pd.Timedelta(days=lag_days) if embeddings is not None else q
        if embeddings is not None:
            past = [d for d in embeddings if d + pd.Timedelta(days=lag_days) <= avail]
            if not past:
                continue
            emb = embeddings[max(past)]
        else:
            if (close.index <= q).sum() < 252:
                continue
            emb = return_embeddings(ret, q)
        emb = emb.loc[emb.index.intersection(close.columns)]
        if len(emb) < n_clusters * 3:
            continue
        labels = cluster(emb, n_clusters)
        prev = ret.loc[q_ends[i - 1]:q]
        keep = select_clusters(prev, labels, n_keep)
        start = close.index[close.index > avail]
        if len(start) == 0:
            continue
        period = ret.loc[start[0]:q_next].index
        sel = labels[labels.isin(keep)]
        # need history for the vol window → use ret from before the period
        hist = ret.loc[:q_next, sel.index]
        resid = _residuals(hist, sel)
        vol = resid.rolling(vol_window, min_periods=vol_window // 2).std()
        z = (resid / vol).loc[period]
        z = z.where(mask.loc[period, z.columns])
        wq = pd.DataFrame(0.0, index=period, columns=z.columns)
        for c in keep:
            m = sel.index[sel == c]
            zc = z[m]
            pct = zc.rank(axis=1, pct=True)
            lo = (pct <= 1 / n_quantiles).astype(float)
            hi = (pct > 1 - 1 / n_quantiles).astype(float)
            lw = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            sw = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            wq[m] += (lw - sw) / len(keep)
        W.loc[period, wq.columns] = wq.values
    from ..backtest.engine import run_weights

    res = run_weights(W, ret, cost_bps=cost_bps)
    active = W.abs().sum(axis=1) > 0
    if active.any():
        first = active.idxmax()
        res.returns, res.gross = res.returns.loc[first:], res.gross.loc[first:]
        res.turnover, res.weights = res.turnover.loc[first:], res.weights.loc[first:]
    return res
