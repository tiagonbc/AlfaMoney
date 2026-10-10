"""Network Momentum across Asset Classes — Pu, Roberts, Dong & Zohren (2023),
arXiv:2308.11294.  Quantitativo post: "Network Momentum" (60 Norgate futures,
Sharpe ≈ 2.1 vs 1.13 for own-features only; placebo graphs 1.24–1.32).

Pipeline:
  1. Eight momentum features per asset: vol-scaled returns over 1, 21, 63, 126,
     252 days and the Baz et al. normalised MACD (8,24), (16,48), (32,96).
  2. Graph learning (Kalofolias 2016, log-degree model) on the feature history of
     each lookback δ ∈ {252, 504, 756, 1008, 1260} days:
        min_W  ‖W∘Z‖₁ − α·1ᵀlog(W1) + β‖W‖²_F   s.t. W ≥ 0, W = Wᵀ, diag W = 0
     where Z_ij is the squared distance between assets' stacked feature histories.
     The five graphs are averaged (ensemble).
  3. Network momentum features: Ã·X (Ã = degree-normalised adjacency).
  4. A single pooled linear regression of next-day vol-scaled return on the network
     features, refit yearly on an expanding window (≥ ``min_train_years``).
  5. Position_i ∝ prediction_i / σ_i, portfolio scaled to ``target_vol`` ex-ante.
Paper: OOS 2000–2022 Sharpe ≈ 1.51.

Data: Norgate continuous back-adjusted futures (norgate.load_futures_panel). Returns
must come from the back-adjusted series' point changes divided by the *unadjusted*
price where possible; with "_CCB" ratio-free series use ``returns_from_points``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, ewm_vol
from .ml_common import macd_signal

RET_WINDOWS = (1, 21, 63, 126, 252)
GRAPH_LOOKBACKS = (252, 504, 756, 1008, 1260)


def returns_from_points(adj_close: pd.DataFrame, unadj_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily returns of a back-adjusted (point-shifted) continuous series."""
    if unadj_close is None:
        return adj_close.pct_change(fill_method=None)
    return adj_close.diff() / unadj_close.shift(1)


def momentum_features(close: pd.DataFrame, ret: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sigma = ret.ewm(span=60, min_periods=30).std()
    cum = (1 + ret.fillna(0)).cumprod()
    feats = {}
    for k in RET_WINDOWS:
        feats[f"r{k}"] = (cum / cum.shift(k) - 1) / (sigma * np.sqrt(k))
    for s, l in ((8, 24), (16, 48), (32, 96)):
        feats[f"macd{s}_{l}"] = macd_signal(close, s, l)
    # winsorise as in the paper (±5 σ-units)
    return {k: v.clip(-5, 5) for k, v in feats.items()}


def learn_graph(Z: np.ndarray, alpha: float = 1.0, beta: float = 1.0, max_iter: int = 2000,
                tol: float = 1e-5) -> np.ndarray:
    """Kalofolias (2016) log-degree graph learning via primal-dual splitting."""
    n = Z.shape[0]
    iu = np.triu_indices(n, 1)
    z = Z[iu]
    m = len(z)
    # S maps edge weights to node degrees
    rows = np.concatenate([iu[0], iu[1]])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    S = np.zeros((n, m))
    S[rows, cols] = 1.0
    mu = 2 * beta + np.sqrt(2 * (n - 1))
    gamma = 0.95 / mu
    w = np.zeros(m)
    v = S @ w
    for _ in range(max_iter):
        Y = w - gamma * (2 * beta * w + S.T @ v)
        y = v + gamma * (S @ w)
        P = np.maximum(0.0, Y - 2 * gamma * z)
        p = (y - np.sqrt(y ** 2 + 4 * alpha * gamma)) / 2
        Q = P - gamma * (2 * beta * P + S.T @ p)
        q = p + gamma * (S @ P)
        w_new, v_new = w - Y + Q, v - y + q
        if np.linalg.norm(w_new - w) / max(np.linalg.norm(w), 1e-12) < tol and _ > 10:
            w, v = w_new, v_new
            break
        w, v = w_new, v_new
    W = np.zeros((n, n))
    W[iu] = w
    return W + W.T


def graph_at(feats: dict[str, pd.DataFrame], t: pd.Timestamp, lookback: int, alpha: float, beta: float) -> pd.DataFrame:
    """Learn the graph from the feature history in (t−lookback, t]."""
    blocks = []
    for f in feats.values():
        blk = f.loc[:t].iloc[-lookback:]
        blocks.append(blk)
    hist = pd.concat(blocks, axis=0)  # (lookback·8) × assets
    ok = hist.notna().mean() > 0.9
    H = hist.loc[:, ok].fillna(0.0)
    if H.shape[1] < 3:
        return pd.DataFrame(0.0, index=H.columns, columns=H.columns)
    H = (H - H.mean()) / H.std().replace(0, 1)
    V = H.T.values
    sq = (V ** 2).sum(axis=1)
    Z = np.maximum(sq[:, None] + sq[None, :] - 2 * V @ V.T, 0.0)
    Z = Z / Z[np.triu_indices(len(Z), 1)].mean()
    W = learn_graph(Z, alpha, beta)
    return pd.DataFrame(W, index=H.columns, columns=H.columns)


def ensemble_graph(feats, t, assets, lookbacks=GRAPH_LOOKBACKS, alpha=1.0, beta=1.0) -> pd.DataFrame:
    A = pd.DataFrame(0.0, index=assets, columns=assets)
    k = 0
    for L in lookbacks:
        if len(feats["r1"].loc[:t]) < L:
            continue
        G = graph_at(feats, t, L, alpha, beta).reindex(index=assets, columns=assets).fillna(0)
        A += G
        k += 1
    return A / max(k, 1)


def network_features(feats: dict[str, pd.DataFrame], graphs: dict[pd.Timestamp, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Ã·X for each feature, using the most recent graph available at each date."""
    dates = feats["r1"].index
    gdates = sorted(graphs)
    out = {k: pd.DataFrame(np.nan, index=dates, columns=v.columns) for k, v in feats.items()}
    for i, g0 in enumerate(gdates):
        g1 = gdates[i + 1] if i + 1 < len(gdates) else dates[-1] + pd.Timedelta(days=1)
        sl = (dates > g0) & (dates <= g1)  # graph learned at close g0 is used from g0+1
        A = graphs[g0]
        deg = A.sum(axis=1).replace(0, np.nan)
        An = A.div(deg, axis=0).fillna(0)
        for k, f in feats.items():
            X = f.loc[sl, An.columns].fillna(0.0)
            out[k].loc[sl, An.columns] = X.values @ An.values.T
    return out


def run(panel: dict[str, pd.DataFrame], ret: pd.DataFrame | None = None, graph_freq: str = "ME",
        alpha: float = 1.0, beta: float = 1.0, min_train_years: int = 10, target_vol: float = 0.15,
        include_own: bool = False, cost_bps: float = 0.0, graph: str = "learned", seed: int = 0) -> BacktestResult:
    """graph: 'learned' | 'uniform' | 'random' (placebos, as in the Quantitativo post)."""
    close = panel["Close"]
    ret = returns_from_points(close) if ret is None else ret
    feats = momentum_features(close, ret)
    sigma = ret.ewm(span=60, min_periods=30).std()
    target = (ret.shift(-1) / sigma).clip(-20, 20)  # next-day vol-scaled return
    assets = close.columns
    gdates = close.resample(graph_freq).last().index
    gdates = [close.index[close.index <= d][-1] for d in gdates if (close.index <= d).any()]
    gdates = [d for d in gdates if (close.index <= d).sum() >= min(GRAPH_LOOKBACKS)]
    rng = np.random.default_rng(seed)
    graphs = {}
    for d in gdates:
        if graph == "uniform":
            G = pd.DataFrame(1.0, index=assets, columns=assets) - np.eye(len(assets))
        elif graph == "random":
            R = rng.random((len(assets), len(assets)))
            R = np.triu(R, 1)
            G = pd.DataFrame(R + R.T, index=assets, columns=assets)
        else:
            G = ensemble_graph(feats, d, assets, alpha=alpha, beta=beta)
        graphs[d] = G
    nfeats = network_features(feats, graphs)
    use = dict(nfeats)
    if include_own:
        use.update({f"own_{k}": v for k, v in feats.items()})
    X = pd.DataFrame({k: v.stack() for k, v in use.items()})
    y = target.stack().reindex(X.index)
    data = X.join(y.rename("y")).dropna()
    dates = data.index.get_level_values(0)
    years = sorted(set(dates.year))
    start_year = years[0] + min_train_years
    preds = []
    for yr in range(start_year, years[-1] + 1):
        tr = data[dates < pd.Timestamp(yr, 1, 1) - pd.Timedelta(days=1)]
        te = X[(X.index.get_level_values(0).year == yr)].dropna()
        if len(tr) < 1000 or te.empty:
            continue
        A = np.column_stack([np.ones(len(tr)), tr.drop(columns="y").values])
        b, *_ = np.linalg.lstsq(A, tr["y"].values, rcond=None)
        p = np.column_stack([np.ones(len(te)), te.values]) @ b
        preds.append(pd.Series(p, index=te.index))
    pred = pd.concat(preds).unstack().reindex(columns=assets)
    raw = pred / (sigma.reindex_like(pred) * np.sqrt(252))
    # scale the book to target_vol using trailing realised vol of the unscaled book
    book = (raw.shift(1) * ret.reindex_like(raw)).sum(axis=1)
    scale = target_vol / (book.rolling(63, min_periods=20).std() * np.sqrt(252))
    w = raw.mul(scale.shift(1).clip(upper=1e3), axis=0).fillna(0)
    held = w.shift(1).fillna(0)
    gross = (held * ret.reindex_like(held).fillna(0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).shift(1).fillna(0)
    net = gross - turnover * cost_bps / 1e4
    first = pred.dropna(how="all").index.min()
    sl = net.index > first
    return BacktestResult(net[sl], gross[sl], held[sl], turnover[sl], meta={"graphs": graphs, "pred": pred})
