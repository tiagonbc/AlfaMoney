"""Smoke tests on synthetic data: every strategy runs end-to-end and respects
basic invariants. Synthetic results say nothing about real performance."""
import numpy as np
import pandas as pd
import pytest

from alfamoney.data import synthetic


@pytest.fixture(scope="module")
def eq():
    return synthetic.equity_panel(120, 252 * 6)


@pytest.fixture(scope="module")
def fut():
    return synthetic.futures_panel(12, 252 * 8)


def _ok(res):
    s = res.stats()
    assert len(res.returns) > 10
    assert np.isfinite(s["Sharpe"])
    return s


def test_trend_factor(eq):
    from alfamoney.strategies import trend_factor
    _ok(trend_factor.run(eq))


def test_learning_to_rank(eq):
    from alfamoney.strategies import learning_to_rank
    _ok(learning_to_rank.run(eq, first_test_year=2019))


def test_deep_momentum(eq):
    from alfamoney.strategies import deep_momentum
    _ok(deep_momentum.run_takeuchi_lee(eq, first_test_year=2019, train_years=3))
    _ok(deep_momentum.run_bimodal(eq, first_test_year=2019, hidden=(32, 16)))


def test_reversal_signals_find_synthetic_reversal(eq):
    from alfamoney.strategies import stat_arb
    # synthetic data has built-in negative autocorrelation -> reversal must be profitable
    assert stat_arb.run_signal(eq, "a46").stats()["Sharpe"] > 1


def test_volume_shocks(eq):
    from alfamoney.strategies import volume_shocks
    r = volume_shocks.run(eq)
    _ok(r)
    assert np.allclose(r.weights.sum(axis=1).iloc[100:], 1.0)


def test_asset_embeddings(eq):
    from alfamoney.strategies import asset_embeddings
    _ok(asset_embeddings.run(eq, n_clusters=8, n_keep=3))


def test_third_friday_calendar():
    from alfamoney.strategies.third_friday import expiration_days
    d = expiration_days(pd.bdate_range("2024-01-01", "2024-12-31"))
    assert list(d[:3].strftime("%Y-%m-%d")) == ["2024-01-19", "2024-02-16", "2024-03-15"]


def test_defense_first_weights_sum_to_one(fut):
    from alfamoney.strategies import defense_first
    px = fut["Close"].iloc[:, :6].copy()
    px.columns = ["TLT", "GLD", "DBC", "UUP", "SPY", "BIL"]
    r = defense_first.run(px)
    assert np.allclose(r.meta["monthly_weights"].sum(axis=1), 1.0)


def test_rebalancing(fut):
    from alfamoney.strategies import rebalancing
    _ok(rebalancing.run(fut["Close"].iloc[:, 0], fut["Close"].iloc[:, 1]))


def test_network_momentum(fut):
    from alfamoney.strategies import network_momentum
    _ok(network_momentum.run(fut, min_train_years=5))


def test_learn_graph_recovers_clusters():
    from alfamoney.strategies.network_momentum import learn_graph
    rng = np.random.default_rng(0)
    V = np.vstack([rng.normal(0, 1, (4, 30)) * 0.2 + 1, rng.normal(0, 1, (4, 30)) * 0.2 - 1])
    sq = (V ** 2).sum(1)
    Z = np.maximum(sq[:, None] + sq[None] - 2 * V @ V.T, 0)
    W = learn_graph(Z / Z[np.triu_indices(8, 1)].mean())
    assert W[:4, :4].sum() > 10 * W[:4, 4:].sum()


def _strip(seed, n_days=800):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    expiries = pd.date_range("2020-02-15", periods=n_days // 20 + 6, freq="MS") + pd.Timedelta(days=14)
    spot = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
    close, exp = {}, {}
    for k, e in enumerate(expiries):
        name = f"C{k:03d}"
        tau = np.maximum((e - dates).days, 0) / 365
        noise = rng.normal(0, 0.002, n_days)
        px = spot * np.exp(0.02 * tau + noise - np.r_[0, noise[:-1]] * 0.5)
        s = pd.Series(px, index=dates)
        s[dates > e] = np.nan
        close[name], exp[name] = s, e
    return {"Close": pd.DataFrame(close), "expiry": pd.Series(exp)}


def test_basis_reversal():
    from alfamoney.strategies import basis_reversal
    strips = {f"M{i}": _strip(i) for i in range(6)}
    r = basis_reversal.run(strips, "cross_sectional")
    _ok(r)
    _ok(basis_reversal.run(strips, "time_series"))


def test_intraday_momentum():
    from alfamoney.strategies import intraday_momentum
    r = intraday_momentum.run(synthetic.intraday_bars(60))
    _ok(r)
    pos = r.meta["positions"]
    assert (pos.iloc[:, -1] == 0).all()  # flat at the close
