import numpy as np
import pandas as pd

from alfamoney.backtest.engine import performance_stats, quantile_portfolio, run_weights


def test_run_weights_lag_avoids_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=5)
    rets = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02, 0.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 0.0, 0.0, 0.0]}, index=idx)
    res = run_weights(w, rets, lag=1)
    # weight decided on day 0 earns day 1; weight of day 1 earns day 2
    assert np.allclose(res.gross.values, [0.0, 0.10, -0.05, 0.0, 0.0])


def test_costs_reduce_returns():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rets = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 0.0, 1.0, 0.0]}, index=idx)
    a = run_weights(w, rets, cost_bps=0).returns.sum()
    b = run_weights(w, rets, cost_bps=10).returns.sum()
    assert b < a


def test_quantile_portfolio_dollar_neutral():
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.normal(size=(3, 50)))
    w = quantile_portfolio(sig, 5)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.clip(lower=0).sum(axis=1), 1)


def test_stats_basic():
    r = pd.Series([0.01, -0.005] * 126, index=pd.bdate_range("2020-01-01", periods=252))
    s = performance_stats(r)
    assert s["Sharpe"] > 0 and s["MaxDD"] < 0
