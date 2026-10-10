"""The AmiBroker/Norgate CSV export reader feeds the same strategies as the Norgate API."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alfamoney.data import amibroker, synthetic


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    d = tmp_path_factory.mktemp("ami_export")
    synthetic.write_amibroker_export(d)
    amibroker.set_export_dir(d)
    return d


def test_equity_panel_layout(export):
    p = amibroker.load_equity_panel("Russell 3000")
    assert {"Open", "High", "Low", "Close", "Volume", "UnadjClose", "member"} <= set(p)
    assert p["member"].dtypes.eq(bool).all()
    assert p["Close"].shape == p["member"].shape
    # symbols never in the Russell 2000 column are dropped
    with pytest.raises(ValueError):
        amibroker.load_equity_panel("Russell 2000")


def test_membership_respected(export):
    p = amibroker.load_equity_panel("S&P 500")
    assert (~p["member"]).any().any()  # synthetic export has entries/exits


def test_futures_and_contracts(export):
    f = amibroker.load_futures_panel(["ES", "ZN", "XX"])
    assert list(f["Close"].columns) == ["ES", "ZN"]
    strip = amibroker.load_contract_strip("ZC")
    assert len(strip["expiry"]) >= 4
    assert strip["expiry"].is_monotonic_increasing


def test_strategies_run_on_export(export):
    from alfamoney.strategies import basis_reversal, defense_first, stat_arb
    p = amibroker.load_equity_panel("Russell 3000")
    assert np.isfinite(stat_arb.run_signal(p, "a46").stats()["Sharpe"])
    px = defense_first.load_norgate("2010-01-01", amibroker)
    assert np.allclose(defense_first.run(px).meta["monthly_weights"].sum(axis=1), 1)
    r = basis_reversal.run({"ZC": amibroker.load_contract_strip("ZC")}, "time_series")
    assert len(r.returns) > 20


def test_runner_cli(export, tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_norgate.py"), "--source", "amibroker",
         "--ami-export", str(export), "--start", "2015-01-01",
         "--strategy", "stat_arb", "defense_first", "rebalancing", "third_friday"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert (tmp_path / "results" / "summary.csv").exists()
