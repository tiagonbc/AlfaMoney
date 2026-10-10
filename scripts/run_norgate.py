"""Run the replications on Norgate Data, read either through the Norgate Python API
(Windows machine with NDU running) or from an AmiBroker/Norgate CSV export
(amibroker/AlfaMoney_Export_Norgate.afl).

    python scripts/run_norgate.py --all
    python scripts/run_norgate.py --all --source amibroker --ami-export C:\AlfaMoney\ami_export
    python scripts/run_norgate.py --strategy trend_factor learning_to_rank --start 1995-01-01
    python scripts/run_norgate.py --strategy intraday_momentum --minute-csv NQ_1min.csv

Results (daily/monthly return series + summary table) go to results/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alfamoney.backtest.engine import compare  # noqa: E402
from alfamoney.data import amibroker, norgate  # noqa: E402

norgate_src = norgate  # replaced by the AmiBroker export reader when --source amibroker

OUT = Path("results")


def equity_panel(args):
    return norgate_src.load_equity_panel(args.index, args.start, args.end)


def s_trend_factor(args, cache):
    from alfamoney.strategies import trend_factor
    return {"TrendFactor (HZZ 2016)": trend_factor.run(cache["eq"](), cost_bps=args.cost_bps)}


def s_learning_to_rank(args, cache):
    from alfamoney.strategies import learning_to_rank as m
    p = cache["eq"]()
    return {"LTR LambdaMART (Quantitativo variant)": m.run(p, variant="quantitativo", cost_bps=args.cost_bps),
            "LTR LambdaMART (paper variant)": m.run(p, variant="paper", cost_bps=args.cost_bps)}


def s_takeuchi_lee(args, cache):
    from alfamoney.strategies import deep_momentum as m
    p = cache["eq"]()
    first = pd.Timestamp(args.start).year + 15
    return {"Takeuchi-Lee MLP": m.run_takeuchi_lee(p, first_test_year=first, cost_bps=args.cost_bps)}


def s_bimodal(args, cache):
    from alfamoney.strategies import deep_momentum as m
    p = cache["eq"]()
    first = pd.Timestamp(args.start).year + 15
    return {"Han 2022 bimodal (EW)": m.run_bimodal(p, first_test_year=first, cost_bps=args.cost_bps)}


def s_volume_shocks(args, cache):
    from alfamoney.strategies import volume_shocks as m
    p = cache["eq"]()
    return {"VolumeShock top decile (long)": m.run(p, cost_bps=args.cost_bps),
            "VolumeShock D10-D1": m.run(p, long_only=False, cost_bps=args.cost_bps)}


def s_stat_arb(args, cache):
    from alfamoney.strategies import stat_arb as m
    p = cache["eq"]()
    return {"Alpha#46": m.run_signal(p, "a46", cost_bps=args.cost_bps),
            "GTJA combo": m.combine(p, cost_bps=args.cost_bps)}


def s_asset_embeddings(args, cache):
    from alfamoney.strategies import asset_embeddings as m
    emb = None
    if args.holdings_dir:
        emb = {}
        for f in sorted(Path(args.holdings_dir).glob("*.parquet")):  # one file per quarter: YYYY-MM-DD.parquet
            emb[pd.Timestamp(f.stem)] = m.holdings_embeddings(pd.read_parquet(f))
    label = "AssetEmbeddings (13F)" if emb else "AssetEmbeddings (return-PCA proxy)"
    return {label: m.run(cache["eq"](), embeddings=emb, cost_bps=args.cost_bps)}


def s_third_friday(args, cache):
    from alfamoney.strategies import third_friday as m
    spy = norgate_src.price_frame("SPY", args.start, args.end)
    tab = m.summary(spy, args.cost_bps)
    print(tab)
    tab.to_csv(OUT / "third_friday_summary.csv")
    ev = m.event_returns(spy, args.cost_bps)
    from alfamoney.backtest.engine import BacktestResult
    mk = lambda r: BacktestResult(r, r, pd.DataFrame(index=r.index), pd.Series(0.0, index=r.index), 12)
    return {"3rd Friday spike (SPY)": mk(ev["spike"]), "3rd Friday reversal (SPY)": mk(ev["reversal"])}


def s_defense_first(args, cache):
    from alfamoney.strategies import defense_first as m
    return {"Defense First": m.run(m.load_norgate(args.start, norgate_src), cost_bps=args.cost_bps)}


def s_rebalancing(args, cache):
    from alfamoney.strategies import rebalancing as m
    es = norgate_src.continuous_future("ES", args.start, args.end)["Close"]
    zn = norgate_src.continuous_future("ZN", args.start, args.end)["Close"]
    return {"Rebalancing flow (thr+cal)": m.run(es, zn, cost_bps=args.cost_bps),
            "Rebalancing flow (threshold)": m.run(es, zn, combine="threshold", cost_bps=args.cost_bps),
            "Rebalancing flow (calendar)": m.run(es, zn, combine="calendar", cost_bps=args.cost_bps)}


def s_basis_reversal(args, cache):
    from alfamoney.strategies import basis_reversal as m
    strips = {}
    for r in m.COMMODITY_ROOTS:
        try:
            strips[r] = norgate_src.load_contract_strip(r, args.start, args.end)
        except Exception as e:  # market not in subscription
            print(f"skip {r}: {e}")
    out = {}
    for d in (30, 60):
        out[f"BasisReversal CS ({d}d)"] = m.run(strips, "cross_sectional", min_days=d)
        out[f"BasisReversal TS ({d}d)"] = m.run(strips, "time_series", min_days=d)
    return out


def s_network_momentum(args, cache):
    from alfamoney.strategies import network_momentum as m
    fut = norgate_src.load_futures_panel(start=args.start, end=args.end)
    return {"NetworkMomentum (learned graph)": m.run(fut, cost_bps=args.cost_bps),
            "Own-features only": m.run(fut, include_own=True, graph="uniform", cost_bps=args.cost_bps),
            "Placebo uniform graph": m.run(fut, graph="uniform", cost_bps=args.cost_bps),
            "Placebo random graph": m.run(fut, graph="random", cost_bps=args.cost_bps)}


def s_intraday_momentum(args, cache):
    from alfamoney.strategies import intraday_momentum as m
    if not args.minute_csv:
        raise SystemExit("intraday_momentum needs --minute-csv (Norgate has no intraday data)")
    bars = m.load_minute_csv(args.minute_csv)
    return {"Intraday momentum (paper: 14d, 2%, 4x)": m.run(bars),
            "Intraday momentum (Quantitativo: 90d, 3%, 8x)": m.run(bars, lookback=90, target_vol=0.03, max_lev=8)}


STRATEGIES = {k[2:]: v for k, v in globals().items() if k.startswith("s_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="*", choices=list(STRATEGIES), default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--end", default="2999-01-01")
    ap.add_argument("--index", default="Russell 3000")
    ap.add_argument("--cost-bps", type=float, default=0.0)
    ap.add_argument("--source", choices=["norgate", "amibroker"], default="norgate",
                    help="norgate = Norgate Python API; amibroker = CSV export from the AmiBroker/Norgate database")
    ap.add_argument("--ami-export", help="folder written by amibroker/AlfaMoney_Export_Norgate.afl")
    ap.add_argument("--minute-csv")
    ap.add_argument("--holdings-dir", help="quarterly 13F holdings matrices for asset embeddings")
    args = ap.parse_args()
    global norgate_src
    if args.source == "amibroker":
        if args.ami_export:
            amibroker.set_export_dir(args.ami_export)
        norgate_src = amibroker
    names = list(STRATEGIES) if args.all else args.strategy
    if not names:
        ap.error("choose --strategy ... or --all")
    OUT.mkdir(exist_ok=True)
    _eq = {}
    cache = {"eq": lambda: _eq.setdefault("p", equity_panel(args))}
    allres = {}
    for n in names:
        if args.all and n == "intraday_momentum" and not args.minute_csv:
            continue
        print(f"=== {n}")
        res = STRATEGIES[n](args, cache)
        for label, r in res.items():
            s = r if isinstance(r, pd.Series) else r.returns
            s.to_csv(OUT / f"{label.replace(' ', '_').replace('/', '-')}.csv")
        allres.update(res)
    table = compare(allres)
    print(table.to_string())
    table.to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
