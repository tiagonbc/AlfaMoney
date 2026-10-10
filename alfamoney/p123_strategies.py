"""Portfolio123 versions of the stock strategies (where P123 is applicable).

P123 is the right tool when a strategy needs (a) point-in-time fundamentals such as
market cap (size filters, size dummies, value weights) or (b) can be expressed as a
weekly/monthly screen or ranking.  It is NOT suitable for: overnight holding (volume
shocks), daily-rebalanced stat-arb, futures (basis reversal, rebalancing flows,
network momentum, intraday momentum) — those use Norgate.

Every function returns the raw API response; credentials via P123_API_ID/P123_API_KEY.
"""
from __future__ import annotations

import pandas as pd

from .data import p123

# ------------------------------------------------------------------ formulas

# GTJA Alpha#46 (Modern Statistical Arbitrage). Higher = further below the blended MA.
ALPHA46 = "(SMA(3,0)+SMA(6,0)+SMA(12,0)+SMA(24,0))/(4*Close(0))"

# Volume shock (Cartea et al.). Approximation: AvgVol is a simple, not EMA, baseline.
VOLUME_SHOCK = "Volume(0)/AvgVol(60,1)-1"

# Investable-universe rules used across the replications (paper conventions)
BASE_RULES = ["Close(0) >= 5", "FRank(\"MktCap\",#All,#Asc) > 10"]  # price ≥ $5, drop smallest decile

# Poh et al. 21 features are price-based and computed from Norgate; for P123 data
# sourcing, the raw-return part maps to:
LTR_P123_FEATURES = {
    "ret_3m": "Close(0)/Close(63)-1",
    "ret_6m": "Close(0)/Close(126)-1",
    "ret_12m": "Close(0)/Close(252)-1",
    "vol_63": "StdDev(63)",
    "mktcap": "MktCap",
}


def _rank_xml(formula: str, lower_is_better: bool = False, name: str = "Signal") -> str:
    return f"""<RankingSystem>
  <RankPerformance>
    <NNodes>1</NNodes>
    <SNode>
      <Name>{name}</Name>
      <Formula>{formula}</Formula>
      <LowerIsBetter>{int(lower_is_better)}</LowerIsBetter>
      <Weight>100</Weight>
    </SNode>
  </RankPerformance>
</RankingSystem>"""


# ------------------------------------------------------------------ strategies

def alpha46_bucket_test(start: str = "2002-01-01", end: str | None = None, universe: str = "SP500",
                        slippage: float = 0.1):
    """Decile bucket test of Alpha#46 rebalanced weekly (P123's minimum frequency)."""
    return p123.rank_perf(_rank_xml(ALPHA46, name="Alpha46"), start, end, universe=universe,
                          numBuckets=10, rebalFreq="Every Week", slippage=slippage,
                          benchmark="SPY", minPrice=5, outputType="ann")


def alpha46_long_short(start: str = "2002-01-01", end: str | None = None, universe: str = "SP500",
                       n: int = 50):
    """Hedged screen: long the top-n Alpha#46 names, short the bottom-n, weekly."""
    screen = {
        "type": "stock", "universe": universe, "maxNumHoldings": n, "method": "long/short",
        "benchmark": "SPY", "ranking": {"formula": ALPHA46, "lowerIsBetter": False},
        "rules": [{"formula": r, "type": "common"} for r in ["Close(0) >= 5"]],
    }
    return p123.screen_backtest(screen, start, end, rebalFreq="Every Week", slippage=0.1, transPrice=4)


def trend_factor_via_upload(signal: pd.DataFrame, start: str, end: str | None = None,
                            universe: str = "Prussell3000"):
    """Upload the Python trend-factor signal (month-end, date × ticker) and run a
    quintile bucket test on P123 with its own point-in-time universe and prices."""
    from .strategies.trend_factor import p123_signal_frame

    p123.upload_stock_factor("TrendFactor", p123_signal_frame(signal),
                             "Han, Zhou & Zhu (2016) trend factor expected return")
    return p123.rank_perf(_rank_xml('StockFactor("TrendFactor")', name="TrendFactor"), start, end,
                          universe=universe, numBuckets=5, rebalFreq="Every 4 Weeks",
                          minPrice=5, outputType="ann")


def volume_shock_bucket_test(start: str = "2005-01-01", end: str | None = None,
                             universe: str = "Prussell3000"):
    """Research-only: weekly buckets on the volume shock. Not the overnight trade —
    P123 cannot hold close→open; use alfamoney.strategies.volume_shocks for that."""
    return p123.rank_perf(_rank_xml(VOLUME_SHOCK, name="VolumeShock"), start, end, universe=universe,
                          numBuckets=10, rebalFreq="Every Week", minPrice=5, outputType="ann")


def monthly_mktcap(start: str, end: str, universe: str = "Prussell3000") -> pd.DataFrame:
    """Point-in-time market cap (date × ticker) at month ends, for Han (2022) size
    dummies/value weights and the trend-factor size filter (instead of the ADV proxy)."""
    long = p123.universe_panel({"MktCap": "MktCap"}, start, end, universe=universe, freq="ME")
    return p123.to_wide(long, "MktCap")


def ltr_features_from_p123(start: str, end: str, universe: str = "Prussell3000") -> pd.DataFrame:
    """Alternative feature source for Learning to Rank / deep momentum (monthly)."""
    return p123.universe_panel(LTR_P123_FEATURES, start, end, universe=universe, freq="ME")
