"""Norgate Data loaders.

Requires the Norgate Data Updater (NDU) running on the same machine (Windows)
and the ``norgatedata`` package. Everything returned is a wide pandas panel
(index = dates, columns = symbols) so the strategies can stay vectorised.

Survivorship bias: equity universes are built from Norgate's historical index
constituent flags, which include delisted securities ("US Equities Delisted"
database) when the subscription covers them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

CACHE_DIR = Path(os.environ.get("ALFAMONEY_CACHE", ".cache")) / "norgate"


def _nd():
    import norgatedata  # imported lazily so the rest of the package works without NDU

    return norgatedata


def _cache_path(kind: str, key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return CACHE_DIR / kind / f"{safe}.parquet"


def price_frame(
    symbol: str,
    start: str = "1990-01-01",
    end: str = "2999-01-01",
    adjustment: str = "TOTALRETURN",
    use_cache: bool = True,
) -> pd.DataFrame:
    """OHLCV(+Turnover, Unadjusted Close, Dividend) for one symbol."""
    path = _cache_path(f"px_{adjustment}", symbol)
    if use_cache and path.exists():
        df = pd.read_parquet(path)
        return df.loc[start:end]
    nd = _nd()
    df = nd.price_timeseries(
        symbol,
        stock_price_adjustment_setting=getattr(nd.StockPriceAdjustmentType, adjustment),
        padding_setting=nd.PaddingType.NONE,
        start_date=start,
        end_date=end,
        timeseriesformat="pandas-dataframe",
    )
    df.index = pd.to_datetime(df.index)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    return df


def index_membership(symbol: str, index_name: str, start: str, end: str) -> pd.Series:
    """Daily 0/1 flag: was ``symbol`` a member of ``index_name`` on each date."""
    nd = _nd()
    df = nd.index_constituent_timeseries(
        symbol, index_name, start_date=start, end_date=end,
        timeseriesformat="pandas-dataframe",
    )
    df.index = pd.to_datetime(df.index)
    return df.iloc[:, 0].astype(bool)


def equity_symbols(include_delisted: bool = True) -> list[str]:
    nd = _nd()
    syms = list(nd.database_symbols("US Equities"))
    if include_delisted:
        syms += list(nd.database_symbols("US Equities Delisted"))
    return syms


def load_equity_panel(
    index_name: str = "Russell 3000",
    start: str = "1990-01-01",
    end: str = "2999-01-01",
    symbols: Iterable[str] | None = None,
    fields: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
    adjustment: str = "TOTALRETURN",
) -> dict[str, pd.DataFrame]:
    """Survivorship-bias-free panel for a historical index universe.

    Returns a dict of wide frames: one per requested field plus ``member``
    (point-in-time index membership) and ``UnadjClose`` (for price filters).
    Typical index names: "S&P 500", "Russell 1000", "Russell 3000",
    "Nasdaq-100", "Russell 2000".
    """
    symbols = list(symbols) if symbols is not None else equity_symbols()
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in (*fields, "member", "UnadjClose")}
    for sym in symbols:
        try:
            mem = index_membership(sym, index_name, start, end)
        except Exception:
            continue
        if not mem.any():
            continue
        px = price_frame(sym, start, end, adjustment)
        if px.empty:
            continue
        for f in fields:
            if f in px:
                out[f][sym] = px[f]
        out["member"][sym] = mem
        if "Unadjusted Close" in px:
            out["UnadjClose"][sym] = px["Unadjusted Close"]
    panel = {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}
    if "member" in panel:
        panel["member"] = panel["member"].fillna(False).astype(bool)
    return panel


# --------------------------------------------------------------------------- futures

# Norgate continuous-contract conventions:
#   "&ES"       -> back-adjusted continuous (roll on volume/OI per Norgate rules)
#   "&ES_CCB"   -> back-adjusted, roll at contract-change (cash-adjusted)
#   individual contracts are e.g. "ES-2024Z" (from futures_market_session_contracts)
FUTURES_UNIVERSE_64 = [
    # equity indices
    "ES", "NQ", "YM", "RTY", "FDAX", "FESX", "FTI", "FCE", "NIY", "HSI", "SSG", "FSMI", "AP", "SXF",
    # rates
    "ZT", "ZF", "ZN", "ZB", "UB", "FGBS", "FGBM", "FGBL", "FGBX", "FBTP", "FOAT", "CGB", "YAP10", "R", "SR3",
    # FX
    "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S", "DX",
    # energy
    "CL", "HO", "RB", "NG", "BRN", "G",
    # metals
    "GC", "SI", "HG", "PL", "PA",
    # ags & softs
    "ZC", "ZW", "ZS", "ZM", "ZL", "KE", "ZO", "CT", "KC", "SB", "CC", "OJ",
    # livestock
    "LE", "HE", "GF",
]


def continuous_future(root: str, start: str = "1980-01-01", end: str = "2999-01-01",
                      suffix: str = "_CCB") -> pd.DataFrame:
    """Back-adjusted continuous contract for a futures root (e.g. 'ES')."""
    return price_frame(f"&{root}{suffix}", start, end, adjustment="NONE")


def load_futures_panel(roots: Iterable[str] = FUTURES_UNIVERSE_64, start: str = "1980-01-01",
                       end: str = "2999-01-01", suffix: str = "_CCB") -> dict[str, pd.DataFrame]:
    """Wide Open/High/Low/Close panels for continuous futures (missing roots skipped)."""
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in ("Open", "High", "Low", "Close", "Volume")}
    for r in roots:
        try:
            df = continuous_future(r, start, end, suffix)
        except Exception:
            continue
        for f in out:
            if f in df:
                out[f][r] = df[f]
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}


def futures_contracts(root: str) -> list[str]:
    """Individual (dated) contracts for a futures market, oldest first."""
    nd = _nd()
    return list(nd.futures_market_session_contracts(root))


def load_contract_strip(root: str, start: str = "1980-01-01", end: str = "2999-01-01") -> dict[str, pd.DataFrame]:
    """Close/Open Interest/Volume for every dated contract of a market (for term-structure work)."""
    close, oi, vol, expiry = {}, {}, {}, {}
    nd = _nd()
    for c in futures_contracts(root):
        try:
            df = price_frame(c, start, end, adjustment="NONE")
        except Exception:
            continue
        if df.empty:
            continue
        close[c] = df["Close"]
        if "Open Interest" in df:
            oi[c] = df["Open Interest"]
        if "Volume" in df:
            vol[c] = df["Volume"]
        try:
            expiry[c] = pd.Timestamp(nd.last_quoted_date(c))
        except Exception:
            expiry[c] = df.index[-1]
    return {
        "Close": pd.DataFrame(close).sort_index(),
        "OpenInterest": pd.DataFrame(oi).sort_index(),
        "Volume": pd.DataFrame(vol).sort_index(),
        "expiry": pd.Series(expiry).sort_values(),
    }
