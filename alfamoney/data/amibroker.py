"""AmiBroker database (Norgate Data plugin) as data source.

AmiBroker cannot be queried directly from Python at speed, so the bridge is a CSV
export made inside AmiBroker by ``amibroker/AlfaMoney_Export_Norgate.afl``
(Exploration over a watchlist, range "All quotes"). The export writes one file per
symbol, keeping Norgate's point-in-time information:

  Equities:   Date,Open,High,Low,Close,Volume,UnadjClose,InSPX,InRUI,InRUA,InRUT,InNDX
  Futures/ETF: Date,Open,High,Low,Close,Volume,OpenInt,UnadjClose

(UnadjClose = NorgateOriginalCloseTimeSeries(); In* = NorgateIndexConstituentTimeSeries();
padded bars are dropped.)  This module exposes the same API as
:mod:`alfamoney.data.norgate`, so every strategy runs unchanged on AmiBroker's data.

Configure the export folder with ``set_export_dir(path)`` or the environment variable
``ALFAMONEY_AMI_EXPORT``. Expected layout (one sub-folder per exploration run):

  <export>/equities/*.csv     Russell 3000 Current & Past (or S&P 500 Current & Past)
  <export>/etfs/*.csv         SPY, TLT, GLD, DBC, UUP, BIL, ...
  <export>/futures/*.csv      continuous (&ES_CCB, ...) and dated contracts (ES-2024Z, ...)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .norgate import FUTURES_UNIVERSE_64

_EXPORT_DIR: Path | None = None

# Norgate index name -> column written by the export AFL
INDEX_COLUMNS = {
    "S&P 500": "InSPX", "$SPX": "InSPX",
    "Russell 1000": "InRUI", "$RUI": "InRUI",
    "Russell 3000": "InRUA", "$RUA": "InRUA",
    "Russell 2000": "InRUT", "$RUT": "InRUT",
    "Nasdaq-100": "InNDX", "NASDAQ-100": "InNDX", "$NDX": "InNDX",
}

MONTH_CODES = "FGHJKMNQUVXZ"
CONTRACT_RE = re.compile(r"^(?P<root>.+)-(?P<year>\d{4})(?P<month>[FGHJKMNQUVXZ])$")


def set_export_dir(path: str | os.PathLike) -> None:
    global _EXPORT_DIR
    _EXPORT_DIR = Path(path)


def export_dir() -> Path:
    d = _EXPORT_DIR or (Path(os.environ["ALFAMONEY_AMI_EXPORT"]) if "ALFAMONEY_AMI_EXPORT" in os.environ else None)
    if d is None:
        raise RuntimeError("Set the AmiBroker export folder: amibroker.set_export_dir(...) "
                           "or ALFAMONEY_AMI_EXPORT")
    return d


def _ts(x) -> pd.Timestamp:
    """Timestamp clipped to pandas' nanosecond range (the API uses '2999-01-01' as 'open end')."""
    t = pd.Timestamp(x)
    return min(max(t, pd.Timestamp("1900-01-01")), pd.Timestamp("2262-01-01"))


def safe_name(symbol: str) -> str:
    """Same character substitution as SafeName() in the export AFL."""
    for ch in '/\\:*?"<>|':
        symbol = symbol.replace(ch, "_")
    return symbol


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _find(symbol: str) -> Path:
    root = export_dir()
    name = safe_name(symbol) + ".csv"
    for sub in ("", "etfs", "equities", "futures"):
        p = root / sub / name if sub else root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{symbol}: {name} not found under {root} (run the export AFL on it)")


def price_frame(symbol: str, start: str = "1900-01-01", end: str = "2999-01-01",
                adjustment: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """OHLCV for one exported symbol. Price adjustment is whatever the Norgate plugin
    was configured with in AmiBroker when exporting (``adjustment`` is ignored)."""
    df = _read(_find(symbol))
    if "UnadjClose" in df:
        df["Unadjusted Close"] = df["UnadjClose"]
    return df.loc[_ts(start):_ts(end)]


# --------------------------------------------------------------------------- equities

def _combined_cache(folder: Path) -> Path:
    return folder / "_alfamoney_combined.parquet"


def _load_folder_long(folder: Path, use_cache: bool = True) -> pd.DataFrame:
    """All CSVs of a folder in one long frame [Date, Symbol, ...] (cached as parquet)."""
    files = sorted(p for p in folder.glob("*.csv"))
    cache = _combined_cache(folder)
    newest = max((p.stat().st_mtime for p in files), default=0)
    if use_cache and cache.exists() and cache.stat().st_mtime >= newest:
        return pd.read_parquet(cache)
    frames = []
    for p in files:
        df = _read(p)
        if df.empty:
            continue
        df["Symbol"] = p.stem
        frames.append(df.reset_index())
    long = pd.concat(frames, ignore_index=True)
    if use_cache:
        try:
            long.to_parquet(cache)
        except Exception:
            pass
    return long


def load_equity_panel(
    index_name: str = "Russell 3000",
    start: str = "1990-01-01",
    end: str = "2999-01-01",
    symbols: Iterable[str] | None = None,
    fields: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume"),
    adjustment: str | None = None,
    folder: str | os.PathLike | None = None,
) -> dict[str, pd.DataFrame]:
    """Same output as norgate.load_equity_panel: wide frames per field + ``member``
    (point-in-time index membership) + ``UnadjClose``."""
    col = INDEX_COLUMNS.get(index_name)
    if col is None:
        raise ValueError(f"unknown index {index_name!r}; choose one of {sorted(set(INDEX_COLUMNS))}")
    folder = Path(folder) if folder else export_dir() / "equities"
    long = _load_folder_long(folder)
    long = long[(long["Date"] >= _ts(start)) & (long["Date"] <= _ts(end))]
    if symbols is not None:
        long = long[long["Symbol"].isin(set(map(safe_name, symbols)))]
    if col not in long:
        raise ValueError(f"column {col} missing: re-export with 'Asset type = Equities'")
    ever = long.groupby("Symbol")[col].max()
    long = long[long["Symbol"].isin(ever.index[ever > 0])]
    if long.empty:
        raise ValueError(f"no exported symbol was ever a member of {index_name} ({col}) in the period")
    panel = {}
    for f in (*fields, "UnadjClose"):
        if f in long:
            panel[f] = long.pivot(index="Date", columns="Symbol", values=f).sort_index()
    panel["member"] = (long.pivot(index="Date", columns="Symbol", values=col)
                       .reindex_like(panel["Close"]).fillna(0).astype(bool))
    return panel


# --------------------------------------------------------------------------- futures

def continuous_future(root: str, start: str = "1980-01-01", end: str = "2999-01-01",
                      suffix: str = "_CCB") -> pd.DataFrame:
    return price_frame(f"&{root}{suffix}", start, end)


def load_futures_panel(roots: Iterable[str] = FUTURES_UNIVERSE_64, start: str = "1980-01-01",
                       end: str = "2999-01-01", suffix: str = "_CCB") -> dict[str, pd.DataFrame]:
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in ("Open", "High", "Low", "Close", "Volume")}
    for r in roots:
        try:
            df = continuous_future(r, start, end, suffix)
        except FileNotFoundError:
            continue
        for f in out:
            if f in df:
                out[f][r] = df[f]
    return {k: pd.DataFrame(v).sort_index() for k, v in out.items() if v}


def contract_month(symbol: str) -> pd.Timestamp | None:
    m = CONTRACT_RE.match(symbol)
    if not m:
        return None
    return pd.Timestamp(int(m["year"]), MONTH_CODES.index(m["month"]) + 1, 1)


def futures_contracts(root: str) -> list[str]:
    folder = export_dir() / "futures"
    syms = [p.stem for p in folder.glob(f"{safe_name(root)}-*.csv")]
    syms = [s for s in syms if CONTRACT_RE.match(s) and CONTRACT_RE.match(s)["root"] == root]
    return sorted(syms, key=contract_month)


def load_contract_strip(root: str, start: str = "1980-01-01", end: str = "2999-01-01") -> dict[str, pd.DataFrame]:
    """Same output as norgate.load_contract_strip. Expiry = last exported bar for
    expired contracts; for contracts still trading at the end of the export it is
    estimated as the day before the contract month (conservative for first notice)."""
    close, oi, vol, expiry = {}, {}, {}, {}
    last_global = None
    for c in futures_contracts(root):
        df = price_frame(c)
        if df.empty:
            continue
        close[c], vol[c] = df["Close"], df.get("Volume")
        if "OpenInt" in df:
            oi[c] = df["OpenInt"]
        expiry[c] = df.index[-1]
        last_global = max(last_global or df.index[-1], df.index[-1])
    for c, e in expiry.items():
        if last_global is not None and (last_global - e).days <= 3:
            expiry[c] = contract_month(c) - pd.Timedelta(days=1)
    sl = slice(_ts(start), _ts(end))
    return {
        "Close": pd.DataFrame(close).sort_index().loc[sl],
        "OpenInterest": pd.DataFrame(oi).sort_index().loc[sl] if oi else pd.DataFrame(),
        "Volume": pd.DataFrame({k: v for k, v in vol.items() if v is not None}).sort_index().loc[sl],
        "expiry": pd.Series(expiry).sort_values(),
    }


def available_symbols(sub: str = "") -> list[str]:
    folder = export_dir() / sub if sub else export_dir()
    return sorted(p.stem for p in folder.glob("*.csv"))


__all__ = ["set_export_dir", "price_frame", "load_equity_panel", "continuous_future",
           "load_futures_panel", "futures_contracts", "load_contract_strip", "available_symbols"]
