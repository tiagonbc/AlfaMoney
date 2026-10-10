"""Portfolio123 helpers (DataMiner API via ``p123api``).

Credentials are read from the environment: ``P123_API_ID`` and ``P123_API_KEY``.
P123 is used for two things here:
  * point-in-time fundamentals / universe panels (``data_universe``) that Norgate
    does not provide;
  * running the screen/ranking versions of the stock strategies directly on the
    P123 backtester (``screen_backtest`` / ``rank_perf``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("ALFAMONEY_CACHE", ".cache")) / "p123"


def client():
    import p123api

    api_id, api_key = os.environ.get("P123_API_ID"), os.environ.get("P123_API_KEY")
    if not api_id or not api_key:
        raise RuntimeError("Set P123_API_ID and P123_API_KEY to use Portfolio123.")
    return p123api.Client(api_id=api_id, api_key=api_key)


def weekend_dates(start: str, end: str, freq: str = "W-SAT") -> list[str]:
    """P123 recommends weekend as-of dates (data as of Friday close)."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq=freq)]


def universe_panel(
    formulas: dict[str, str],
    start: str,
    end: str,
    universe: str = "Prussell3000",
    freq: str = "W-SAT",
    chunk: int = 4,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long frame [date, p123Uid, ticker, <names...>] for a universe over time.

    ``formulas`` maps output column name -> P123 formula.
    """
    names, fmls = list(formulas), list(formulas.values())
    key = f"{universe}_{start}_{end}_{freq}_{abs(hash(tuple(fmls)))}"
    path = CACHE_DIR / f"{key}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    cl = client()
    dates = weekend_dates(start, end, freq)
    frames = []
    for i in range(0, len(dates), chunk):
        res = cl.data_universe(
            {
                "universe": universe,
                "asOfDts": dates[i:i + chunk],
                "formulas": fmls,
                "names": names,
                "pitMethod": "Complete",
                "precision": None,
                "includeNames": False,
            },
            to_pandas=True,
        )
        frames.append(res)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c if c not in ("Date", "date") else "date" for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    return df


def to_wide(long: pd.DataFrame, value: str, id_col: str = "Ticker") -> pd.DataFrame:
    return long.pivot_table(index="date", columns=id_col, values=value, aggfunc="last")


def screen_backtest(screen: dict, start: str, end: str | None = None, **kw) -> dict:
    """Run a screen on the P123 backtester. ``screen`` follows the API schema."""
    params = {"screen": screen, "startDt": start, **kw}
    if end:
        params["endDt"] = end
    return client().screen_backtest(params, True)


def rank_perf(ranking_xml: str, start: str, end: str | None = None, **kw) -> dict:
    """Upload a ranking system to ApiRankingSystem and run the bucket test."""
    cl = client()
    cl.rank_update({"nodes": ranking_xml, "type": "stock", "rankingMethod": 2})
    params = {"rankingSystem": "ApiRankingSystem", "startDt": start, **kw}
    if end:
        params["endDt"] = end
    return cl.rank_perf(params)


def upload_stock_factor(name: str, frame: pd.DataFrame, description: str = "") -> int:
    """Upload a long frame [date, ticker, value] as a P123 custom stock factor."""
    cl = client()
    fid = cl.stock_factor_create_update({"name": name, "description": description})["id"]
    tmp = CACHE_DIR / f"factor_{name}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    frame[["date", "ticker", "value"]].to_csv(tmp, index=False)
    cl.stock_factor_upload(file=str(tmp), factor_id=fid, column_separator="comma",
                           existing_data="overwrite", date_format="yyyy-mm-dd")
    return fid
