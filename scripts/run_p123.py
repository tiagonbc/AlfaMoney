"""Run the Portfolio123 versions (needs P123_API_ID / P123_API_KEY).

    python scripts/run_p123.py alpha46
    python scripts/run_p123.py volume_shock
    python scripts/run_p123.py trend_factor   # needs Norgate for the signal, P123 for the test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alfamoney import p123_strategies as ps  # noqa: E402


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "alpha46"
    if what == "alpha46":
        res = {"buckets": ps.alpha46_bucket_test(), "long_short": ps.alpha46_long_short()}
    elif what == "volume_shock":
        res = ps.volume_shock_bucket_test()
    elif what == "trend_factor":
        from alfamoney.data import norgate
        from alfamoney.strategies import trend_factor
        from alfamoney.strategies.common import monthly, tradable_mask

        panel = norgate.load_equity_panel("Russell 3000", "1995-01-01")
        mask_m = monthly(tradable_mask(panel, size_pct_cut=0.10).astype(float)).astype(bool)
        sig, _ = trend_factor.trend_signal(panel["Close"], mask_m)
        res = ps.trend_factor_via_upload(sig.dropna(how="all"), "2000-01-01")
    else:
        raise SystemExit(f"unknown: {what}")
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"p123_{what}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
