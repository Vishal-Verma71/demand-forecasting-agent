"""External trend signals (free: pytrends). Graceful fallback."""
from __future__ import annotations

import time

TREND_MAP = {
    "FOODS": "grocery demand",
    "HOBBIES": "toys hobbies",
    "HOUSEHOLD": "home cleaning supplies",
}


def _fetch_live_trend(q: str, timeframe: str) -> dict:
    from pytrends.request import TrendReq

    pt = TrendReq(hl="en-US", tz=360, timeout=(5, 8))
    pt.build_payload([q], cat=0, timeframe=timeframe, geo="US", gprop="")
    df = pt.interest_over_time()
    if df is None or df.empty:
        raise RuntimeError("empty trend response")
    series = df[q]
    latest = float(series.iloc[-1])
    baseline = float(series.mean())
    delta_pct = (latest - baseline) / max(baseline, 1) * 100
    direction = "up" if delta_pct > 5 else ("down" if delta_pct < -5 else "flat")
    return dict(latest=round(latest, 1), baseline=round(baseline, 1),
                delta_pct=round(delta_pct, 1), direction=direction,
                source="Google Trends (pytrends)")


def query_trend(category: str, timeframe: str = "today 3-m") -> dict:
    q = TREND_MAP.get(category.upper(), category)
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_live_trend, q, timeframe)
            result = future.result(timeout=8)
        return dict(category=category, query=q, **result)
    except Exception as e:  # network / rate-limit / missing dep / timeout
        # deterministic plausible fallback so the agent always works
        seed_val = sum(ord(c) for c in category) % 9
        delta = (seed_val - 4) * 3.5
        return dict(category=category, query=q, latest=round(50 + delta, 1), baseline=50.0,
                    delta_pct=round(delta, 1),
                    direction="up" if delta > 5 else ("down" if delta < -5 else "flat"),
                    source=f"fallback (offline): {type(e).__name__}")
