"""External trend signals (free: Google Trends via pytrends).

Google Trends is rate-limited, so results are cached for an hour.  When it is
unreachable the tool says so plainly and returns no numbers, rather than
inventing a signal.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

TREND_MAP = {
    "FOODS": "grocery delivery",
    "HOBBIES": "toys",
    "HOUSEHOLD": "cleaning supplies",
}
_CACHE_TTL_SECONDS = 3600
_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def _fetch_live_trend(q: str, timeframe: str) -> dict:
    from pytrends.request import TrendReq

    pt = TrendReq(hl="en-US", tz=360, timeout=(5, 8))
    pt.build_payload([q], cat=0, timeframe=timeframe, geo="US", gprop="")
    df = pt.interest_over_time()
    if df is None or df.empty:
        raise RuntimeError("empty trend response")
    series = df[q]
    latest = float(series.tail(4).mean())     # last ~4 weeks, smoother than one point
    baseline = float(series.mean())
    delta_pct = (latest - baseline) / max(baseline, 1) * 100
    direction = "up" if delta_pct > 5 else ("down" if delta_pct < -5 else "flat")
    return dict(latest=round(latest, 1), baseline=round(baseline, 1),
                delta_pct=round(delta_pct, 1), direction=direction,
                source="Google Trends (pytrends), US, last 12 months")


def query_trend(category: str, timeframe: str = "today 12-m") -> dict:
    category = category.upper()
    q = TREND_MAP.get(category, category.lower())
    key = (q, timeframe)
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return dict(category=category, query=q, **cached[1])
    try:
        # No context manager: its exit would wait for a hung request and defeat the timeout.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            result = executor.submit(_fetch_live_trend, q, timeframe).result(timeout=10)
        finally:
            executor.shutdown(wait=False)
        _cache[key] = (time.time(), result)
        return dict(category=category, query=q, **result)
    except Exception as e:  # network / rate-limit / missing dep / timeout
        return dict(category=category, query=q, latest=None, baseline=None, delta_pct=None,
                    direction="unavailable",
                    source=f"Google Trends unavailable ({type(e).__name__}); no external signal used")
