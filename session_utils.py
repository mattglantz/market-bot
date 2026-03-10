"""
Session phase detection, economic calendar, and position sizing utilities.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Optional, Tuple

from bot_config import logger, now_et, CFG, HTTP, _cache_lock, ET


# =================================================================
# --- SESSION PHASE ---
# =================================================================

def get_session_phase() -> str:
    t = now_et().time()
    if t < dtime(4, 0):
        return "OVERNIGHT"
    elif t < dtime(9, 30):
        return "PRE-MARKET"
    elif t < dtime(10, 30):
        return "OPENING DRIVE"
    elif t < dtime(12, 0):
        return "MID-MORNING"
    elif t < dtime(13, 30):
        return "LUNCH CHOP"
    elif t < dtime(15, 50):
        return "AFTERNOON / POWER HOUR"
    elif t < dtime(16, 15):
        return "CLOSING IMBALANCE"
    elif t < dtime(18, 0):
        return "POST-CLOSE"
    else:
        return "GLOBEX OVERNIGHT"


# =================================================================
# --- ECONOMIC CALENDAR ---
# =================================================================

_econ_calendar_cache: list = []
_econ_calendar_fetched: Optional[datetime] = None


def _fetch_economic_calendar() -> list:
    """Fetch today's high-impact economic events. Caches for 4 hours."""
    global _econ_calendar_cache, _econ_calendar_fetched
    now = now_et()

    # Return cache if fresh (fetched today within last 4 hours)
    with _cache_lock:
        if _econ_calendar_fetched and (now - _econ_calendar_fetched).total_seconds() < 14400:
            return list(_econ_calendar_cache)

    events = []
    try:
        import requests
        today_str = now.strftime("%Y-%m-%d")
        url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                event_date = str(item.get("date", ""))[:10]
                country = item.get("country", "")
                impact = item.get("impact", "").lower()
                title = item.get("title", "")
                event_time_str = str(item.get("date", ""))

                if event_date != today_str:
                    continue
                if country != "USD":
                    continue
                if impact not in ("high", "medium"):
                    continue

                try:
                    from dateutil import parser as dtparser
                    event_dt = dtparser.parse(event_time_str)
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=ET)
                    else:
                        event_dt = event_dt.astimezone(ET)
                except Exception:
                    continue

                events.append({
                    "title": title,
                    "time": event_dt,
                    "impact": impact,
                })
                logger.info(f"Econ event: {title} @ {event_dt.strftime('%H:%M')} ET ({impact})")

    except Exception as e:
        logger.warning(f"Economic calendar fetch failed: {e}")

    with _cache_lock:
        _econ_calendar_cache = events
        _econ_calendar_fetched = now
    return events


def is_news_approaching() -> Tuple[bool, Optional[str], Optional[datetime], Optional[str]]:
    """Returns (is_approaching, event_label, event_datetime, impact) if a real economic event is within 2 hours.
    impact is 'high' or 'medium'."""
    now = now_et()
    events = _fetch_economic_calendar()

    for ev in events:
        event_dt = ev["time"]
        minutes_until = (event_dt - now).total_seconds() / 60
        if 0 < minutes_until <= 120:
            impact_tag = "HIGH" if ev["impact"] == "high" else "MED"
            label = f"[{impact_tag}] {ev['title']} @ {event_dt.strftime('%H:%M')} ET"
            return True, label, event_dt, ev["impact"]

    return False, None, None, None


# =================================================================
# --- POSITION SIZING ---
# =================================================================

def position_suggestion(confidence: int, flat_threshold: int = 60) -> Tuple[int, str]:
    """
    Map confidence % to a fixed contract tier.

    60%+ = 1 ct, 70%+ = 2 ct, 80%+ = 3 ct, 90%+ = 4 ct.
    """
    if confidence < flat_threshold:
        return 0, f"NO TRADE (Conf {confidence}% < {flat_threshold}%)"

    if confidence >= 90:
        contracts = 4
    elif confidence >= 80:
        contracts = 3
    elif confidence >= 70:
        contracts = 2
    else:
        contracts = 1
    return contracts, f"{contracts} ct ({confidence}% conf)"
