"""
Real-time price monitor and level collection for ES futures.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

import time
import asyncio
import threading
from typing import TYPE_CHECKING, Dict

from bot_config import CFG, logger
from ibkr_client import IBKRClient

if TYPE_CHECKING:
    from journal import Journal


class PriceMonitor:
    """
    Background thread that polls ES price every 5 seconds and alerts
    when key levels are touched (gamma walls, targets, stops, prior day H/L/C).
    Debounces each level for 30 minutes to avoid spam.
    """

    def __init__(self, ibkr_host: str, ibkr_port: int, client_id: int = 11):
        self._ibkr_host = ibkr_host
        self._ibkr_port = ibkr_port
        self._client_id = client_id
        self._levels: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._debounce: Dict[str, float] = {}  # level_name -> timestamp of last alert
        self._running = False
        self._thread: threading.Thread = None
        self._ibkr: IBKRClient = None
        self.DEBOUNCE_SECS = 1800  # 30 min
        self.POLL_INTERVAL = 5     # seconds
        self.PROXIMITY_PTS = 2.0   # alert within 2 pts of level

    def update_levels(self, levels: Dict[str, float]):
        """Update key levels from main thread. Thread-safe."""
        with self._lock:
            self._levels = {k: v for k, v in levels.items() if v and v > 0}

    def start(self):
        """Start the monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("PriceMonitor started (polling every 5s)")

    def stop(self):
        self._running = False

    def _run(self):
        """Main monitoring loop."""
        # Lazy import to avoid circular dependency
        from telegram_bot import send_telegram

        # ib_insync needs an event loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Create a separate IBKR connection for this thread
        try:
            self._ibkr = IBKRClient(
                host=self._ibkr_host,
                port=self._ibkr_port,
                client_id=self._client_id,
            )
            if not self._ibkr.connected:
                logger.warning("PriceMonitor: IBKR connection failed -- monitor disabled.")
                return
        except Exception as e:
            logger.warning(f"PriceMonitor: Init failed: {e}")
            return

        consecutive_errors = 0
        _spx_price_cache = 0.0
        _spx_cache_ts = 0.0
        while self._running:
            try:
                es_price = self._ibkr.get_live_price("ES")
                if es_price <= 0:
                    time.sleep(self.POLL_INTERVAL)
                    continue

                # Fetch SPX price for gamma wall checks (cache 10s to reduce API calls)
                now_ts = time.time()
                if now_ts - _spx_cache_ts > 10:
                    spx = self._ibkr.get_live_price("SPX")
                    if spx > 0:
                        _spx_price_cache = spx
                    _spx_cache_ts = now_ts

                with self._lock:
                    levels = dict(self._levels)

                alerted_prices = set()  # prevent duplicate alerts for levels at same price
                for name, level in levels.items():
                    # Use SPX price for gamma levels, ES price for everything else
                    is_gamma = "Gamma" in name
                    check_price = _spx_price_cache if (is_gamma and _spx_price_cache > 0) else es_price
                    ticker_label = "SPX" if is_gamma else "ES"

                    if abs(check_price - level) <= self.PROXIMITY_PTS:
                        last_alert = self._debounce.get(name, 0)
                        # Round to nearest 5 pts to group nearby levels
                        price_bucket = round(level / 5) * 5
                        if now_ts - last_alert > self.DEBOUNCE_SECS and price_bucket not in alerted_prices:
                            self._debounce[name] = now_ts
                            alerted_prices.add(price_bucket)
                            direction = "above" if check_price >= level else "below"
                            # Proximity alerts for trade stops/targets say "approaching"
                            is_trade_level = name.startswith("Trade ")
                            if is_trade_level:
                                label = "APPROACHING"
                                emoji = "\u26a0\ufe0f"
                            else:
                                label = "LEVEL TOUCH:"
                                emoji = ""
                            send_telegram(
                                f"*{label}* {name}\n"
                                f"{emoji}{ticker_label} {check_price:.2f} {direction} {level:.2f}"
                            )
                            logger.info(f"PriceMonitor: {name} {'approaching' if is_trade_level else 'touched'} @ {ticker_label} {check_price:.2f} (level {level:.2f})")

                # Prune stale debounce entries to prevent memory leak
                cutoff_ts = now_ts - (2 * self.DEBOUNCE_SECS)
                self._debounce = {k: v for k, v in self._debounce.items() if v > cutoff_ts}

                consecutive_errors = 0
                time.sleep(self.POLL_INTERVAL)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3 or consecutive_errors % 20 == 0:
                    logger.warning(f"PriceMonitor error ({consecutive_errors}): {e}")
                # Attempt IBKR reconnect after sustained errors
                if consecutive_errors >= 10 and consecutive_errors % 10 == 0:
                    logger.warning(f"PriceMonitor: {consecutive_errors} consecutive errors, attempting reconnect...")
                    try:
                        if self._ibkr and self._ibkr.ib:
                            self._ibkr.disconnect()
                        self._ibkr = IBKRClient(
                            host=self._ibkr_host,
                            port=self._ibkr_port,
                            client_id=self._client_id,
                        )
                        if self._ibkr.connected:
                            logger.info("PriceMonitor: IBKR reconnected successfully.")
                            consecutive_errors = 0
                        else:
                            logger.warning("PriceMonitor: reconnect failed -- will retry.")
                    except Exception as re:
                        logger.warning(f"PriceMonitor: reconnect attempt failed: {re}")
                time.sleep(self.POLL_INTERVAL * 2)

        # Cleanup
        if self._ibkr and self._ibkr.ib:
            try:
                self._ibkr.disconnect()
            except Exception:
                pass


def _collect_monitor_levels(metrics: dict, journal: 'Journal') -> Dict[str, float]:
    """Gather key levels for the price monitor from the latest analysis."""
    levels = {}

    # Gamma walls
    g_call = metrics.get("g_call")
    g_put = metrics.get("g_put")
    if g_call and g_call != "N/A":
        try:
            levels["Gamma Call Wall"] = float(g_call)
        except (ValueError, TypeError):
            pass
    if g_put and g_put != "N/A":
        try:
            levels["Gamma Put Wall"] = float(g_put)
        except (ValueError, TypeError):
            pass

    # Prior day H/L/C
    prior = metrics.get("prior", {})
    for key, label in [("prev_high", "Prior High"), ("prev_low", "Prior Low"), ("prev_close", "Prior Close")]:
        val = prior.get(key)
        if val and val != "N/A":
            try:
                levels[label] = float(val)
            except (ValueError, TypeError):
                pass

    # VWAP bands
    vwap_levels = metrics.get("vwap_levels", {})
    for key, label in [("upper_2", "VWAP +2SD"), ("lower_2", "VWAP -2SD")]:
        val = vwap_levels.get(key)
        if val and val != "N/A":
            try:
                levels[label] = float(val)
            except (ValueError, TypeError):
                pass

    # Active trade target/stop from journal
    open_trades = journal.get_open_trades()
    for t in open_trades:
        try:
            tid = t.get("id", "?")
            levels[f"Trade {tid} Target"] = float(t["target"])
            levels[f"Trade {tid} Stop"] = float(t["stop"])
        except (ValueError, TypeError, KeyError):
            pass

    return levels
