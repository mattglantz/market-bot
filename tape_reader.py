"""
Order Flow / Tape Reader for ES Futures.

Analyzes time & sales data to detect institutional activity:
- Large trade detection (>= 50 lots)
- Aggressive buyer/seller ratio
- Absorption patterns at key levels
- Volume clustering

This module provides the infrastructure. The actual IBKR tick-by-tick
subscription is opt-in — call TapeReader.start() to begin collecting data.

Usage:
    reader = TapeReader(ibkr_client)
    reader.start()  # begins background collection

    # Later, in your analysis cycle:
    summary = reader.get_summary()
    # Returns: {"large_buys": 12, "large_sells": 8, ...}
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from bot_config import logger, now_et


@dataclass
class TapeEntry:
    """A single time & sales record."""
    timestamp: datetime
    price: float
    size: int
    side: str  # "BUY" or "SELL" (inferred from aggressor)


@dataclass
class TapeSummary:
    """Aggregated tape statistics for a time window."""
    window_minutes: int = 30
    total_trades: int = 0
    total_volume: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    large_buys: int = 0       # trades >= 50 lots
    large_sells: int = 0
    large_buy_volume: int = 0
    large_sell_volume: int = 0
    aggressive_ratio: float = 0.5  # buy_vol / total_vol
    max_single_trade: int = 0
    absorption_detected: bool = False
    absorption_side: str = "NONE"  # "BUY" or "SELL" absorption

    @property
    def bias(self) -> str:
        """Directional bias from tape flow."""
        if self.total_volume == 0:
            return "NEUTRAL"
        ratio = self.buy_volume / self.total_volume
        if ratio > 0.6:
            return "BULLISH"
        elif ratio < 0.4:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def large_trade_bias(self) -> str:
        """Bias from large trades only (institutional flow)."""
        total_large = self.large_buy_volume + self.large_sell_volume
        if total_large == 0:
            return "NEUTRAL"
        ratio = self.large_buy_volume / total_large
        if ratio > 0.6:
            return "INSTITUTIONAL BUY"
        elif ratio < 0.4:
            return "INSTITUTIONAL SELL"
        return "NEUTRAL"

    def to_dict(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "total_volume": self.total_volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "large_buys": self.large_buys,
            "large_sells": self.large_sells,
            "aggressive_ratio": round(self.aggressive_ratio, 3),
            "bias": self.bias,
            "large_trade_bias": self.large_trade_bias,
            "max_single_trade": self.max_single_trade,
            "absorption_detected": self.absorption_detected,
            "absorption_side": self.absorption_side,
        }


class TapeReader:
    """
    Collects and analyzes ES time & sales data.

    Uses a rolling window (default 30 minutes) of tape entries.
    Large trades (>= LARGE_THRESHOLD lots) are tracked separately
    to detect institutional flow.
    """

    LARGE_THRESHOLD = 50  # contracts
    ABSORPTION_WINDOW = 60  # seconds to check for absorption
    ABSORPTION_MIN_VOLUME = 200  # min contracts absorbed at a level

    def __init__(self, ibkr_client=None, window_minutes: int = 30):
        self._ibkr = ibkr_client
        self._window_minutes = window_minutes
        self._entries: deque = deque(maxlen=50000)  # Rolling buffer
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_entry(self, price: float, size: int, side: str):
        """Manually add a tape entry (for testing or external data sources)."""
        entry = TapeEntry(
            timestamp=now_et(),
            price=price,
            size=size,
            side=side.upper(),
        )
        with self._lock:
            self._entries.append(entry)

    def get_summary(self, window_minutes: int = None) -> TapeSummary:
        """Get aggregated tape statistics for the specified time window."""
        window = window_minutes or self._window_minutes
        cutoff = now_et() - timedelta(minutes=window)

        summary = TapeSummary(window_minutes=window)

        with self._lock:
            entries = [e for e in self._entries if e.timestamp >= cutoff]

        if not entries:
            return summary

        summary.total_trades = len(entries)

        for e in entries:
            summary.total_volume += e.size
            if e.side == "BUY":
                summary.buy_volume += e.size
                if e.size >= self.LARGE_THRESHOLD:
                    summary.large_buys += 1
                    summary.large_buy_volume += e.size
            else:
                summary.sell_volume += e.size
                if e.size >= self.LARGE_THRESHOLD:
                    summary.large_sells += 1
                    summary.large_sell_volume += e.size

            if e.size > summary.max_single_trade:
                summary.max_single_trade = e.size

        if summary.total_volume > 0:
            summary.aggressive_ratio = summary.buy_volume / summary.total_volume

        # Check for absorption patterns
        summary.absorption_detected, summary.absorption_side = self._detect_absorption(entries)

        return summary

    def _detect_absorption(self, entries: List[TapeEntry]) -> tuple:
        """
        Detect absorption: large volume at a price level without price moving.
        This indicates a resting order absorbing aggressor flow.
        """
        if len(entries) < 10:
            return False, "NONE"

        # Group recent trades by price level (round to nearest 0.25 for ES)
        from collections import Counter
        price_volume: Dict[float, Dict[str, int]] = {}

        cutoff = now_et() - timedelta(seconds=self.ABSORPTION_WINDOW)
        recent = [e for e in entries if e.timestamp >= cutoff]

        for e in recent:
            level = round(e.price * 4) / 4  # Round to nearest 0.25
            if level not in price_volume:
                price_volume[level] = {"BUY": 0, "SELL": 0}
            price_volume[level][e.side] += e.size

        # Find levels with heavy one-sided volume
        for level, vol in price_volume.items():
            total = vol["BUY"] + vol["SELL"]
            if total >= self.ABSORPTION_MIN_VOLUME:
                # Heavy sell volume absorbed at a level = buy absorption (support)
                if vol["SELL"] > vol["BUY"] * 2:
                    return True, "BUY"  # Buyers absorbing sellers
                # Heavy buy volume absorbed = sell absorption (resistance)
                if vol["BUY"] > vol["SELL"] * 2:
                    return True, "SELL"  # Sellers absorbing buyers

        return False, "NONE"

    def start(self):
        """Start collecting tape data from IBKR (if available)."""
        if self._running or self._ibkr is None:
            return

        logger.info("TapeReader: Starting tick-by-tick collection (placeholder)")
        self._running = True
        # NOTE: Actual IBKR tick subscription would go here.
        # For now, this is a placeholder. To enable:
        # 1. Subscribe to IBKR tick-by-tick data for ES
        # 2. In the callback, call self.add_entry(price, size, side)
        # The infrastructure is ready — just needs the IBKR subscription wiring.

    def stop(self):
        """Stop collecting tape data."""
        self._running = False

    def clear(self):
        """Clear all collected entries."""
        with self._lock:
            self._entries.clear()

    @property
    def is_active(self) -> bool:
        return self._running and len(self._entries) > 0

    def get_prompt_text(self) -> str:
        """Format tape summary for inclusion in Claude prompt."""
        summary = self.get_summary()
        if summary.total_trades == 0:
            return "TAPE: No data available."

        lines = [
            f"TAPE READER ({summary.window_minutes}min window):",
            f"  Trades: {summary.total_trades} | Vol: {summary.total_volume:,}",
            f"  Buy Vol: {summary.buy_volume:,} ({summary.aggressive_ratio:.0%}) | Sell Vol: {summary.sell_volume:,}",
            f"  Large Buys: {summary.large_buys} ({summary.large_buy_volume:,} lots) | Large Sells: {summary.large_sells} ({summary.large_sell_volume:,} lots)",
            f"  Bias: {summary.bias} | Institutional: {summary.large_trade_bias}",
            f"  Max Single: {summary.max_single_trade} lots",
        ]

        if summary.absorption_detected:
            lines.append(f"  *** ABSORPTION DETECTED: {summary.absorption_side} side absorbing ***")

        return "\n".join(lines)
