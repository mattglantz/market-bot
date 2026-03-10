"""
Trade status constants for the trading bot.
Single source of truth — all modules import from here.
"""

# Individual statuses
WIN = "WIN"
LOSS = "LOSS"
TIME_EXIT = "TIME-EXIT"
STOP = "STOP"
BREAKEVEN = "BREAKEVEN"
FLOATING = "Floating"
OPEN = "Open"
SKIPPED = "SKIPPED"  # R:R gate or other filter blocked the trade

# Grouped sets for Python `in` checks (O(1) lookup)
CLOSED_STATUSES = frozenset({WIN, LOSS, TIME_EXIT, STOP, BREAKEVEN})
LOSS_STATUSES = frozenset({LOSS, TIME_EXIT, STOP})
DECIDED_STATUSES = frozenset({WIN, LOSS, TIME_EXIT, STOP})  # excludes BREAKEVEN (for accuracy tracker)

# Pre-built SQL IN clause strings (do NOT use frozenset in SQL)
CLOSED_SQL = "('WIN','LOSS','TIME-EXIT','STOP','BREAKEVEN')"
LOSS_SQL = "('LOSS','TIME-EXIT','STOP')"
DECIDED_SQL = "('WIN','LOSS','TIME-EXIT','STOP')"


# --- Direction helpers (single source of truth) ---

def is_long(verdict: str) -> bool:
    """Return True if the verdict indicates a long/bullish position."""
    v = str(verdict).upper()
    return "BULL" in v or v in ("LONG", "BUY")


def is_short(verdict: str) -> bool:
    """Return True if the verdict indicates a short/bearish position."""
    v = str(verdict).upper()
    return "BEAR" in v or v in ("SHORT", "SELL")
