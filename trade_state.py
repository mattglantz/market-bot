"""
Trade State Machine — enforces valid state transitions for trade lifecycle.

Replaces implicit if/else state transitions with a formal transition table.
Illegal transitions (e.g., WIN → FLOATING) raise InvalidTransitionError.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import trade_status as ts
from bot_config import logger


class TradeState(Enum):
    """All possible trade states. Values match trade_status.py string constants."""
    OPEN = ts.OPEN              # "Open"
    FLOATING = ts.FLOATING      # "Floating"
    WIN = ts.WIN                # "WIN"
    LOSS = ts.LOSS              # "LOSS"
    STOP = ts.STOP              # "STOP"
    TIME_EXIT = ts.TIME_EXIT    # "TIME-EXIT"
    BREAKEVEN = ts.BREAKEVEN    # "BREAKEVEN"
    SKIPPED = ts.SKIPPED        # "SKIPPED"

    @classmethod
    def from_str(cls, s: str) -> TradeState:
        """Convert legacy string status to enum. Handles case variations."""
        if not s:
            return cls.OPEN
        # Direct match first (covers exact status strings)
        for member in cls:
            if member.value == s:
                return member
        # Case-insensitive fallback
        s_upper = s.upper()
        for member in cls:
            if member.value.upper() == s_upper:
                return member
        logger.warning(f"TradeState.from_str: unknown status '{s}', defaulting to OPEN")
        return cls.OPEN

    @property
    def is_terminal(self) -> bool:
        return self.value in ts.CLOSED_STATUSES


class TradeEvent(Enum):
    """Events that trigger state transitions."""
    AUDIT = "AUDIT"                     # Regular audit check (first touch: OPEN→FLOATING)
    TARGET_HIT = "TARGET_HIT"           # Price reached target
    STOP_HIT = "STOP_HIT"              # Price reached stop
    TIME_EXPIRED = "TIME_EXPIRED"       # 2-hour time exit
    TRAIL_TRIGGERED = "TRAIL_TRIGGERED" # Trailing stop moved
    PARTIAL_CLOSE = "PARTIAL_CLOSE"     # Half position closed at target
    SCALEOUT = "SCALEOUT"              # 50% intermediate scale-out
    EOD_CLOSE = "EOD_CLOSE"            # End-of-day force close
    REVERSAL = "REVERSAL"              # Signal reversed, closing position
    PNL_UPDATE = "PNL_UPDATE"          # Floating P&L refresh (no state change)


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""
    pass


# ─── Transition Table ──────────────────────────────────────────
# Maps (current_state, event) -> allowed new_state
# None = dynamic (determined from context, e.g., P&L sign)
# Missing key = illegal transition

_TRANSITIONS: Dict[Tuple[TradeState, TradeEvent], Optional[TradeState]] = {
    # OPEN → FLOATING on first audit
    (TradeState.OPEN, TradeEvent.AUDIT): TradeState.FLOATING,
    (TradeState.OPEN, TradeEvent.PNL_UPDATE): TradeState.FLOATING,

    # OPEN can go directly to terminal (target/stop hit on first check)
    (TradeState.OPEN, TradeEvent.TARGET_HIT): TradeState.WIN,
    (TradeState.OPEN, TradeEvent.STOP_HIT): None,       # LOSS/STOP/BREAKEVEN from P&L
    (TradeState.OPEN, TradeEvent.TIME_EXPIRED): TradeState.TIME_EXIT,
    (TradeState.OPEN, TradeEvent.EOD_CLOSE): None,       # dynamic from P&L
    (TradeState.OPEN, TradeEvent.REVERSAL): None,         # dynamic from P&L

    # FLOATING → terminal states
    (TradeState.FLOATING, TradeEvent.TARGET_HIT): TradeState.WIN,
    (TradeState.FLOATING, TradeEvent.STOP_HIT): None,    # LOSS/STOP/BREAKEVEN from P&L
    (TradeState.FLOATING, TradeEvent.TIME_EXPIRED): TradeState.TIME_EXIT,
    (TradeState.FLOATING, TradeEvent.EOD_CLOSE): None,   # dynamic from P&L
    (TradeState.FLOATING, TradeEvent.REVERSAL): None,     # dynamic from P&L

    # FLOATING → FLOATING (non-terminal events)
    (TradeState.FLOATING, TradeEvent.TRAIL_TRIGGERED): TradeState.FLOATING,
    (TradeState.FLOATING, TradeEvent.PARTIAL_CLOSE): TradeState.FLOATING,
    (TradeState.FLOATING, TradeEvent.SCALEOUT): TradeState.FLOATING,
    (TradeState.FLOATING, TradeEvent.PNL_UPDATE): TradeState.FLOATING,
    (TradeState.FLOATING, TradeEvent.AUDIT): TradeState.FLOATING,
}


def _resolve_dynamic_state(pnl: float) -> TradeState:
    """Resolve dynamic (None) transitions based on P&L sign."""
    if pnl > 0.5:  # > 0.5 pts is a win
        return TradeState.WIN
    elif pnl < -0.5:
        return TradeState.LOSS
    else:
        return TradeState.BREAKEVEN


class TradeStateMachine:
    """Validates and executes state transitions for a single trade."""

    def __init__(self, trade_id: int, current_state: TradeState):
        self.trade_id = trade_id
        self.state = current_state

    def can_transition(self, event: TradeEvent) -> bool:
        """Check if a transition is valid without executing it."""
        return (self.state, event) in _TRANSITIONS

    def transition(self, event: TradeEvent, context: dict = None) -> TradeState:
        """
        Execute a state transition. Returns new state.
        Raises InvalidTransitionError if the transition is not in the table.

        context: optional dict with {"pnl": float} for dynamic state resolution.
        """
        key = (self.state, event)
        if key not in _TRANSITIONS:
            valid_events = [e.value for (s, e) in _TRANSITIONS if s == self.state]
            raise InvalidTransitionError(
                f"Trade {self.trade_id}: Cannot {event.value} from {self.state.value}. "
                f"Valid events: {valid_events}"
            )

        new_state = _TRANSITIONS[key]

        # Dynamic resolution for None entries
        if new_state is None:
            pnl = (context or {}).get("pnl", 0)
            new_state = _resolve_dynamic_state(pnl)

        old_state = self.state
        self.state = new_state

        if old_state != new_state:
            logger.debug(
                f"Trade {self.trade_id}: {old_state.value} --[{event.value}]--> {new_state.value}"
            )

        return new_state

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal


def create_machine(trade: dict) -> TradeStateMachine:
    """Factory: create a state machine from a trade DB row dict."""
    state = TradeState.from_str(trade.get("status", ts.OPEN))
    return TradeStateMachine(trade["id"], state)
