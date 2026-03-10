"""
Trade audit engine -- checks and updates all open trades against current market data.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Tuple

from bot_config import logger, CFG, now_et
import trade_status as ts
from trade_state import TradeStateMachine, TradeState, TradeEvent, InvalidTransitionError, create_machine

if TYPE_CHECKING:
    from journal import Journal
    from market_data import MarketData


# =================================================================
# --- SHARED CLOSE FUNCTION (used by audit, EOD close, reversals) ---
# =================================================================

def close_trade_at_price(trade: dict, close_price: float, journal: 'Journal',
                         reason: str = "CLOSE") -> Tuple[str, float, float]:
    """
    Close a trade at a given price with stop-capped P&L.

    Returns (outcome, pnl_pts, close_price_used).
    The close_price_used may differ from close_price if P&L was capped at stop.
    """
    entry = float(trade["price"])
    stop = float(trade["stop"])
    cts = int(trade.get("contracts", 1) or 1)
    long = ts.is_long(trade["verdict"])

    pnl = (close_price - entry) if long else (entry - close_price)

    # Cap loss at stop level
    stop_pnl = (stop - entry) if long else (entry - stop)
    outcome = ts.TIME_EXIT
    price_used = close_price

    if pnl < 0 and stop_pnl < 0 and pnl < stop_pnl:
        # Worse than stop -- cap at stop loss
        pnl = stop_pnl
        outcome = ts.STOP
        price_used = stop
    elif pnl < 0 and stop_pnl >= 0:
        # Stop was at breakeven or better, but price reversed past it.
        # Cap at the stop level (would have been stopped out there).
        pnl = stop_pnl
        if pnl == 0.0:
            outcome = ts.BREAKEVEN
            price_used = entry
        elif pnl > 0:
            outcome = ts.WIN
            price_used = stop
        else:
            outcome = ts.TIME_EXIT
    elif pnl > 0:
        outcome = ts.WIN
    elif pnl == 0:
        outcome = ts.BREAKEVEN

    journal.update_trade(trade["id"], outcome, pnl)
    return outcome, pnl, price_used


# =================================================================
# --- PROGRESSIVE TRAILING STOP ---
# =================================================================

def _calc_trailing_stop(entry: float, current_stop: float,
                        float_pnl: float, long: bool,
                        atr: float = 0.0) -> Tuple[float, bool]:
    """
    Calculate the new stop using progressive trailing levels from config.
    If atr > 0, scale levels by ATR (40% of ATR per level).

    Returns (new_stop, changed).
    For longs: stop goes UP as profit grows.
    For shorts: stop goes DOWN as profit grows.
    """
    if atr > 0:
        # ATR-scaled dynamic levels
        base = atr * 0.4
        levels = [
            (base * 1.0, 0.0),           # 1x base -> breakeven
            (base * 1.5, base * 0.5),     # 1.5x base -> lock 0.5x base
            (base * 2.0, base * 1.0),     # 2x base -> lock 1x base
            (base * 2.5, base * 1.5),     # 2.5x base -> lock 1.5x base
        ]
    else:
        levels = CFG.TRAILING_STOP_LEVELS

    best_offset = None
    for threshold, offset in levels:
        if float_pnl >= threshold:
            best_offset = offset
        else:
            break  # levels are sorted ascending

    if best_offset is None:
        return current_stop, False

    if long:
        new_stop = entry + best_offset
        if new_stop > current_stop:
            return new_stop, True
    else:
        new_stop = entry - best_offset
        if new_stop < current_stop:
            return new_stop, True

    return current_stop, False


# =================================================================
# --- PARTIAL PROFIT TAKING ---
# =================================================================

def _handle_partial_close(trade: dict, close_price: float, pnl_pts: float,
                          journal: 'Journal', send_telegram) -> int:
    """
    If a multi-contract trade hits target, close half at target and trail the rest.

    Returns the number of contracts closed (0 if no partial close).
    """
    cts = int(trade.get("contracts", 1) or 1)
    if cts < 2:
        return 0  # single contract — full close, handled by caller

    close_cts = cts // 2  # close half (rounded down)
    remain_cts = cts - close_cts
    entry = float(trade["price"])
    long = ts.is_long(trade["verdict"])

    # Record partial close: update trade with remaining contracts, stop to entry
    new_stop = entry  # move stop to breakeven on remaining contracts
    journal.update_trade(
        trade["id"], ts.FLOATING, pnl_pts,
        stop=new_stop, contracts=remain_cts,
        stop_updated_at=now_et().strftime("%Y-%m-%d %H:%M:%S"),
    )

    pnl_dollars = pnl_pts * CFG.POINT_VALUE * close_cts
    send_telegram(
        f"*\u2705 PARTIAL CLOSE — {close_cts} of {cts} ct*\n"
        f"\n"
        f"`Entry ` {entry:.2f}\n"
        f"`Close ` {close_price:.2f}\n"
        f"`P&L   ` *+${pnl_dollars:,.0f}* (+{pnl_pts:.2f} pts x{close_cts})\n"
        f"Remaining: {remain_cts} ct, stop \u2192 {new_stop:.2f}"
    )
    logger.info(
        f"Trade {trade['id']} PARTIAL CLOSE: {close_cts}/{cts} ct at {close_price:.2f} "
        f"| +{pnl_pts:.2f} pts | Remaining {remain_cts} ct, stop->{new_stop:.2f}"
    )
    return close_cts


# =================================================================
# --- INTERMEDIATE SCALE-OUT AT 50% TARGET ---
# =================================================================

def _handle_intermediate_scaleout(trade: dict, curr_price: float,
                                  journal: 'Journal', send_telegram) -> bool:
    """
    Intermediate scale-out: when P&L reaches 50% of target distance,
    close half and move stop to breakeven.
    Returns True if scale-out executed.
    """
    cts = int(trade.get("contracts", 1) or 1)
    if cts < 2:
        return False

    if int(trade.get("partial_closed", 0)):
        return False

    entry = float(trade["price"])
    target = float(trade["target"])
    long = ts.is_long(trade["verdict"])

    target_dist = abs(target - entry)
    if target_dist < 1.0:
        return False

    half_target = target_dist * 0.5
    float_pnl = (curr_price - entry) if long else (entry - curr_price)

    if float_pnl < half_target:
        return False

    close_cts = cts // 2
    remain_cts = cts - close_cts
    new_stop = entry  # breakeven

    journal.update_trade(
        trade["id"], ts.FLOATING, float_pnl,
        stop=new_stop, contracts=remain_cts,
        stop_updated_at=now_et().strftime("%Y-%m-%d %H:%M:%S"),
    )
    # Mark partial_closed flag
    with journal._conn() as conn:
        conn.execute(
            "UPDATE trades SET partial_closed=1 WHERE id=?",
            (trade["id"],),
        )
        conn.commit()

    pnl_dollars = float_pnl * CFG.POINT_VALUE * close_cts
    send_telegram(
        f"*\u2705 50% SCALE-OUT \u2014 {close_cts} of {cts} ct*\n"
        f"\n"
        f"`Entry ` {entry:.2f}\n"
        f"`Close ` {curr_price:.2f}\n"
        f"`P&L   ` *+${pnl_dollars:,.0f}* (+{float_pnl:.2f} pts x{close_cts})\n"
        f"Remaining: {remain_cts} ct, stop \u2192 {new_stop:.2f} (breakeven)"
    )
    logger.info(
        f"Trade {trade['id']} 50% SCALE-OUT: {close_cts}/{cts} ct at {curr_price:.2f} "
        f"| +{float_pnl:.2f} pts | Remaining {remain_cts} ct, stop->{new_stop:.2f}"
    )
    return True


# =================================================================
def _validate_transition(machine: TradeStateMachine, event: TradeEvent,
                         context: dict = None) -> bool:
    """
    Validate a state transition. Returns True if valid, False if not.
    Logs warning on invalid transitions but never raises — audit continues.
    """
    if machine.can_transition(event):
        try:
            machine.transition(event, context)
            return True
        except InvalidTransitionError as e:
            logger.warning(f"State machine: {e}")
            return False
    else:
        logger.warning(
            f"State machine: Trade {machine.trade_id} cannot {event.value} "
            f"from {machine.state.value} — transition not in table"
        )
        return False


# --- MAIN AUDIT FUNCTION ---
# =================================================================

def audit_open_trades(journal: 'Journal', md: 'MarketData') -> Tuple[float, float]:
    """Check and update all open trades against current market data."""
    # Lazy imports to avoid circular dependencies
    from telegram_bot import send_telegram, _escape_telegram_md

    open_trades = journal.get_open_trades()
    if not open_trades:
        return 0.0, 0.0

    hist = md.es_1m.copy()
    if hist.empty:
        return 0.0, 0.0

    # Convert IBKR bar timestamps to Eastern then strip tz
    if hist.index.tz is not None:
        hist.index = hist.index.tz_convert("America/New_York").tz_localize(None)

    curr_price = md.current_price

    for trade in open_trades:
        try:
            entry = float(trade["price"])
            target = float(trade["target"])
            stop = float(trade["stop"])
            verdict = str(trade["verdict"]).upper()
            cts = int(trade.get("contracts", 1) or 1)
            trade_time = datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M")
            long = ts.is_long(verdict)
            short = ts.is_short(verdict)

            # State machine: validates transitions (defensive — logs warning if invalid)
            machine = create_machine(trade)

            # Adaptive time-based exit
            trade_age_seconds = (now_et().replace(tzinfo=None) - trade_time).total_seconds()
            base_exit_time = CFG.TIME_EXIT_SECONDS  # default 7200 (2 hours)

            # Extend time for profitable trades in trend
            float_pnl_check = (curr_price - entry) if long else (entry - curr_price)
            if float_pnl_check > 5.0:
                # Profitable and trending — extend to 1.5x normal time
                effective_exit_time = int(base_exit_time * 1.5)
            elif float_pnl_check < -2.0 and trade_age_seconds > base_exit_time * 0.75:
                # Losing and aged — exit slightly early
                effective_exit_time = int(base_exit_time * 0.75)
            else:
                effective_exit_time = base_exit_time

            if trade_age_seconds > effective_exit_time:
                _validate_transition(machine, TradeEvent.TIME_EXPIRED,
                                     {"pnl": (curr_price - entry) if long else (entry - curr_price)})
                outcome, pnl, close_price = close_trade_at_price(
                    trade, curr_price, journal, reason="TIME_EXIT"
                )
                logger.info(f"Trade {trade['id']} {outcome}: PnL {pnl:+.2f}")
                if outcome == ts.STOP:
                    exit_label = "STOP (capped from time-exit)"
                else:
                    exit_label = "TIME EXIT (2hr)"
                pnl_dollars = pnl * CFG.POINT_VALUE * cts
                send_telegram(
                    f"*\u23f1 {exit_label}*\n"
                    f"\n"
                    f"`Entry ` {entry:.2f}\n"
                    f"`Close ` {close_price:.2f}\n"
                    f"`P&L   ` *{'+' if pnl_dollars >= 0 else '-'}${abs(pnl_dollars):,.0f}* ({pnl:+.2f} pts)"
                )
                continue

            # Check price path since entry
            path = hist[hist.index >= trade_time]
            if path.empty:
                float_pnl = (curr_price - entry) if long else (entry - curr_price)
                journal.update_trade(trade["id"], ts.FLOATING, float_pnl)
                logger.info(f"Trade {trade['id']}: No path data, float P&L from current price: {float_pnl:+.2f}")
                continue

            path_high = float(path["High"].max())
            path_low = float(path["Low"].min())

            if long:
                hit_target = path_high >= target
                # If stop was moved above original (trailing), only check bars AFTER
                # the stop was actually changed -- not retroactively from the trigger bar.
                if stop >= entry and CFG.TRAILING_STOP_TRIGGER > 0:
                    stop_changed_at = trade.get("stop_updated_at")
                    if stop_changed_at:
                        try:
                            stop_changed_time = datetime.strptime(stop_changed_at, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            stop_changed_time = datetime.strptime(stop_changed_at, "%Y-%m-%d %H:%M")
                        stop_path = path[path.index >= stop_changed_time]
                    else:
                        # Legacy trade without stop_updated_at -- use conservative recent window
                        recent_cutoff = now_et().replace(tzinfo=None) - timedelta(seconds=CFG.ACTIVE_INTERVAL + 120)
                        stop_path = path[path.index >= recent_cutoff]
                    if not stop_path.empty:
                        hit_stop = float(stop_path["Low"].min()) <= stop
                    else:
                        hit_stop = False
                else:
                    hit_stop = path_low <= stop
                if hit_target and hit_stop:
                    first_target_bar = path[path["High"] >= target].index[0]
                    if stop >= entry and CFG.TRAILING_STOP_TRIGGER > 0 and not stop_path.empty:
                        stop_bars = stop_path[stop_path["Low"] <= stop]
                        first_stop_bar = stop_bars.index[0] if not stop_bars.empty else first_target_bar
                    else:
                        first_stop_bar = path[path["Low"] <= stop].index[0]
                    if first_stop_bar < first_target_bar:
                        hit_target = False
                    elif first_target_bar < first_stop_bar:
                        hit_stop = False
                    else:
                        hit_target = False
                        logger.info(f"Trade {trade['id']}: Target and stop hit in same bar -- awarding LOSS (conservative)")
                if hit_target:
                    pnl_pts = target - entry
                    # Partial profit taking (#5): close half, trail rest
                    closed_cts = _handle_partial_close(trade, target, pnl_pts, journal, send_telegram)
                    if closed_cts > 0:
                        _validate_transition(machine, TradeEvent.PARTIAL_CLOSE)
                        continue  # remaining contracts still floating
                    # Full close (single contract or no partial)
                    _validate_transition(machine, TradeEvent.TARGET_HIT, {"pnl": pnl_pts})
                    journal.update_trade(trade["id"], ts.WIN, pnl_pts)
                    logger.info(f"Trade {trade['id']} WIN: +{pnl_pts:.2f} pts x{cts}")
                    pnl_d = pnl_pts * CFG.POINT_VALUE * cts
                    send_telegram(
                        f"*\u2705 TARGET HIT \u2014 WIN*\n"
                        f"\n"
                        f"`Entry ` {entry:.2f}\n"
                        f"`Close ` {target:.2f}\n"
                        f"`P&L   ` *+${pnl_d:,.0f}* (+{pnl_pts:.2f} pts x{cts})"
                    )
                    continue
                if hit_stop:
                    pnl_pts = stop - entry
                    stop_status = ts.LOSS if pnl_pts < 0 else (ts.WIN if pnl_pts > 0 else ts.BREAKEVEN)
                    _validate_transition(machine, TradeEvent.STOP_HIT, {"pnl": pnl_pts})
                    journal.update_trade(trade["id"], stop_status, pnl_pts)
                    logger.info(f"Trade {trade['id']} {stop_status}: {pnl_pts:.2f} pts x{cts}")
                    if pnl_pts == 0:
                        send_telegram(
                            f"*\u2796 STOP HIT \u2014 BREAKEVEN*\n"
                            f"\n"
                            f"`Entry ` {entry:.2f}\n"
                            f"`Close ` {stop:.2f}\n"
                            f"`P&L   ` *$0* (0.00 pts)"
                        )
                    else:
                        pnl_d = pnl_pts * CFG.POINT_VALUE * cts
                        send_telegram(
                            f"*\u274c STOP HIT \u2014 LOSS*\n"
                            f"\n"
                            f"`Entry ` {entry:.2f}\n"
                            f"`Close ` {stop:.2f}\n"
                            f"`P&L   ` *-${abs(pnl_d):,.0f}* ({pnl_pts:.2f} pts x{cts})"
                        )
                    continue

                # Intermediate scale-out at 50% of target
                if _handle_intermediate_scaleout(trade, curr_price, journal, send_telegram):
                    _validate_transition(machine, TradeEvent.SCALEOUT)
                    continue

                # Still floating — progressive trailing stop (#4)
                float_pnl = curr_price - entry
                new_stop, trail_changed = _calc_trailing_stop(entry, stop, float_pnl, long=True)
                if trail_changed:
                    # Determine what level we're at for the message
                    offset = new_stop - entry
                    if offset == 0:
                        label = "BREAKEVEN"
                        detail = "Risk eliminated \u2014 stop moved to entry"
                    else:
                        label = f"ENTRY +{offset:.0f}"
                        detail = f"Locking in {offset:.0f} pts profit"
                    logger.info(f"Trade {trade['id']}: Stop trailed to {new_stop:.2f} ({label})")
                    send_telegram(
                        f"*\U0001f6e1 STOP \u2192 {label}*\n"
                        f"\n"
                        f"`Entry ` {entry:.2f} \u2502 {_escape_telegram_md(verdict)}\n"
                        f"`Stop  ` {stop:.2f} \u2192 {new_stop:.2f}\n"
                        f"`Float ` {float_pnl:+.2f} pts\n"
                        f"{detail}"
                    )
                _stop_ts = now_et().strftime("%Y-%m-%d %H:%M:%S") if trail_changed else None
                if trail_changed:
                    _validate_transition(machine, TradeEvent.TRAIL_TRIGGERED)
                else:
                    _validate_transition(machine, TradeEvent.PNL_UPDATE)
                journal.update_trade(trade["id"], ts.FLOATING, float_pnl, new_stop,
                                     stop_updated_at=_stop_ts)
                logger.info(f"Trade {trade['id']} OPEN (long): {float_pnl:+.2f} pts | Price {curr_price:.2f} vs Entry {entry:.2f}")

            elif short:
                hit_target = path_low <= target
                # If stop was moved below original (trailing), only check bars AFTER
                # the stop was actually changed -- not retroactively from the trigger bar.
                if stop <= entry and CFG.TRAILING_STOP_TRIGGER > 0:
                    stop_changed_at = trade.get("stop_updated_at")
                    if stop_changed_at:
                        try:
                            stop_changed_time = datetime.strptime(stop_changed_at, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            stop_changed_time = datetime.strptime(stop_changed_at, "%Y-%m-%d %H:%M")
                        stop_path = path[path.index >= stop_changed_time]
                    else:
                        # Legacy trade without stop_updated_at -- use conservative recent window
                        recent_cutoff = now_et().replace(tzinfo=None) - timedelta(seconds=CFG.ACTIVE_INTERVAL + 120)
                        stop_path = path[path.index >= recent_cutoff]
                    if not stop_path.empty:
                        hit_stop = float(stop_path["High"].max()) >= stop
                    else:
                        hit_stop = False
                else:
                    hit_stop = path_high >= stop
                if hit_target and hit_stop:
                    first_target_bar = path[path["Low"] <= target].index[0]
                    if stop <= entry and CFG.TRAILING_STOP_TRIGGER > 0 and not stop_path.empty:
                        stop_bars = stop_path[stop_path["High"] >= stop]
                        first_stop_bar = stop_bars.index[0] if not stop_bars.empty else first_target_bar
                    else:
                        first_stop_bar = path[path["High"] >= stop].index[0]
                    if first_stop_bar < first_target_bar:
                        hit_target = False
                    elif first_target_bar < first_stop_bar:
                        hit_stop = False
                    else:
                        hit_target = False
                        logger.info(f"Trade {trade['id']}: Target and stop hit in same bar -- awarding LOSS (conservative)")
                if hit_target:
                    pnl_pts = entry - target
                    # Partial profit taking (#5)
                    closed_cts = _handle_partial_close(trade, target, pnl_pts, journal, send_telegram)
                    if closed_cts > 0:
                        _validate_transition(machine, TradeEvent.PARTIAL_CLOSE)
                        continue
                    # Full close (single contract or no partial)
                    _validate_transition(machine, TradeEvent.TARGET_HIT, {"pnl": pnl_pts})
                    journal.update_trade(trade["id"], ts.WIN, pnl_pts)
                    logger.info(f"Trade {trade['id']} WIN: +{pnl_pts:.2f} pts x{cts}")
                    pnl_d = pnl_pts * CFG.POINT_VALUE * cts
                    send_telegram(
                        f"*\u2705 TARGET HIT \u2014 WIN*\n"
                        f"\n"
                        f"`Entry ` {entry:.2f}\n"
                        f"`Close ` {target:.2f}\n"
                        f"`P&L   ` *+${pnl_d:,.0f}* (+{pnl_pts:.2f} pts x{cts})"
                    )
                    continue
                if hit_stop:
                    pnl_pts = entry - stop
                    stop_status = ts.LOSS if pnl_pts < 0 else (ts.WIN if pnl_pts > 0 else ts.BREAKEVEN)
                    _validate_transition(machine, TradeEvent.STOP_HIT, {"pnl": pnl_pts})
                    journal.update_trade(trade["id"], stop_status, pnl_pts)
                    logger.info(f"Trade {trade['id']} {stop_status}: {pnl_pts:.2f} pts x{cts}")
                    if pnl_pts == 0:
                        send_telegram(
                            f"*\u2796 STOP HIT \u2014 BREAKEVEN*\n"
                            f"\n"
                            f"`Entry ` {entry:.2f}\n"
                            f"`Close ` {stop:.2f}\n"
                            f"`P&L   ` *$0* (0.00 pts)"
                        )
                    else:
                        pnl_d = pnl_pts * CFG.POINT_VALUE * cts
                        send_telegram(
                            f"*\u274c STOP HIT \u2014 LOSS*\n"
                            f"\n"
                            f"`Entry ` {entry:.2f}\n"
                            f"`Close ` {stop:.2f}\n"
                            f"`P&L   ` *-${abs(pnl_d):,.0f}* ({pnl_pts:.2f} pts x{cts})"
                        )
                    continue

                # Intermediate scale-out at 50% of target
                if _handle_intermediate_scaleout(trade, curr_price, journal, send_telegram):
                    _validate_transition(machine, TradeEvent.SCALEOUT)
                    continue

                # Still floating — progressive trailing stop (#4)
                float_pnl = entry - curr_price
                new_stop, trail_changed = _calc_trailing_stop(entry, stop, float_pnl, long=False)
                if trail_changed:
                    offset = entry - new_stop
                    if offset == 0:
                        label = "BREAKEVEN"
                        detail = "Risk eliminated \u2014 stop moved to entry"
                    else:
                        label = f"ENTRY -{offset:.0f}"
                        detail = f"Locking in {offset:.0f} pts profit"
                    logger.info(f"Trade {trade['id']}: Stop trailed to {new_stop:.2f} ({label})")
                    send_telegram(
                        f"*\U0001f6e1 STOP \u2192 {label}*\n"
                        f"\n"
                        f"`Entry ` {entry:.2f} \u2502 {_escape_telegram_md(verdict)}\n"
                        f"`Stop  ` {stop:.2f} \u2192 {new_stop:.2f}\n"
                        f"`Float ` {float_pnl:+.2f} pts\n"
                        f"{detail}"
                    )
                _stop_ts = now_et().strftime("%Y-%m-%d %H:%M:%S") if trail_changed else None
                if trail_changed:
                    _validate_transition(machine, TradeEvent.TRAIL_TRIGGERED)
                else:
                    _validate_transition(machine, TradeEvent.PNL_UPDATE)
                journal.update_trade(trade["id"], ts.FLOATING, float_pnl, new_stop,
                                     stop_updated_at=_stop_ts)
                logger.info(f"Trade {trade['id']} OPEN (short): {float_pnl:+.2f} pts | Price {curr_price:.2f} vs Entry {entry:.2f}")

            else:
                logger.warning(f"Trade {trade['id']}: Unknown verdict '{verdict}' -- marking Floating")
                journal.update_trade(trade["id"], ts.FLOATING, 0.0)

        except Exception as e:
            logger.warning(f"Audit error on trade {trade.get('id')}: {e}")
            continue

    # Also update phantom P&L for skipped trades (#8)
    _audit_skipped_trades(journal, curr_price)

    stats = journal.get_today_stats()
    return stats["realized"], stats["floating"]


def _audit_skipped_trades(journal: 'Journal', curr_price: float):
    """Update phantom P&L on today's skipped trades (never closes them)."""
    try:
        skipped = journal.get_skipped_trades_today()
        for trade in skipped:
            entry = float(trade["price"])
            target = float(trade["target"])
            stop = float(trade["stop"])
            long = ts.is_long(trade["verdict"])

            float_pnl = (curr_price - entry) if long else (entry - curr_price)

            # Check if target or stop would have been hit
            if long:
                if curr_price >= target:
                    float_pnl = target - entry  # cap at target
                elif curr_price <= stop:
                    float_pnl = stop - entry    # cap at stop
            else:
                if curr_price <= target:
                    float_pnl = entry - target
                elif curr_price >= stop:
                    float_pnl = entry - stop

            # Update pnl field on skipped trade (status stays SKIPPED)
            with journal._conn() as conn:
                conn.execute(
                    "UPDATE trades SET pnl=? WHERE id=?",
                    (round(float_pnl, 2), trade["id"]),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"Skipped trade audit error: {e}")
