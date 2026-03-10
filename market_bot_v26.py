"""
MARKET BOT v26.0 — ES Futures Trading Assistant (Claude-Powered + IBKR Live)
==============================================================================
DATA SOURCES:
- IBKR TWS/Gateway (primary) — live prices, historical bars, SPX options
- yfinance (automatic fallback if IBKR disconnected)

PRIMARY SIGNALS:
- Fractal pattern recognition engine (historical day matching)
- Multi-timeframe momentum alignment (5m/15m/1H/Daily)
- TICK proxy from Mag7 breadth (institutional flow detection)
- Cross-asset correlation regime (ES/Bonds/DXY/VIX)

SESSION STRUCTURE:
- Overnight/gap analysis with fill probabilities
- Opening type classification (Dalton Market Profile)
- Day type detection (trend vs range probability)
- VPOC migration + naked POC tracking (price magnets)
- Relative volume by time of day (RVOL)
- Weekly context and calendar effects (OpEx, month-end)
- Economic calendar integration (CPI, FOMC, NFP, etc.)
- Liquidity sweep / stop hunt detection

LEVELS:
- Anchored VWAPs (weekly, monthly, swing-point)
- 0DTE SPX gamma walls with net gamma exposure
- Volume Profile POC / VAH / VAL
- VWAP with standard deviation bands
- Supply/demand zones

RISK MANAGEMENT:
- VIX term structure regime filter (adaptive thresholds)
- Adaptive confidence based on regime + performance
- Time-of-day win rate tracking
- Signal combination win rate database (self-learning)
- Dynamic FLAT threshold per regime

INFRASTRUCTURE:
- Claude API with two-pass analysis (primary + risk review)
- Hourly chart screenshot archive
- SQLite journal with trade audit
- Telegram alerts + command listener
"""

import time
import json
import re
import base64
from datetime import timedelta, time as dtime
from pathlib import Path
from typing import Tuple

import requests
import anthropic

from bot_config import CFG, logger, now_et, HTTP, generate_cycle_id

from fractal_engine import FractalEngine, format_fractal_telegram
from ibkr_client import IBKRClient
from advanced_features import (
    calc_gex_regime, FlowScanner, calc_vol_regime_shift,
    calc_divergence_score,
)
import trade_status as ts
from confidence_engine import (
    AccuracyTracker,
    calc_regime_adjustment,
    apply_confidence_pipeline,
)

# --- Extracted modules ---
from journal import Journal
from market_data import MarketData
from indicators import (
    calc_vwap, calc_volume_profile, calc_cumulative_delta,
    calc_prior_day_levels, calc_initial_balance, calc_synthetic_breadth,
    calc_gamma_levels, calc_vsa_and_structure, calc_gap_analysis,
    calc_rvol, calc_vix_term_structure,
    calc_mtf_momentum, calc_vpoc_migration, calc_tick_proxy,
    calc_anchored_vwaps, calc_liquidity_sweeps,
    classify_opening_type, classify_day_type,
    calc_cross_asset_correlation, calc_iv_skew,
    calc_delta_at_levels,
)
from session_utils import get_session_phase, is_news_approaching, position_suggestion
from trade_audit import audit_open_trades
from charts import capture_triple_screen, ChartLibrary
from claude_analysis import CycleMemory, AlertTier, build_analysis_prompt, SYSTEM_PROMPT
from telegram_bot import (
    send_telegram, send_telegram_photo, send_daily_recap, send_heartbeat,
    format_analysis_message, format_action_card,
    _escape_telegram_md, _detail_msg_lock, _latest_detail_msg,
    TelegramCommandListener,
)
from price_monitor import PriceMonitor, _collect_monitor_levels
from health_metrics import HealthMetrics
from tape_reader import TapeReader


# =================================================================
# --- POST-TRADE REVIEW ---
# =================================================================

def _run_post_trade_review(journal, md, claude_client):
    """Run Claude post-trade review of today's closed trades."""
    from claude_analysis import build_review_prompt
    try:
        closed = journal.get_closed_trades_today()
        if not closed:
            logger.info("Post-trade review: no closed trades today.")
            return
        today_str = now_et().strftime("%Y-%m-%d")
        existing = journal.get_trade_review(today_str)
        if existing:
            logger.info("Post-trade review: already done for today.")
            return
        prompt = build_review_prompt(closed, md.current_price)
        resp = claude_client.messages.create(
            model=CFG.CLAUDE_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        resp_text = resp.content[0].text if resp.content else ""
        match = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if match:
            review_data = json.loads(match.group(0))
            journal.save_trade_review(
                today_str,
                review_data.get("reviews", []),
                review_data.get("summary", ""),
            )
            logger.info(f"Post-trade review saved: {len(review_data.get('reviews', []))} lessons.")
    except Exception as e:
        logger.warning(f"Post-trade review failed: {e}")


# =================================================================
# --- SIGNAL SCORE UPDATER ---
# =================================================================

def _update_signal_scores(journal, current_price):
    """Fill in 30/60/120m prices for pending signal_scores rows."""
    try:
        from datetime import datetime
        pending = journal.get_pending_signal_scores()
        _now = now_et()
        for s in pending:
            try:
                sig_time = datetime.strptime(s["signal_time"], "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            elapsed_min = (_now.replace(tzinfo=None) - sig_time).total_seconds() / 60
            updates = {}
            if s["price_30m"] is None and elapsed_min >= 30:
                updates["price_30m"] = current_price
            if s["price_60m"] is None and elapsed_min >= 60:
                updates["price_60m"] = current_price
            if s["price_120m"] is None and elapsed_min >= 120:
                updates["price_120m"] = current_price

            # Max favorable/adverse excursion
            p0 = s["price_at_signal"]
            is_long = "BULL" in s["verdict"].upper()
            move = current_price - p0 if is_long else p0 - current_price
            if move > 0:
                if s["max_favorable"] is None or move > s["max_favorable"]:
                    updates["max_favorable"] = round(move, 2)
            elif move < 0:
                adverse = abs(move)
                if s["max_adverse"] is None or adverse > s["max_adverse"]:
                    updates["max_adverse"] = round(adverse, 2)

            if updates:
                journal.update_signal_score(s["id"], **updates)
    except Exception as e:
        logger.warning(f"Signal score update failed: {e}")


# =================================================================
# --- SLEEP / SCHEDULE ---
# =================================================================

def get_sleep_interval(atr_14=None, last_atr=None) -> Tuple[int, str]:
    now = now_et()
    weekday = now.weekday()
    current_time = now.time()

    # Weekend shutdown: Fri 5pm -> Sun 6:01pm
    if weekday == 4 and current_time >= dtime(17, 0):
        target = now.replace(hour=18, minute=1, second=0) + timedelta(days=2)
        return int((target - now).total_seconds()), "Weekend (-> Sun 6:01pm)"
    if weekday == 5:
        target = now.replace(hour=18, minute=1, second=0) + timedelta(days=1)
        return int((target - now).total_seconds()), "Weekend (-> Sun 6:01pm)"
    if weekday == 6 and current_time < dtime(18, 1):
        target = now.replace(hour=18, minute=1, second=0)
        return int((target - now).total_seconds()), "Weekend (-> Sun 6:01pm)"

    # Daily maintenance break: 5pm-6pm ET (ES futures closed)
    if dtime(17, 0) <= current_time < dtime(18, 0):
        target = now.replace(hour=18, minute=1, second=0)
        return int((target - now).total_seconds()), "CME maintenance (-> 6:01pm)"

    # Time-of-day buckets (minutes since midnight ET)
    hour, minute = now.hour, now.minute
    t = hour * 60 + minute

    if 570 <= t < 600:       # 9:30-10:00 -- opening drive
        base = 300           # 5 min
    elif 600 <= t < 660:     # 10:00-11:00 -- morning momentum
        base = 420           # 7 min
    elif 660 <= t < 780:     # 11:00-13:00 -- lunch lull
        base = 600           # 10 min
    elif 780 <= t < 930:     # 13:00-15:30 -- afternoon
        base = 600           # 10 min
    elif 930 <= t < 960:     # 15:30-16:00 -- closing drive
        base = 300           # 5 min
    elif 240 <= t < 570:     # 4:00-9:30 -- pre-open
        base = 300           # 5 min
    else:                    # overnight/after-hours
        base = 1800          # 30 min

    # ATR spike override: if ATR jumped 30%+ since last cycle, cut interval in half
    if atr_14 and last_atr and last_atr > 0:
        if atr_14 / last_atr >= 1.3:
            base = max(base // 2, 180)  # floor 3 min

    if t < 570 or t >= 960:
        mode = "OVERNIGHT"
    elif t < 600:
        mode = "OPENING"
    elif t >= 930:
        mode = "CLOSING"
    elif 660 <= t < 780:
        mode = "LUNCH"
    else:
        mode = "ACTIVE"

    return base, f"{mode} ({base // 60}m)"


# =================================================================
# --- SHUTDOWN HELPER ---
# =================================================================

def _shutdown(tape_reader, price_monitor, cmd_listener, ibkr):
    """Cleanly stop all background workers and disconnect IBKR."""
    if tape_reader:
        tape_reader.stop()
    if price_monitor:
        price_monitor.stop()
    if cmd_listener:
        cmd_listener.stop()
    if ibkr and ibkr.ib:
        try:
            ibkr.ib.disconnect()
            logger.info("IBKR disconnected cleanly.")
        except Exception as e:
            logger.debug(f"IBKR disconnect failed: {e}")


# =================================================================
# --- MAIN LOOP ---
# =================================================================

def main():
    # Import _latest_detail_msg at module scope is read-only;
    # we need to write to the module-level variable in telegram_bot
    import telegram_bot as _tg_mod

    logger.info("=" * 60)
    logger.info("MARKET BOT v26.0 Starting...")
    logger.info("=" * 60)

    # Initialize components
    journal = Journal()
    fractal_engine = FractalEngine()
    chart_lib = ChartLibrary()
    accuracy_tracker = AccuracyTracker(journal)
    flow_scanner = FlowScanner()
    from signal_logger import SignalLogger
    signal_logger = SignalLogger()
    from shadow_mode import ShadowMode
    shadow_mode = ShadowMode(shadow_params=CFG.SHADOW_PARAMS) if CFG.SHADOW_ENABLED else None
    cycle_memory = CycleMemory(max_cycles=5)
    alert_tier = AlertTier()
    health = HealthMetrics()

    # Initialize IBKR connection
    ibkr = IBKRClient(host=CFG.IBKR_HOST, port=CFG.IBKR_PORT, client_id=CFG.IBKR_CLIENT_ID)
    logger.info(f"IBKR: {ibkr.get_status()}")

    # Fractal v2.0: backfill historical data from IBKR
    if ibkr.connected:
        try:
            fractal_engine.backfill(ibkr, target_days=1000)
        except Exception as e:
            logger.warning(f"Fractal backfill failed (non-fatal): {e}")

    # Initialize tape reader (order flow from time & sales)
    tape_reader = TapeReader(ibkr)
    if ibkr.connected:
        tape_reader.start()

    # Test Telegram -- skip if credentials aren't set
    telegram_ok = False
    telegram_enabled = (
        CFG.TELEGRAM_TOKEN
        and CFG.TELEGRAM_CHAT_ID
        and "your_" not in CFG.TELEGRAM_TOKEN.lower()
        and "paste" not in CFG.TELEGRAM_TOKEN.lower()
        and "your_" not in str(CFG.TELEGRAM_CHAT_ID).lower()
        and len(CFG.TELEGRAM_TOKEN) > 20
    )

    logger.info(f"Telegram: {'CONFIGURED' if telegram_enabled else 'NOT CONFIGURED (running console-only)'}")
    logger.info(f"Anthropic API Key: {'SET' if CFG.ANTHROPIC_API_KEY and CFG.ANTHROPIC_API_KEY != '' else 'MISSING'}")

    if telegram_enabled:
        logger.info(f"Telegram Token: ***configured ({len(CFG.TELEGRAM_TOKEN)} chars)***")
        logger.info(f"Telegram Chat ID: '{CFG.TELEGRAM_CHAT_ID}'")

        # --- Network Diagnostic ---
        logger.info("--- Network Diagnostic ---")
        try:
            r = HTTP.get("https://www.google.com", timeout=10)
            logger.info(f"Google.com: OK ({r.status_code})")
        except Exception as e:
            logger.error(f"Google.com: FAILED ({e})")

        try:
            r = HTTP.get("https://api.telegram.org", timeout=10)
            logger.info(f"api.telegram.org: OK ({r.status_code})")
        except Exception as e:
            logger.error(f"api.telegram.org: FAILED ({e})")

        # --- Try sending with retries ---
        logger.info("Attempting Telegram connection...")
        test_url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendMessage"
        for attempt in range(4):
            try:
                test_res = requests.post(test_url, data={
                    "chat_id": CFG.TELEGRAM_CHAT_ID,
                    "text": f"MARKET BOT v26.0 ONLINE (Claude-Powered)\nGhost P&L | GEX | Flow Scanner | Vol Shift | Divergence | Adaptive Weights\nIBKR: {ibkr.get_status()}"
                }, timeout=15)
                logger.info(f"Telegram response: {test_res.status_code} -- {test_res.text[:300]}")
                if test_res.status_code == 200:
                    telegram_ok = True
                    break
            except Exception as e:
                logger.warning(f"Telegram attempt {attempt + 1}/4 failed: {e}")
                time.sleep(2)

        if not telegram_ok:
            logger.warning("Telegram connection failed -- running in CONSOLE-ONLY mode.")
            logger.warning("Bot will still analyze and log, but no Telegram alerts.")
            logger.warning("Set up Telegram later: create bot via @BotFather, add to group, set .env")
        else:
            logger.info("Telegram connection SUCCESS!")
    else:
        logger.info("Telegram not configured. Running in CONSOLE-ONLY mode.")
        logger.info("To enable: message @BotFather on Telegram, create a bot, add it to your group,")
        logger.info("then set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in your .env file.")

    # Start command listener only if Telegram is working
    cmd_listener = None
    if telegram_ok:
        cmd_listener = TelegramCommandListener(journal, alert_tier=alert_tier, health=health)
        cmd_listener.start()

    # Start real-time price monitor (separate IBKR connection)
    price_monitor = None
    if ibkr.connected and telegram_ok:
        try:
            price_monitor = PriceMonitor(
                ibkr_host=CFG.IBKR_HOST,
                ibkr_port=CFG.IBKR_PORT,
                client_id=CFG.IBKR_CLIENT_ID + 1,  # Different clientId for separate connection
            )
            price_monitor.start()
        except Exception as e:
            logger.warning(f"PriceMonitor init failed (non-fatal): {e}")

    # Claude client
    claude_client = anthropic.Anthropic(api_key=CFG.ANTHROPIC_API_KEY)

    last_heartbeat = now_et()
    # If starting after 4:55 PM ET, assume today's recap already sent to avoid duplicate
    _startup_time = now_et()
    last_recap = _startup_time if _startup_time.time() >= dtime(16, 55) else None
    # If starting after 4:00 PM ET, assume today's EOD close already ran
    last_eod_close = _startup_time if _startup_time.time() >= dtime(16, 0) else None
    first_run = True  # Run one cycle immediately on startup, even weekends
    news_alerts_sent: set = set()  # Tracks sent news alerts to prevent per-cycle spam
    last_atr: float = 0.0  # Track ATR between cycles for dynamic interval
    md = None  # Initialized in loop; needed before first data fetch for recap guard

    while True:
        try:
            now = now_et()
            sleep_secs, sleep_mode = get_sleep_interval()

            # --- Weekend sleep (skip on first run) ---
            if "Weekend" in sleep_mode and not first_run:
                logger.info(f"{sleep_mode}. Sleeping {sleep_secs / 3600:.1f} hrs...")
                time.sleep(sleep_secs)
                continue

            # --- Daily recap at 17:00 ET ---
            if now.time() >= dtime(16, 55) and (last_recap is None or last_recap.date() < now.date()):
                if md is not None:
                    _run_post_trade_review(journal, md, claude_client)
                send_daily_recap(journal)
                journal.save_daily_stats()  # Persist daily stats for historical tracking
                health.cleanup(days=30)  # Prune old health metrics
                last_recap = now

            # --- Heartbeat (RTH weekdays only) ---
            _wd = now.weekday()  # 0=Mon … 4=Fri
            if (_wd < 5 and dtime(9, 0) <= now.time() <= dtime(16, 0)
                    and (now - last_heartbeat).total_seconds() >= CFG.HEARTBEAT_INTERVAL):
                send_heartbeat(journal, accuracy_tracker)
                last_heartbeat = now

            # --- Hourly Chart Library Capture ---
            if chart_lib.should_capture():
                try:
                    if chart_lib.capture_and_save():
                        logger.info(f"Chart library: {chart_lib.get_status()}")
                except Exception as e:
                    logger.warning(f"Chart library capture failed: {e}")

            # --- Upcoming news alert (no blackout -- trading continues) ---
            is_approaching, event_str, event_dt, event_impact = is_news_approaching()
            if is_approaching:
                minutes_until = int((event_dt - now_et()).total_seconds() / 60)
                # Only send the alert once per event (track with a set to avoid spam)
                alert_key = f"{event_str}_{event_dt.date()}"
                if alert_key not in news_alerts_sent:
                    logger.info(f"NEWS APPROACHING: {event_str} in ~{minutes_until} min. Trading continues.")
                    news_alert_msg = (
                        f"*NEWS EVENT APPROACHING* -- {event_str}\n"
                        f"~{minutes_until} min away\n"
                        f"Bot continues trading -- manage risk accordingly."
                    )
                    for _try in range(3):
                        try:
                            send_telegram(news_alert_msg)
                            break
                        except Exception:
                            time.sleep(2)
                    news_alerts_sent.add(alert_key)

            # ======================
            # CORE ANALYSIS CYCLE
            # ======================
            # Hot-reload config overrides at the start of each cycle
            from bot_config import reload_config
            reload_config()

            cycle_id = generate_cycle_id()
            cycle_start = time.time()
            ibkr_ms = 0.0
            fractal_ms = 0.0
            claude_ms = 0.0
            cycle_errors = 0
            cycle_error_details = []
            logger.info(f"[{cycle_id}] === Analysis cycle start ===")

            # 0. IBKR auto-reconnect -- attempt reconnect if disconnected
            if ibkr and not ibkr.connected:
                logger.warning("IBKR disconnected -- attempting reconnect...")
                if ibkr.connect():
                    logger.info("IBKR reconnected successfully.")
                    send_telegram("IBKR RECONNECTED -- live data restored.")
                else:
                    logger.warning("IBKR reconnect failed -- skipping cycle (no yfinance fallback).")
                    time.sleep(30)
                    continue

            # 1. Fetch all market data (ONCE)
            t0 = time.time()
            md = MarketData(ibkr=ibkr)
            ibkr_ms = (time.time() - t0) * 1000
            logger.info(f"[{cycle_id}] Data fetch: {ibkr_ms:.0f}ms ({md.data_source})")

            # 2. Audit open trades first -- ensures realized P&L is current before loss limit check
            realized, floating = audit_open_trades(journal, md)
            _update_signal_scores(journal, md.current_price)

            # 2b. EOD force-close: close ALL remaining open positions at 4:00 PM ET
            #     Uses shared close_trade_at_price() to avoid duplicating P&L logic (#14)
            if now.time() >= dtime(16, 0) and (last_eod_close is None or last_eod_close.date() < now.date()):
                from trade_audit import close_trade_at_price
                eod_trades = journal.get_open_trades()
                if eod_trades:
                    logger.info(f"EOD CLOSE: Force-closing {len(eod_trades)} open trade(s) at 4:00 PM ET")
                    for trade in eod_trades:
                        try:
                            entry = float(trade["price"])
                            verdict = str(trade["verdict"]).upper()
                            cts = int(trade.get("contracts", 1) or 1)
                            outcome, pnl, close_price = close_trade_at_price(
                                trade, md.current_price, journal, reason="EOD"
                            )
                            pnl_dollars = pnl * CFG.POINT_VALUE * cts
                            result_emoji = "\u2705" if pnl >= 0 else "\u274c"
                            logger.info(f"EOD CLOSE trade {trade['id']}: {outcome} {pnl:+.2f} pts (${pnl_dollars:+,.0f})")
                            send_telegram(
                                f"*\U0001f4ca EOD CLOSE*\n"
                                f"\n"
                                f"`Signal` {_escape_telegram_md(verdict)}\n"
                                f"`Entry ` {entry:.2f}\n"
                                f"`Close ` {close_price:.2f}\n"
                                f"`P&L   ` *{'+' if pnl_dollars >= 0 else '-'}${abs(pnl_dollars):,.0f}* ({pnl:+.2f} pts x{cts})\n"
                                f"{result_emoji} {outcome}"
                            )
                        except Exception as e:
                            logger.warning(f"EOD close error on trade {trade.get('id')}: {e}")
                    # Re-audit to update realized P&L after EOD closes
                    realized, floating = audit_open_trades(journal, md)
                last_eod_close = now

            # 3. Daily loss limit -- includes floating P&L for true risk picture (#1)
            today_stats = journal.get_today_stats()
            realized_dollars = today_stats['realized'] * CFG.POINT_VALUE
            floating_dollars = today_stats['floating'] * CFG.POINT_VALUE
            net_dollars = realized_dollars + floating_dollars
            if net_dollars <= CFG.MAX_DAILY_LOSS:
                logger.warning(f"DAILY LOSS LIMIT HIT (Net: {'+' if net_dollars >= 0 else '-'}${abs(net_dollars):,.0f} | Realized: ${realized_dollars:,.0f} + Float: ${floating_dollars:,.0f}). Pausing until tomorrow.")
                send_telegram(
                    f"*DAILY LOSS LIMIT HIT*\n"
                    f"Realized: {'+' if realized_dollars >= 0 else '-'}${abs(realized_dollars):,.0f}\n"
                    f"Floating: {'+' if floating_dollars >= 0 else '-'}${abs(floating_dollars):,.0f}\n"
                    f"Net: *{'+' if net_dollars >= 0 else '-'}${abs(net_dollars):,.0f}*\n"
                    f"Bot paused until next session."
                )
                tomorrow = (now + timedelta(days=1)).replace(hour=5, minute=30, second=0)
                time.sleep((tomorrow - now).total_seconds())
                continue

            # 3. Compute all indicators from the single data snapshot
            vwap_val, vwap_status, vwap_levels = calc_vwap(md)
            vpoc = calc_volume_profile(md)
            cum_delta_val, cum_delta_bias = calc_cumulative_delta(md)
            prior = calc_prior_day_levels(md)
            initial_balance = calc_initial_balance(md)
            breadth = calc_synthetic_breadth(md)
            _now_time = now_et().time()
            _rth_open = dtime(9, 30)
            _rth_close = dtime(16, 0)
            if _rth_open <= _now_time <= _rth_close:
                g_call, g_put = calc_gamma_levels(md)
                gamma_detail = getattr(md, '_gamma_detail', {})
            else:
                g_call, g_put = "N/A", "N/A"
                gamma_detail = {}

            structure = calc_vsa_and_structure(md)
            session = get_session_phase()
            prev_verdict = journal.get_last_verdict()

            # Record fractal outcome from the most recently closed trade.
            # Previously used (current_price - open_entry) which was wrong:
            # (a) recorded every 10-min cycle, not once at close
            # (b) a 15-min pullback before a 30-pt rally got recorded as BEARISH
            # Now: only record when a trade actually closes, use final P&L direction.
            try:
                with journal._conn() as _jconn:
                    _all_closed = _jconn.execute(
                        "SELECT id, verdict, pnl, price FROM trades "
                        f"WHERE status IN {ts.DECIDED_SQL} "
                        "AND fractal_recorded IS NULL ORDER BY id ASC"
                    ).fetchall()
                    for _closed in _all_closed:
                        _pnl = float(_closed["pnl"])
                        _verdict = str(_closed["verdict"]).upper()
                        _entry = float(_closed["price"])
                        # Convert P&L to market move direction:
                        # Bullish win (+5 pts) = market moved UP +5
                        # Bearish win (+5 pts) = market moved DOWN -5
                        if "BEAR" in _verdict:
                            _move = -_pnl  # invert: bearish profit means market fell
                        else:
                            _move = _pnl   # bullish profit means market rose
                        # Actual direction = which way did the market actually move?
                        if "BULL" in _verdict:
                            _actual = "BULLISH" if _pnl > 0 else "BEARISH"
                        elif "BEAR" in _verdict:
                            _actual = "BEARISH" if _pnl > 0 else "BULLISH"
                        else:
                            _actual = "FLAT"
                        fractal_engine.record_outcome(_actual, _move)
                        _jconn.execute(
                            "UPDATE trades SET fractal_recorded=1 WHERE id=?",
                            (_closed["id"],)
                        )
                    if _all_closed:
                        _jconn.commit()
            except Exception as e:
                logger.warning(f"Fractal recording error: {e}")

            # opening_type needed by fractal engine -- compute before analyze()
            opening_type = classify_opening_type(md, initial_balance)

            # --- FRACTAL ENGINE v2.2 (Primary Signal) ---
            # Default in case fractal engine fails -- prevents NameError downstream
            class _FallbackProj:
                direction = "NEUTRAL"
                confidence = 0
            fractal_proj = _FallbackProj()
            t0 = time.time()
            fractal_result = fractal_engine.analyze(
                md.es_5m, md.current_price,
                es_1m=md.es_1m, es_15m=md.es_15m, vix=md.vix,
                nq_15m=md.nq_15m,
                tnx=md.tnx if not md.tnx.empty else None,
                open_type=opening_type.get("type", None),
            )
            fractal_ms = (time.time() - t0) * 1000
            fractal_proj = fractal_result["projection"]
            logger.info(
                f"Fractal v2.2: {fractal_result['match_count']} matches from "
                f"{fractal_result['total_days_scanned']} days | "
                f"Projection: {fractal_proj.direction} ({fractal_proj.confidence}%) | "
                f"Cache: {fractal_result['cached_days']} days"
            )

            # --- NEW ES-SPECIFIC INDICATORS ---
            # opening_type already computed above (needed by fractal engine)
            day_type = classify_day_type(md, initial_balance, structure)
            gap = calc_gap_analysis(md)
            rvol = calc_rvol(md)
            vix_term = calc_vix_term_structure(md)

            # --- v25 ADVANCED INDICATORS ---
            mtf_momentum = calc_mtf_momentum(md)
            vpoc_migration = calc_vpoc_migration(md)
            tick_proxy = calc_tick_proxy(md)
            regime = calc_regime_adjustment(vix_term, rvol, day_type, mtf_momentum)

            # --- v25.1 NEW FEATURES ---
            anchored_vwaps = calc_anchored_vwaps(md)
            liq_sweeps = calc_liquidity_sweeps(md)

            # --- v26.1 REAL DATA UPGRADES ---
            cross_corr = calc_cross_asset_correlation(md)
            iv_skew = calc_iv_skew(md)
            # Time-of-day context
            tod_context = accuracy_tracker.get_time_of_day_context()

            # --- v25.2 ADVANCED FEATURES ---
            if _rth_open <= _now_time <= _rth_close:
                gex_regime = calc_gex_regime(md)
            else:
                from advanced_features import _gex_default
                gex_regime = _gex_default()
            flow_data = flow_scanner.scan(md)
            # Normalize: FlowScanner may return "bias" or "flow_bias" -- ensure both exist
            if "flow_bias" not in flow_data and "bias" in flow_data:
                flow_data["flow_bias"] = flow_data["bias"]
            elif "bias" not in flow_data and "flow_bias" in flow_data:
                flow_data["bias"] = flow_data["flow_bias"]
            vol_shift = calc_vol_regime_shift(md)

            logger.info(
                f"Session: Gap {gap.get('gap_size', 0):+.1f} ({gap.get('fill_status', 'N/A')}) | "
                f"Open: {opening_type.get('type', 'N/A')} | "
                f"RVOL: {rvol.get('rvol', 1.0):.1f}x | "
                f"VIX: {vix_term.get('structure', 'N/A')} | "
                f"MTF: {mtf_momentum.get('alignment', 'N/A')} | "
                f"TICK: {tick_proxy.get('extreme', 'N/A')} | "
                f"Regime: {regime.get('regime', 'N/A')} (FLAT@{regime.get('flat_threshold', 60)}%)"
            )
            if liq_sweeps.get("active_sweep") != "NONE":
                logger.info(f"SWEEP: {liq_sweeps['signal']}")

            vix_val = vix_term.get("vix", 0.0)

            # Package all metrics
            metrics = {
                "vwap_val": vwap_val,
                "vwap_status": vwap_status,
                "vwap_levels": vwap_levels,
                "vpoc": vpoc,
                "cum_delta_val": cum_delta_val,
                "cum_delta_bias": cum_delta_bias,
                "prior": prior,
                "ib": initial_balance,
                "breadth": breadth,
                "g_call": g_call,
                "g_put": g_put,
                "gamma_detail": gamma_detail,
                "fractal": fractal_result,
                "structure": structure,
                "session": session,
                "vix": vix_val,
                "opening_type": opening_type,
                "day_type": day_type,
                "gap": gap,
                "rvol": rvol,
                "vix_term": vix_term,
                "mtf_momentum": mtf_momentum,
                "vpoc_migration": vpoc_migration,
                "tick_proxy": tick_proxy,
                "regime": regime,
                "tod_context": tod_context,
                "anchored_vwaps": anchored_vwaps,
                "liq_sweeps": liq_sweeps,
                "data_source": md.data_source,
                "gex_regime": gex_regime,
                "flow_data": flow_data,
                "vol_shift": vol_shift,
                "tape_text": tape_reader.get_prompt_text() if tape_reader and tape_reader.is_active else "",
                "cross_corr": cross_corr,
                "iv_skew": iv_skew,
            }

            # Delta at key levels -- needs metrics dict (VWAP, prior, VPOC, gamma)
            delta_levels = calc_delta_at_levels(md, metrics)
            metrics["delta_levels"] = delta_levels
            if delta_levels.get("net_bias") != "NEUTRAL":
                logger.info(
                    f"DELTA@LEVELS: {delta_levels['net_bias']} ({delta_levels['bias_score']:+d}) | "
                    f"{delta_levels['strongest_signal']} [{delta_levels['source']}]"
                )

            # Divergence score needs the full metrics dict
            divergence = calc_divergence_score(md, metrics)
            metrics["divergence"] = divergence


            # 3b. Data quality gate -- skip Claude call if data is clearly bad
            if md.current_price <= 0 or md.es_1m.empty:
                logger.warning("Data quality gate: price=0 or empty 1m bars. Skipping analysis cycle.")
                time.sleep(60)
                continue

            # Freshness check -- don't analyze stale data if IBKR disconnected mid-fetch
            _data_age = (now_et() - md.fetch_time).total_seconds()
            if _data_age > 120:
                logger.warning(f"Data quality gate: market data is {_data_age:.0f}s old (>120s). Skipping cycle.")
                time.sleep(30)
                continue

            # 4. Capture charts
            img_bytes, img_obj = capture_triple_screen()
            if not img_bytes:
                if first_run:
                    # Check if we already sent a briefing today (prevents duplicates on restart)
                    _briefing_file = Path("logs/.last_briefing")
                    _today_str = now_et().strftime("%Y-%m-%d")
                    if _briefing_file.exists() and _briefing_file.read_text().strip() == _today_str:
                        logger.info("Startup briefing already sent today -- skipping.")
                        first_run = False
                        sleep_secs, sleep_mode = get_sleep_interval()
                        logger.info(f"Sleeping {sleep_secs / 3600:.1f} hrs ({sleep_mode})...")
                        time.sleep(sleep_secs)
                        continue
                    # No charts open on first run -- send a data-only summary
                    logger.info("No chart windows found. Sending data-only briefing...")
                    fractal_tg = format_fractal_telegram(fractal_result)
                    _is_weekend = now.weekday() >= 5 or (now.weekday() == 4 and now.time() >= dtime(17, 0))
                    _briefing_label = "WEEKEND BRIEFING" if _is_weekend else "STARTUP BRIEFING"
                    _next_update = "Sunday 6:01pm ET" if _is_weekend else f"~{sleep_secs // 60} min"
                    summary = (
                        f"*{_briefing_label} (No Charts)*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"*ES Price:* {md.current_price:.2f}\n"
                        f"*VWAP:* {vwap_val} ({vwap_status})\n"
                        f"*POC:* {vpoc['poc']} | VAH: {vpoc['vah']} | VAL: {vpoc['val']}\n"
                        f"*Prior Day:* H {prior['prev_high']} | L {prior['prev_low']} | C {prior['prev_close']}\n"
                        f"*Gamma (0DTE):* Call {g_call} | Put {g_put} | {gamma_detail.get('net_gamma', 'N/A')}\n"
                        f"*Breadth:* {breadth}\n"
                        f"*Delta:* {cum_delta_bias}\n"
                        f"*VIX:* {vix_val:.2f}\n"
                        f"\n{fractal_tg}\n"
                        f"\nNext update: {_next_update}"
                    )
                    send_telegram(summary)
                    logger.info(f"{_briefing_label} sent.")
                    _briefing_file.write_text(_today_str)
                    first_run = False
                    # Now sleep until Sunday
                    sleep_secs, sleep_mode = get_sleep_interval()
                    logger.info(f"{sleep_mode}. Sleeping {sleep_secs / 3600:.1f} hrs...")
                    time.sleep(sleep_secs)
                    continue
                else:
                    # Chart optional after first run (#12): proceed with data-only analysis
                    logger.warning("No chart capture — running Claude analysis without chart image.")

            # 5. Build prompt and call Claude
            # Use today_stats (line 347) -- audit_open_trades returns 0,0 when no trades are open
            realized = today_stats["realized"]
            floating = today_stats["floating"]
            pnl_str = f"Advisory P&L: {'+' if realized*CFG.POINT_VALUE >= 0 else '-'}${abs(realized*CFG.POINT_VALUE):,.0f} | Open Trades: {'+' if floating*CFG.POINT_VALUE >= 0 else '-'}${abs(floating*CFG.POINT_VALUE):,.0f}"
            accuracy_str = accuracy_tracker.get_accuracy_context()
            logger.info(
                f"[{now:%H:%M}] Price: {md.current_price:.2f} | Breadth: {breadth} | "
                f"VWAP: {vwap_status} | Delta: {cum_delta_bias} | {pnl_str}"
            )

            calibration_str = accuracy_tracker.get_calibration_context()
            prompt = build_analysis_prompt(md, metrics, prev_verdict, pnl_str, accuracy_str,
                                          cycle_memory_text=cycle_memory.get_prompt_text(),
                                          calibration_str=calibration_str)

            # --- PASS 1: Primary analysis ---
            # Build message content — include chart image only if available (#12)
            msg_content = []
            if img_bytes:
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                msg_content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
            msg_content.append({"type": "text", "text": prompt})

            t0 = time.time()
            resp_text = None
            for attempt in range(3):
                try:
                    resp = claude_client.messages.create(
                        model=CFG.CLAUDE_MODEL,
                        max_tokens=2000,
                        system=SYSTEM_PROMPT,
                        messages=[{
                            "role": "user",
                            "content": msg_content,
                        }],
                    )
                    resp_text = resp.content[0].text if resp.content else ""
                    if resp_text:
                        break
                except Exception as e:
                    logger.warning(f"Claude attempt {attempt + 1} failed: {e}")
                    time.sleep(5)

            claude_ms = (time.time() - t0) * 1000
            if not resp_text:
                logger.error(f"[{cycle_id}] Claude failed after 3 attempts.")
                cycle_errors += 1
                cycle_error_details.append("Claude API failed")
                # Record failed cycle to health metrics
                total_ms = (time.time() - cycle_start) * 1000
                health.record_cycle(
                    cycle_id=cycle_id, ibkr_ms=ibkr_ms, claude_ms=claude_ms,
                    fractal_ms=fractal_ms, total_ms=total_ms,
                    data_source=md.data_source, errors=cycle_errors,
                    error_details="; ".join(cycle_error_details),
                )
                time.sleep(sleep_secs)
                continue

            # 6. Parse response
            match = re.search(r"\{.*\}", resp_text, re.DOTALL)
            if not match:
                logger.warning(f"No JSON in Claude response: {resp_text[:200]}")
                time.sleep(sleep_secs)
                continue

            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                time.sleep(sleep_secs)
                continue

            # 6b. Sanitize "after-hours locked" dismissals from Claude setup
            _setup_raw = str(data.get("setup", "")).upper()
            if "AFTER" in _setup_raw and "LOCK" in _setup_raw:
                logger.warning("Claude returned 'after-hours locked' setup — overriding with reasoning")
                _reason = str(data.get("reasoning", ""))[:120].strip()
                data["setup"] = _reason if _reason else "Overnight Globex session analysis"

            # 7. Position sizing with confidence pipeline fix (v25.3)
            data, verdict, conf, contracts, pos_str, confluence, decomposition = \
                apply_confidence_pipeline(
                    data, metrics, accuracy_tracker, regime, md,
                    news_info=(is_approaching, event_str, event_dt, event_impact),
                )
            flat_threshold = regime.get("flat_threshold", 60)

            # Add confluence to metrics for prompt/telegram
            metrics["confluence"] = confluence

            # 8. Log advisory trade to journal for P&L tracking
            _skipped_rr = None  # Track R:R gate skips for action card
            _new_trade_id = None  # Track for signal logger linkage
            if verdict not in ("FLAT", "NEUTRAL") and conf >= flat_threshold:
                # Parse entry/target/stop independently so a text stop
                # doesn't kill the entire trade (override can fix it)
                try:
                    entry_price = float(data.get("current_price", md.current_price))
                except (ValueError, TypeError):
                    entry_price = md.current_price
                try:
                    target_price = float(data.get("target", 0))
                except (ValueError, TypeError):
                    target_price = 0
                try:
                    stop_price = float(data.get("invalidation", 0))
                except (ValueError, TypeError):
                    stop_price = 0
                if target_price == 0 or stop_price == 0:
                    logger.info(f"Non-numeric or missing target/stop from Claude: target={data.get('target')}, stop={data.get('invalidation')}")

                # CONFLUENCE OVERRIDE target fix: when the pipeline overrode
                # FLAT → directional, Claude's target/stop are meaningless
                # (set for a FLAT verdict). Always use ATR-based targets for
                # overrides to guarantee consistent ~2:1 R:R.
                # Also fires when target is missing, equals entry, or is garbage.
                _was_override = decomposition.get("confluence_override", False)
                _target_garbage = (
                    target_price == 0 or stop_price == 0
                    or abs(target_price - entry_price) < 2.0
                )
                if (_was_override or _target_garbage) and entry_price > 0:
                    _is_long = ts.is_long(verdict)
                    _old_tp, _old_sl = target_price, stop_price
                    _atr = metrics.get("structure", {}).get("atr", 0) or 12.0
                    # Always use ATR-based targets for overrides.
                    # Claude said FLAT → its targets are garbage.
                    # Wider stops reduce stop-outs; maintains ~2:1 R:R.
                    if _is_long:
                        target_price = entry_price + max(12, _atr * 1.5)
                        stop_price = entry_price - max(8, _atr * 0.75)
                    else:
                        target_price = entry_price - max(12, _atr * 1.5)
                        stop_price = entry_price + max(8, _atr * 0.75)
                    target_price = round(target_price * 4) / 4  # snap to ES tick
                    stop_price = round(stop_price * 4) / 4
                    # Write back to data so Telegram card shows correct values
                    data["target"] = target_price
                    data["invalidation"] = stop_price
                    _rr = abs(target_price - entry_price) / abs(entry_price - stop_price) if abs(entry_price - stop_price) > 0 else 0
                    logger.info(
                        f"[OVERRIDE TARGETS] ATR={_atr:.1f} | "
                        f"Old TP {_old_tp:.2f} SL {_old_sl:.2f} → "
                        f"New TP {target_price:.2f} SL {stop_price:.2f} | "
                        f"R:R {_rr:.1f}:1"
                    )

                # Minimum target distance sanity check (matches backtest.py logic)
                MIN_TARGET_PTS = 6.0
                if target_price > 0 and entry_price > 0:
                    if ts.is_long(verdict) and target_price < entry_price + MIN_TARGET_PTS:
                        logger.info(f"[SANITY] Bullish target too close: {target_price:.2f} -> {entry_price + MIN_TARGET_PTS:.2f} (min {MIN_TARGET_PTS} pts)")
                        target_price = entry_price + MIN_TARGET_PTS
                    elif not ts.is_long(verdict) and target_price > entry_price - MIN_TARGET_PTS:
                        logger.info(f"[SANITY] Bearish target too close: {target_price:.2f} -> {entry_price - MIN_TARGET_PTS:.2f} (min {MIN_TARGET_PTS} pts)")
                        target_price = entry_price - MIN_TARGET_PTS

                # Build signal snapshot for attribution (#6)
                _trade_signals = {
                    "fractal": fractal_proj.direction if fractal_proj else "N/A",
                    "fractal_conf": fractal_proj.confidence if fractal_proj else 0,
                    "mtf": metrics.get("mtf_momentum", {}).get("alignment", "N/A"),
                    "flow": metrics.get("flow_data", {}).get("flow_bias", "N/A"),
                    "gex": metrics.get("gex_regime", {}).get("regime", "N/A"),
                    "sweep": str(metrics.get("liq_sweeps", {}).get("active_sweep", "NONE")),
                    "tick": metrics.get("tick_proxy", {}).get("extreme", "N/A"),
                    "divergence": metrics.get("divergence", {}).get("severity", "NONE"),
                    "rvol": metrics.get("rvol", {}).get("rvol", 1.0),
                    "confluence": confluence.get("confluence_level", "NONE"),
                    "regime": regime.get("regime", "N/A"),
                }

                # Check open trades -- close opposite-direction trades before opening new one
                open_trades = journal.get_open_trades()
                is_long_signal = ts.is_long(verdict)
                already_same_dir = False
                for t in open_trades:
                    try:
                        t_is_long = ts.is_long(t.get("verdict", ""))
                        if t_is_long == is_long_signal:
                            already_same_dir = True
                            # Don't break -- continue loop to close any opposite trades too
                            continue
                        else:
                            # Opposite direction -- close existing trade at current price (reversal)
                            t_entry = float(t.get("price", 0))
                            t_stop = float(t.get("stop", 0))
                            t_cts = int(t.get("contracts", 1) or 1)
                            if t_entry <= 0:
                                logger.warning(f"Trade {t.get('id')}: Invalid entry price -- skipping reversal")
                                continue
                            close_price = md.current_price  # use actual market price, not Claude's suggested entry
                            if t_is_long:
                                pnl = md.current_price - t_entry  # long closed: market price - entry
                                # Cap loss at stop level -- don't penalize worse than stop
                                if pnl < 0 and t_stop > 0:
                                    stop_loss = t_stop - t_entry  # negative for long
                                    if pnl < stop_loss:
                                        pnl = stop_loss
                                        close_price = t_stop  # show stop price since P&L is capped there
                            else:
                                pnl = t_entry - md.current_price  # short closed: entry - market price
                                # Cap loss at stop level -- don't penalize worse than stop
                                if pnl < 0 and t_stop > 0:
                                    stop_loss = t_entry - t_stop  # negative for short
                                    if pnl < stop_loss:
                                        pnl = stop_loss
                                        close_price = t_stop  # show stop price since P&L is capped there
                            status = ts.WIN if pnl > 0 else (ts.LOSS if pnl < 0 else ts.BREAKEVEN)
                            journal.update_trade(t["id"], status, round(pnl, 2))
                            t_dir = "LONG" if t_is_long else "SHORT"
                            new_dir = "LONG" if is_long_signal else "SHORT"
                            pnl_dollars = pnl * CFG.POINT_VALUE * t_cts
                            send_telegram(
                                f"*\u21c4 REVERSAL*\n"
                                f"\n"
                                f"Closed *{t_dir}* \u2192 Opening *{new_dir}*\n"
                                f"`Closed` {t.get('verdict','')} {t_cts}x @ `{close_price:.2f}`\n"
                                f"`P&L   ` *{'+' if pnl_dollars >= 0 else '-'}${abs(pnl_dollars):,.0f}* ({pnl:+.2f} pts) ({status})"
                            )
                            logger.info(f"[REVERSAL] Closed trade #{t['id']} ({t_dir}) {t_cts}x at {close_price:.2f} | P&L {pnl:+.2f}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Trade {t.get('id')} reversal parse error: {e}")
                        continue

                # Refresh P&L and open trades after reversal closures
                _refreshed = journal.get_today_stats()
                realized = _refreshed["realized"]
                floating = _refreshed["floating"]
                pnl_str = f"Advisory P&L: {'+' if realized*CFG.POINT_VALUE >= 0 else '-'}${abs(realized*CFG.POINT_VALUE):,.0f} | Open Trades: {'+' if floating*CFG.POINT_VALUE >= 0 else '-'}${abs(floating*CFG.POINT_VALUE):,.0f}"
                open_trades = journal.get_open_trades()
                if target_price > 0 and stop_price > 0 and not already_same_dir and len(open_trades) < 2:
                    # R:R gate -- skip trades below 1.2:1 reward-to-risk
                    target_dist = abs(target_price - entry_price)
                    stop_dist = abs(entry_price - stop_price)
                    rr = round(target_dist / stop_dist, 2) if stop_dist > 0 else 0

                    if rr < 1.2:
                        _skipped_rr = rr
                        logger.info(
                            f"[SKIPPED] R:R {rr:.1f}:1 below 1.2 gate "
                            f"| {verdict} @ {entry_price:.2f} TP {target_price:.2f} SL {stop_price:.2f}"
                        )
                        # Log skipped trade for phantom P&L tracking (#8)
                        journal.add_skipped_trade(
                            price=entry_price,
                            verdict=verdict,
                            confidence=conf,
                            target=target_price,
                            stop=stop_price,
                            contracts=contracts,
                            reason=f"R:R {rr:.1f}:1 below 1.2 gate",
                            signals=_trade_signals,
                        )
                    else:
                        _new_trade_id = journal.add_trade(
                            price=entry_price,
                            verdict=verdict,
                            confidence=conf,
                            target=target_price,
                            stop=stop_price,
                            contracts=contracts,
                            reasoning=data.get("reasoning", "")[:500],
                            session=session,
                            signals=_trade_signals,
                        )
                        # Record signal for forward-looking scoring
                        if _new_trade_id:
                            journal.add_signal_score(
                                trade_id=_new_trade_id,
                                signal_time=now_et().strftime("%Y-%m-%d %H:%M"),
                                price_at_signal=entry_price,
                                verdict=verdict,
                            )
                        arrow = "\U0001f7e2" if ts.is_long(verdict) else "\U0001f534"
                        send_telegram(
                            f"*{arrow} ADVISORY TRADE*\n"
                            f"\n"
                            f"*{verdict}* ({conf}%) \u2502 {contracts} ct\n"
                            f"\n"
                            f"`Entry  {entry_price:.2f}`\n"
                            f"`Target {target_price:.2f}`\n"
                            f"`Stop   {stop_price:.2f}`\n"
                            f"`R:R    {rr:.1f}:1`"
                        )
                        logger.info(
                            f"[ADVISORY] {verdict} {contracts}x ES @ {entry_price:.2f} "
                            f"| TP {target_price:.2f} | SL {stop_price:.2f} | R:R {rr:.1f}:1"
                        )

            # 9. Send to Telegram (with tiered alert system)
            final_msg, status_line = format_analysis_message(data, metrics, prev_verdict, pnl_str, pos_str)
            with _tg_mod._detail_msg_lock:
                _tg_mod._latest_detail_msg = final_msg  # Store for /detail command
                _tg_mod._latest_detail_time = now_et().strftime("%H:%M")

            # Build state for alert tier comparison
            alert_state = {
                "verdict": verdict,
                "confidence": conf,
                "divergence_alert": metrics.get("divergence", {}).get("alert", False),
                "sweep": str(metrics.get("liq_sweeps", {}).get("active_sweep", "NONE")),
                "fractal_dir": fractal_proj.direction,
                "vol_alert": metrics.get("vol_shift", {}).get("alert", False),
            }

            # Compute sleep interval early so we can show it in action card
            curr_atr_for_card = structure.get("atr", 0)
            _card_sleep, _card_mode = get_sleep_interval(atr_14=curr_atr_for_card, last_atr=last_atr)
            action_card = format_action_card(data, metrics, pos_str, _card_mode,
                                               pnl_str=pnl_str, open_trades=journal.get_open_trades(),
                                               skipped_rr=_skipped_rr)

            if alert_tier.should_send_full(alert_state, first_run=first_run):
                if img_obj:
                    send_telegram_photo(img_obj, status_line)
                    time.sleep(1.0)
                send_telegram(action_card)  # Short action card instead of full wall of text
                logger.info(f"Analysis sent: {status_line}")
            else:
                # Quiet mode -- action card only
                send_telegram(action_card)
                logger.info(f"Quiet update: {action_card}")

            # 10. Record cycle for cross-cycle memory
            key_signals = {
                "gex_regime": metrics.get("gex_regime", {}).get("regime", "N/A"),
                "flow_bias": metrics.get("flow_data", {}).get("bias", "N/A"),
                "mtf_alignment": metrics.get("mtf_momentum", {}).get("alignment", "N/A"),
                "divergence": metrics.get("divergence", {}).get("severity", "NONE"),
                "sweep": str(metrics.get("liq_sweeps", {}).get("active_sweep", "NONE")),
            }
            cycle_memory.record(
                price=md.current_price,
                verdict=verdict,
                confidence=conf,
                fractal_dir=fractal_proj.direction,
                fractal_conf=fractal_proj.confidence,
                key_signals=key_signals,
                data=data,
            )

            # Record signal decomposition for this cycle
            try:
                signal_logger.log_cycle(
                    cycle_id=cycle_id,
                    price=md.current_price,
                    metrics=metrics,
                    raw_verdict=decomposition["raw_verdict"],
                    raw_confidence=decomposition["raw_confidence"],
                    accuracy_adj=decomposition["accuracy_adj"],
                    regime_adj=decomposition["regime_adj"],
                    confluence=confluence,
                    floor_applied=decomposition["floor_applied"],
                    forced_flat=decomposition["forced_flat"],
                    mtf_conflict=decomposition["mtf_conflict"],
                    final_verdict=verdict,
                    final_confidence=conf,
                    contracts=contracts,
                    trade_taken=verdict not in ("FLAT", "NEUTRAL") and conf >= flat_threshold,
                    trade_id=_new_trade_id,
                )
            except Exception as e:
                logger.warning(f"Signal logger error: {e}")

            # Shadow mode: compare live vs alternate params
            if shadow_mode and shadow_mode.enabled:
                try:
                    shadow_result = shadow_mode.evaluate(
                        cycle_id=cycle_id,
                        price=md.current_price,
                        data=data,
                        metrics=metrics,
                        accuracy_tracker=accuracy_tracker,
                        regime=regime,
                        md=md,
                        live_verdict=verdict,
                        live_confidence=conf,
                        live_contracts=contracts,
                        news_info=(is_approaching, event_str, event_dt, event_impact),
                    )
                    if shadow_result and not shadow_result["verdict_match"]:
                        logger.info(
                            f"[SHADOW] Disagreement: Live={verdict}/{conf} "
                            f"Shadow={shadow_result['shadow_verdict']}/{shadow_result['shadow_confidence']}"
                        )
                    # Update phantom P&L on open shadow positions
                    shadow_mode.update_phantom_pnl(md.current_price)
                except Exception as e:
                    logger.warning(f"Shadow mode error: {e}")

            # Record health metrics for this cycle
            total_ms = (time.time() - cycle_start) * 1000
            health.record_cycle(
                cycle_id=cycle_id, ibkr_ms=ibkr_ms, claude_ms=claude_ms,
                fractal_ms=fractal_ms, total_ms=total_ms,
                data_source=md.data_source, verdict=verdict,
                confidence=conf, errors=cycle_errors,
                error_details="; ".join(cycle_error_details) if cycle_error_details else None,
            )
            logger.info(
                f"[{cycle_id}] Cycle complete: {total_ms:.0f}ms total "
                f"(data:{ibkr_ms:.0f} fractal:{fractal_ms:.0f} claude:{claude_ms:.0f})"
            )

            first_run = False

            # Update price monitor levels
            if price_monitor:
                try:
                    monitor_levels = _collect_monitor_levels(metrics, journal)
                    price_monitor.update_levels(monitor_levels)
                except Exception as e:
                    logger.warning(f"PriceMonitor level update failed: {e}")

            # Track ATR for dynamic sleep intervals
            curr_atr = structure.get("atr", 0)
            sleep_secs, sleep_mode = get_sleep_interval(atr_14=curr_atr, last_atr=last_atr)
            if curr_atr > 0:
                last_atr = curr_atr

            # Auto-shutdown window: 4:56 PM – 5:09 PM ET on weekdays
            _now_shutdown = now_et()
            if (_now_shutdown.weekday() < 5
                    and dtime(16, 56) <= _now_shutdown.time() <= dtime(17, 9)):
                logger.info("Auto-shutdown: weekday 4:56-5:09 PM ET window reached.")
                send_telegram("*MARKET BOT v26.0 OFFLINE* (auto-shutdown)")
                _shutdown(tape_reader, price_monitor, cmd_listener, ibkr)
                break

            # If it's a weekend, go to sleep now after the one-time run
            if "Weekend" in sleep_mode:
                logger.info(f"First run complete. {sleep_mode}. Sleeping {sleep_secs / 3600:.1f} hrs...")
                time.sleep(sleep_secs)
                continue

            # 10. Sleep
            logger.info(f"Next cycle in {sleep_secs / 60:.0f} min ({sleep_mode})")
            time.sleep(sleep_secs)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            _shutdown(tape_reader, price_monitor, cmd_listener, ibkr)
            send_telegram("*MARKET BOT v26.0 OFFLINE* (Manual Stop)")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            try:
                err_msg = _escape_telegram_md(f"{type(e).__name__}: {str(e)[:200]}")
                send_telegram(f"*BOT ERROR*\n`{err_msg}`\nRetrying in 30s...")
            except Exception as e2:
                logger.debug(f"Telegram error alert failed: {e2}")
            time.sleep(30)


if __name__ == "__main__":
    import signal
    import sys

    # SIGBREAK handler: covers Ctrl+Break and some Windows service stop scenarios.
    # Converts to KeyboardInterrupt so the main loop's existing cleanup code runs.
    def _graceful_shutdown(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGBREAK, _graceful_shutdown)

    headless = "--headless" in sys.argv

    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"FATAL ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        if not headless:
            input("Press ENTER to close this window...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        if not headless:
            input("Press ENTER to close this window...")
