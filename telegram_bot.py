"""
Telegram messaging, formatting, and command listener.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

import io
import time
import threading
from datetime import time as dtime
from typing import TYPE_CHECKING

import requests
from PIL import Image

from bot_config import CFG, logger, HTTP, now_et
import trade_status as ts

if TYPE_CHECKING:
    from journal import Journal
    from claude_analysis import AlertTier


# =================================================================
# --- COLOR HELPERS ---
# =================================================================

_GREEN = "\U0001f7e2"  # 🟢
_RED = "\U0001f534"    # 🔴
_YELLOW = "\U0001f7e1" # 🟡
_WHITE = "\u26aa"      # ⚪


def _dir_emoji(text: str) -> str:
    """Return 🟢 for bullish/long signals, 🔴 for bearish/short, ⚪ otherwise."""
    t = str(text).upper()
    # Traps first (reversal signals — trap name is OPPOSITE of trade direction)
    if any(k in t for k in ("BULL TRAP", "BULL_TRAP", "BULLISH_TRAP")):
        return _RED   # bull trap = short signal
    if any(k in t for k in ("BEAR TRAP", "BEAR_TRAP", "BEARISH_TRAP")):
        return _GREEN  # bear trap = long signal
    # Green: bullish / long / call-leaning
    if any(k in t for k in ("BULL", "LONG", "BUY", "CALL", "RISK-ON")):
        return _GREEN
    # Red: bearish / short / put-leaning
    if any(k in t for k in ("BEAR", "SHORT", "SELL", "PUT", "RISK-OFF")):
        return _RED
    return _WHITE


# =================================================================
# --- ESCAPING AND SENDING ---
# =================================================================

def _escape_telegram_md(text: str) -> str:
    """Escape special Telegram Markdown v1 characters in dynamic content."""
    for ch in ("\\", "`", "*", "_", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram(text: str, parse_mode: str = "Markdown") -> bool:
    if not CFG.TELEGRAM_TOKEN or "your_" in CFG.TELEGRAM_TOKEN.lower() or len(CFG.TELEGRAM_TOKEN) < 20:
        return False

    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CFG.TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    for attempt in range(3):
        try:
            if attempt == 0:
                res = HTTP.post(url, data=payload, timeout=15)
            else:
                res = requests.post(url, data=payload, timeout=15)
            if res.status_code == 200:
                return True
            logger.warning(f"Telegram send HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.warning(f"Telegram send attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    if parse_mode:
        try:
            fallback_payload = {"chat_id": CFG.TELEGRAM_CHAT_ID, "text": text}
            res = requests.post(url, data=fallback_payload, timeout=15)
            if res.status_code == 200:
                logger.info("Telegram: sent without Markdown (parse_mode fallback)")
                return True
            logger.warning(f"Telegram fallback also failed: {res.status_code}")
        except Exception as e:
            logger.warning(f"Telegram fallback exception: {e}")
    return False


def send_telegram_photo(img: Image.Image, caption: str) -> bool:
    if not CFG.TELEGRAM_TOKEN or "your_" in CFG.TELEGRAM_TOKEN.lower() or len(CFG.TELEGRAM_TOKEN) < 20:
        return False

    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendPhoto"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    for attempt in range(3):
        try:
            buf.seek(0)
            if attempt == 0:
                res = HTTP.post(url,
                    data={"chat_id": CFG.TELEGRAM_CHAT_ID, "caption": caption[:1024]},
                    files={"photo": ("chart.jpg", buf, "image/jpeg")},
                    timeout=30)
            else:
                res = requests.post(url,
                    data={"chat_id": CFG.TELEGRAM_CHAT_ID, "caption": caption[:1024]},
                    files={"photo": ("chart.jpg", buf, "image/jpeg")},
                    timeout=30)
            if res.status_code == 200:
                return True
            logger.warning(f"Telegram photo HTTP {res.status_code}: {res.text[:200]}")
        except Exception as e:
            logger.warning(f"Telegram photo attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return False


# =================================================================
# --- MESSAGE FORMATTING ---
# =================================================================

def format_analysis_message(data: dict, metrics: dict, prev_verdict: str, pnl_str: str, pos_suggestion: str) -> str:
    verdict = data.get("verdict", "FLAT").upper()
    conf = int(data.get("confidence", 0))
    flat_threshold = metrics.get("regime", {}).get("flat_threshold", 60)

    if verdict in ("FLAT", "NEUTRAL"):
        emoji, status = _WHITE, "NO SIGNAL \u2014 MONITORING"
    elif "BULL" in verdict:
        if conf < flat_threshold + 8:
            emoji, status = _GREEN, f"LOW CONFIDENCE {verdict} ({conf}%)"
        elif conf < flat_threshold + 18:
            emoji, status = _GREEN, f"TENTATIVE {verdict} ({conf}%)"
        else:
            emoji, status = _GREEN, f"BULLISH CONVICTION ({conf}%)"
    elif "BEAR" in verdict:
        if conf < flat_threshold + 8:
            emoji, status = _RED, f"LOW CONFIDENCE {verdict} ({conf}%)"
        elif conf < flat_threshold + 18:
            emoji, status = _RED, f"TENTATIVE {verdict} ({conf}%)"
        else:
            emoji, status = _RED, f"BEARISH CONVICTION ({conf}%)"
    else:
        emoji, status = _WHITE, f"{verdict} ({conf}%)"

    fractal = metrics.get("fractal", {})
    proj = fractal.get("projection", None)
    fractal_line = ""
    if proj and fractal.get("match_count", 0) > 0:
        fp_emoji = _dir_emoji(proj.direction)
        fractal_line = (
            f"\n*FRACTAL SIGNAL*\n"
            f"  {fp_emoji} {proj.direction} ({proj.confidence}%) \u2014 {proj.match_count} matches\n"
            f"  {proj.bullish_pct:.0f}% rallied | {proj.bearish_pct:.0f}% sold off\n"
            f"  Expected: {proj.expected_close_vs_current:+.1f} pts | Range: +{proj.projected_high:.1f} / {proj.projected_low:.1f}\n"
            f"  Targets: Up {proj.upside_target:.2f}  Down {proj.downside_target:.2f}\n"
        )

        top = fractal.get("top_matches", [])[:3]
        if top:
            for m in top:
                fractal_line += f"  * {m['date']} ({m['composite_score']:.0f}%) -> {m['outcome']} {m['remaining_move']:+.1f}\n"

    highlights = []
    breadth = metrics.get("breadth", "")
    if "STRONG" in breadth:
        highlights.append(f"Mag7: {breadth}")
    vwap_s = metrics.get("vwap_status", "")
    if "OVER" in vwap_s or "Stretched" in vwap_s:
        highlights.append(f"VWAP: {vwap_s}")
    context_line = " . ".join(highlights) if highlights else "Normal"

    rev_alert = ""
    rev_reason = data.get("reversal_reason", "N/A")
    if rev_reason and rev_reason != "N/A" and verdict != prev_verdict:
        rev_alert = f"\n*REVERSAL:* _{rev_reason}_"

    gap = metrics.get("gap", {})
    opening = metrics.get("opening_type", {})
    rvol = metrics.get("rvol", {})
    vix_ts = metrics.get("vix_term", {})
    mtf = metrics.get("mtf_momentum", {})
    tick = metrics.get("tick_proxy", {})
    regime = metrics.get("regime", {})
    vpoc_mig = metrics.get("vpoc_migration", {})
    day_read = data.get("day_type_read", "")
    rr = data.get("risk_reward", "N/A")
    frac_agree = data.get("fractal_agreement", "N/A")
    mtf_align = data.get("mtf_alignment", mtf.get("alignment", "N/A"))

    session_line = ""
    if gap.get("fill_status") and gap["fill_status"] not in ("N/A", "NO GAP"):
        session_line += f"*Gap:* {gap['gap_size']:+.1f} pts -- {gap['fill_status']}\n"
    if opening.get("type") and opening["type"] not in ("N/A", "FORMING"):
        session_line += f"*Open:* {opening['type']} ({opening.get('bias', 'N/A')})\n"
    if rvol.get("rvol", 1.0) != 1.0 and rvol.get("status") != "N/A":
        session_line += f"*RVOL:* {rvol['rvol']:.1f}x -- {rvol['status']}\n"
    if vix_ts.get("structure") and vix_ts["structure"] not in ("N/A", "FLAT"):
        session_line += f"*VIX:* {vix_ts['structure']} -- {vix_ts['signal']}\n"

    mtf_line = ""
    tf = mtf.get("timeframes", {})
    if mtf.get("alignment") and mtf["alignment"] != "N/A":
        # Show 3 independent timeframes; include ADX if details available
        details = mtf.get("details", {})
        if details:
            parts = []
            for k, label in [("5m", "5m"), ("1h", "1H"), ("daily", "D")]:
                d = details.get(k, {})
                direction = d.get("direction", tf.get(k, "?"))
                adx = d.get("adx", 0)
                if adx > 0:
                    parts.append(f"{label}:{direction}({adx:.0f})")
                else:
                    parts.append(f"{label}:{direction}")
            tf_str = " ".join(parts)
        else:
            tf_str = f"5m:{tf.get('5m','?')} 1H:{tf.get('1h','?')} D:{tf.get('daily','?')}"
        mtf_line = f"{_dir_emoji(mtf['alignment'])} *MTF:* {mtf['alignment']} ({tf_str})\n"

    tick_line = ""
    if tick.get("extreme") and tick["extreme"] not in ("N/A", "NEUTRAL"):
        tick_line = f"{_dir_emoji(tick['extreme'])} *TICK:* {tick['extreme']} -- {tick.get('signal', '')}\n"

    vpoc_line = ""
    if vpoc_mig.get("migration") and vpoc_mig["migration"] != "N/A":
        naked = vpoc_mig.get("naked_pocs", [])
        vpoc_line = f"*POC Migration:* {vpoc_mig['migration']}"
        if naked:
            closest = min(naked, key=lambda x: abs(x["distance"]))
            vpoc_line += f" | Naked: {closest['poc']} ({closest['distance']:+.1f})"
        vpoc_line += "\n"

    regime_line = ""
    if regime.get("regime") and regime["regime"] != "N/A":
        regime_line = f"*Regime:* {regime['regime']} (FLAT@{regime.get('flat_threshold', 60)}%)\n"

    avwaps = metrics.get("anchored_vwaps", {})
    sweeps_data = metrics.get("liq_sweeps", {})

    avwap_line = ""
    if avwaps.get("convergence_level", 0) > 0:
        avwap_line = f"*AVWAP:* {avwaps['convergence']}\n"

    sweep_line = ""
    sweep_detected = data.get("sweep_detected", sweeps_data.get("active_sweep", "NONE"))
    if sweep_detected != "NONE":
        sweep_line = f"{_dir_emoji(sweep_detected)} *SWEEP:* {sweeps_data.get('signal', sweep_detected)}\n"

    gex_data = metrics.get("gex_regime", {})
    gex_line = ""
    if gex_data.get("regime") and gex_data["regime"] not in ("N/A", "UNKNOWN"):
        gex_line = (
            f"*GEX:* {gex_data['regime']} | "
            f"Flip: {gex_data.get('flip_level', 0):.0f} ({gex_data.get('distance_to_flip', 0):+.0f}) | "
            f"{gex_data.get('playbook', 'MIXED')}\n"
        )

    flow = metrics.get("flow_data", {})
    flow_line = ""
    _flow_bias = (flow.get("flow_bias") or flow.get("bias") or "")
    if _flow_bias and _flow_bias not in ("N/A", "BALANCED", "BALANCED (low volume)"):
        _conf = flow.get("flow_confidence", "")
        _conf_tag = f" [{_conf}]" if _conf and _conf != "N/A" else ""
        _wpc = flow.get("weighted_pc_ratio", flow.get("pc_ratio", 0))
        flow_line = f"{_dir_emoji(_flow_bias)} *Flow:* {_flow_bias}{_conf_tag} (wP/C: {_wpc:.2f})"
        top_alert = flow.get("alerts", [])
        if top_alert:
            a = top_alert[0]
            _money = a.get("moneyness", "")
            _money_str = f" {_money}" if _money else ""
            flow_line += f" | Top: {a['type']}{_money_str} {a['strike']:.0f} ${a['notional']:.1f}M"
        flow_line += "\n"
    elif _flow_bias == "BALANCED (low volume)":
        flow_line = f"\U0001f4a4 *Flow:* Low volume -- bias suppressed\n"

    vol_shift_data = metrics.get("vol_shift", {})
    vol_shift_line = ""
    if vol_shift_data.get("alert"):
        vol_shift_line = f"*Vol Shift:* {vol_shift_data['shift']} ({vol_shift_data.get('vol_ratio', 1):.1f}x) | +/-{vol_shift_data.get('expected_hourly_range', 0):.0f}/hr\n"

    div_data = metrics.get("divergence", {})
    div_line = ""
    if div_data.get("alert"):
        div_line = (
            f"*{div_data['severity']}*: "
            f"{div_data.get('score', 0)}/{div_data.get('max_score', 0)} vs price {div_data.get('price_direction', '')} -- "
            f"{', '.join(div_data.get('divergent', []))}\n"
        )

    gex_read_line = ""
    gex_read = data.get("gex_read", "")
    if gex_read and gex_read != "N/A":
        gex_read_line = f"*GEX Read:* {gex_read}\n"

    source_tag = metrics.get('data_source', 'yf')

    # --- Build detail message with filtered sections ---
    lines = [
        f"*{emoji} {status}*",
        "\u2500\u2500\u2500 Trade Setup \u2500\u2500\u2500",
    ]

    if div_line:
        lines.append(div_line.rstrip())
    if vol_shift_line:
        lines.append(vol_shift_line.rstrip())

    lines.append(f"*Setup:* {data.get('setup', 'N/A')}")
    lines.append(f"*Trigger:* {data.get('action_plan', 'N/A')}")
    lines.append(f"*Target:* {data.get('target', 'N/A')}  \u2502  *Stop:* {data.get('invalidation', 'N/A')}")
    lines.append(f"*R:R:* {rr}  \u2502  *Size:* {pos_suggestion}")
    lines.append(f"*Fractal:* {frac_agree} \u2502 *MTF:* {mtf_align}")

    if gex_read_line:
        lines.append(gex_read_line.rstrip())
    if rev_alert:
        lines.append(rev_alert.strip())
    if sweep_line:
        lines.append(sweep_line.rstrip())
    if fractal_line:
        lines.append(fractal_line.rstrip())

    # --- Indicators section (only show lines with real data) ---
    indicator_lines = []
    if session_line:
        indicator_lines.append(session_line.rstrip())
    if mtf_line:
        indicator_lines.append(mtf_line.rstrip())
    if tick_line:
        indicator_lines.append(tick_line.rstrip())
    if vpoc_line:
        indicator_lines.append(vpoc_line.rstrip())
    if regime_line:
        indicator_lines.append(regime_line.rstrip())
    if gex_line:
        indicator_lines.append(gex_line.rstrip())
    if flow_line:
        indicator_lines.append(flow_line.rstrip())
    if avwap_line:
        indicator_lines.append(avwap_line.rstrip())

    # Core levels (always show)
    vwap_val = metrics.get('vwap_val', 'N/A')
    poc_val = metrics.get('vpoc', {}).get('poc', 'N/A')
    ib_val = metrics.get('ib', {}).get('ib_status', 'N/A')
    g_call_val = metrics.get('g_call', 'N/A')
    g_put_val = metrics.get('g_put', 'N/A')
    net_gamma = metrics.get('gamma_detail', {}).get('net_gamma', '')
    gamma_src = metrics.get('gamma_detail', {}).get('source', '')
    delta_val = metrics.get('cum_delta_bias', 'N/A')

    indicator_lines.append(f"`VWAP  ` {vwap_val} ({vwap_s})")
    indicator_lines.append(f"`POC   ` {poc_val} \u2502 IB: {ib_val}")
    if str(g_call_val) != "N/A" or str(g_put_val) != "N/A":
        indicator_lines.append(f"`Gamma ` {g_call_val} / {g_put_val} ({net_gamma}) [{gamma_src}]")
    if context_line != "Normal":
        indicator_lines.append(f"`Context` {context_line}")
    indicator_lines.append(f"`Delta ` {delta_val}")

    # Delta at key levels
    dal = metrics.get("delta_levels", {})
    dal_bias = dal.get("net_bias", "NEUTRAL")
    if dal_bias != "NEUTRAL":
        dal_str = f"{dal_bias} ({dal.get('bias_score', 0):+d})"
        dal_top = dal.get("strongest_signal", "")
        indicator_lines.append(f"`\u0394@Lvl ` {dal_str}")
        if dal_top and dal_top != "N/A":
            indicator_lines.append(f"        {dal_top}")
    elif dal.get("signal") and dal["signal"] != "N/A":
        indicator_lines.append(f"`\u0394@Lvl ` {dal.get('signal', 'N/A')}")

    if indicator_lines:
        lines.append("")
        lines.append("\u2500\u2500\u2500 Indicators \u2500\u2500\u2500")
        lines.extend(indicator_lines)

    # --- P&L footer ---
    pnl_emoji = "\U0001f4c8" if "+" in pnl_str[:15] else "\U0001f4c9"
    lines.append("")
    lines.append(f"{pnl_emoji} {pnl_str} [{source_tag}]")

    msg = "\n".join(lines)
    return msg.strip(), status


# Module-level store for full detail message (used by /detail command)
_latest_detail_msg: str = ""
_latest_detail_time: str = ""  # timestamp of when detail was captured
_detail_msg_lock = threading.Lock()


def format_action_card(data: dict, metrics: dict, pos_suggestion: str, sleep_mode: str,
                       pnl_str: str = "", open_trades: list = None,
                       skipped_rr: float = None) -> str:
    """Produce a descriptive Telegram action card with thesis, confluence, and P&L."""
    from session_utils import is_news_approaching

    verdict = data.get("verdict", "FLAT").upper()
    conf = int(data.get("confidence", 0))
    regime = metrics.get("regime", {}).get("regime", "N/A")

    if "BULL" in verdict:
        arrow = _GREEN
    elif "BEAR" in verdict:
        arrow = _RED
    else:
        arrow = _WHITE

    # --- Header ---
    price = data.get("current_price", "N/A")
    now_card = now_et()
    time_str = now_card.strftime("%H:%M")
    lines = [
        f"*{arrow} {verdict} {conf}%* \u2502 {regime}",
        f"ES `{price}` \u2502 {time_str} ET",
    ]

    # --- Trade Levels (only for directional verdicts) ---
    _has_trade = "BULL" in verdict or "BEAR" in verdict
    if _has_trade:
        target = data.get("target", "N/A")
        stop = data.get("invalidation", "N/A")
        MIN_TARGET_PTS = 4.0
        try:
            _p, _t, _s = float(price), float(target), float(stop)
            if "BULL" in verdict and _t < _p + MIN_TARGET_PTS:
                _t = _p + MIN_TARGET_PTS
                target = f"{_t:.2f}"
            elif "BEAR" in verdict and _t > _p - MIN_TARGET_PTS:
                _t = _p - MIN_TARGET_PTS
                target = f"{_t:.2f}"
            _reward = abs(_t - _p)
            _risk = abs(_p - _s)
            rr = f"{_reward / _risk:.1f}:1" if _risk > 0 else "N/A"
        except (ValueError, TypeError):
            rr = data.get("risk_reward", "N/A")

        lines.append("")
        lines.append(f"`Entry  {price}`")
        lines.append(f"`Target {target}`")
        lines.append(f"`Stop   {stop}`")
        if skipped_rr is not None:
            lines.append(f"`R:R    {rr}` \u26a0 *SKIPPED (min 1.2)*")
        else:
            lines.append(f"`R:R    {rr}    Size: {pos_suggestion}`")

    # --- Thesis ---
    setup = data.get("setup", "").strip()
    if setup and setup != "N/A":
        if len(setup) > 120:
            setup = setup[:117] + "..."
        lines.append("")
        lines.append(f"\u25b8 *Thesis:* {setup}")

    # --- Confluence ---
    is_rth = dtime(9, 30) <= dtime(now_card.hour, now_card.minute) <= dtime(16, 15)

    fractal_proj = metrics.get("fractal", {}).get("projection", None)
    if fractal_proj:
        f_dir = getattr(fractal_proj, "direction", None) or "N/A"
        f_conf = getattr(fractal_proj, "confidence", None)
        fractal_str = f"{f_dir} ({f_conf}%)" if f_conf else f_dir
    else:
        fractal_str = "N/A"

    mtf = metrics.get("mtf_momentum", {})
    mtf_str = mtf.get("alignment", "N/A")

    lines.append("")
    lines.append("\u2500\u2500\u2500 Confluence \u2500\u2500\u2500")
    lines.append(f"{_dir_emoji(fractal_str)} `Fractal` {fractal_str}")
    lines.append(f"{_dir_emoji(mtf_str)} `MTF    ` {mtf_str}")

    if is_rth:
        flow_data = metrics.get("flow_data", {})
        flow_bias = flow_data.get("flow_bias", "N/A")
        g_call = metrics.get("g_call", "N/A")
        g_put = metrics.get("g_put", "N/A")
        gamma_detail = metrics.get("gamma_detail", {})
        net_gamma = gamma_detail.get("net_gamma", "")
        lines.append(f"{_dir_emoji(flow_bias)} `Flow   ` {flow_bias}")
        lines.append(f"`GEX    ` Call {g_call} / Put {g_put} ({net_gamma})")

    rvol = metrics.get("rvol", {})
    rvol_val = rvol.get("rvol", 1.0) if isinstance(rvol, dict) else 1.0
    rvol_status = rvol.get("status", "N/A") if isinstance(rvol, dict) else "N/A"
    lines.append(f"`RVOL   ` {rvol_val:.1f}x ({rvol_status})")

    sweeps = metrics.get("liq_sweeps", {})
    active_sweep = sweeps.get("active_sweep", "NONE")
    if active_sweep != "NONE":
        sweep_signal = sweeps.get("signal", "")
        sweep_str = sweep_signal[:60] + "..." if len(sweep_signal) > 60 else sweep_signal
        lines.append(f"{_dir_emoji(active_sweep)} `Sweep  ` {sweep_str}")

    div = metrics.get("divergence", {})
    if div.get("score", 0) > 0:
        lines.append(f"`Diverg ` {div.get('severity', 'None')}")

    # --- P&L ---
    pnl_part = pnl_str if pnl_str else "P&L: $0 | Open Trades: $0"
    pnl_emoji = "\U0001f4c8" if "+" in pnl_part[:15] else "\U0001f4c9"
    lines.append("")
    lines.append(f"\u2500\u2500\u2500 P&L \u2500\u2500\u2500")
    lines.append(f"{pnl_emoji} {pnl_part}")

    # --- Open trades ---
    if open_trades:
        lines.append("")
        lines.append("\u2500\u2500\u2500 Open Trades \u2500\u2500\u2500")
        for t in open_trades:
            t_verdict = t.get("verdict", "?")
            t_entry = t.get("price", 0)
            t_target = t.get("target", 0)
            t_stop = t.get("stop", 0)
            t_pnl = t.get("pnl", 0)
            t_cts = int(t.get("contracts", 1) or 1)
            t_d = t_pnl * CFG.POINT_VALUE * t_cts
            is_long = ts.is_long(t_verdict)
            d = "L" if is_long else "S"
            t_emoji = _GREEN if is_long else _RED
            lines.append(
                f"{t_emoji} *{d} {t_cts}x* @ `{t_entry:.2f}` \u2502 "
                f"*{'+' if t_d >= 0 else '-'}${abs(t_d):,.0f}*"
            )
            lines.append(
                f"  TP `{t_target:.2f}` \u2502 SL `{t_stop:.2f}`"
            )

    # --- Footer ---
    news_approaching, news_label, _, _ = is_news_approaching()
    if news_approaching:
        lines.append("")
        lines.append(f"\u26a0 *News:* {news_label}")

    lines.append("")
    lines.append(f"_Next: {sleep_mode}_")

    return "\n".join(lines)


# =================================================================
# --- DAILY RECAP AND HEARTBEAT ---
# =================================================================

def send_daily_recap(journal: Journal):
    """End-of-day performance summary."""
    from charts import ChartLibrary
    import trade_status as _ts

    stats = journal.get_today_stats()
    weekly = journal.get_weekly_stats()
    cl = ChartLibrary()
    wk_d = weekly['pnl'] * CFG.POINT_VALUE

    if stats["total"] == 0:
        msg = (
            f"*\U0001f4ca END OF DAY RECAP*\n"
            f"\n"
            f"No trades today. Capital preserved.\n"
            f"\n"
            f"*Week:* {weekly['total']} trades \u2502 "
            f"{'+' if wk_d >= 0 else '-'}${abs(wk_d):,.0f}\n"
            f"Archive: {cl.get_status()}"
        )
    else:
        wr = f"{stats['win_rate']:.0f}%"
        real_d = stats['realized'] * CFG.POINT_VALUE
        float_d = stats['floating'] * CFG.POINT_VALUE
        net_d = stats['net'] * CFG.POINT_VALUE

        net_emoji = "\U0001f4c8" if net_d >= 0 else "\U0001f4c9"
        lines = [
            f"*\U0001f4ca END OF DAY RECAP*",
            "",
            "\u2500\u2500\u2500 Performance \u2500\u2500\u2500",
            f"`Trades  ` {stats['total']} ({stats['wins']}W {stats['losses']}L)",
            f"`Win Rate` {wr}",
            f"`Realized` *{'+' if real_d >= 0 else '-'}${abs(real_d):,.0f}*",
            f"`Open    ` {'+' if float_d >= 0 else '-'}${abs(float_d):,.0f}",
            f"{net_emoji} `Net  ` *{'+' if net_d >= 0 else '-'}${abs(net_d):,.0f}*",
        ]

        # Avg W/L and expectancy
        today_str = now_et().strftime("%Y-%m-%d")
        try:
            with journal._conn() as conn:
                rows = conn.execute(
                    f"SELECT pnl, contracts, status FROM trades "
                    f"WHERE timestamp LIKE ? AND status IN {_ts.CLOSED_SQL}",
                    (f"{today_str}%",),
                ).fetchall()
            closed_trades = [dict(r) for r in rows]
            if len(closed_trades) >= 2:
                wins_pts = [t["pnl"] for t in closed_trades if t["status"] == _ts.WIN]
                losses_pts = [abs(t["pnl"]) for t in closed_trades if t["status"] in _ts.LOSS_STATUSES]
                avg_w = sum(wins_pts) / len(wins_pts) if wins_pts else 0
                avg_l = sum(losses_pts) / len(losses_pts) if losses_pts else 0
                total_pnl_d = sum(
                    t["pnl"] * int(t.get("contracts", 1) or 1) * CFG.POINT_VALUE
                    for t in closed_trades
                )
                expectancy_d = total_pnl_d / len(closed_trades) if closed_trades else 0
                lines.append(f"`Avg W/L ` +{avg_w:.1f} / -{avg_l:.1f} pts")
                lines.append(f"`E[trade]` *{'+' if expectancy_d >= 0 else '-'}${abs(expectancy_d):,.0f}*")
        except Exception as e:
            logger.debug(f"Daily recap stats computation failed: {e}")

        lines.append("")
        lines.append(
            f"*Week:* {weekly['total']} trades \u2502 "
            f"{'+' if wk_d >= 0 else '-'}${abs(wk_d):,.0f}"
        )
        lines.append(f"Archive: {cl.get_status()}")
        msg = "\n".join(lines)

    # Append post-trade review if available
    try:
        today_str = now_et().strftime("%Y-%m-%d")
        review = journal.get_trade_review(today_str)
        if review and review.get("summary"):
            review_lines = ["\n\u2500\u2500\u2500 Post-Trade Review \u2500\u2500\u2500"]
            for lesson in review.get("lessons", [])[:3]:
                review_lines.append(f"\u25b8 #{lesson['trade_id']}: {lesson['lesson'][:120]}")
            review_lines.append(f"\n*TIL:* _{review['summary'][:200]}_")
            review_text = "\n".join(review_lines)
            if len(msg) + len(review_text) < 4000:
                msg += "\n" + review_text
    except Exception as e:
        logger.debug(f"Post-trade review append failed: {e}")

    send_telegram(msg)
    logger.info("Daily recap sent.")


def send_heartbeat(journal: Journal = None, accuracy_tracker=None):
    """Hourly performance scorecard sent to Telegram."""
    from charts import ChartLibrary
    from session_utils import get_session_phase
    import trade_status as _ts

    now = now_et()
    now_str = now.strftime("%H:%M")

    if journal is None:
        from journal import Journal as _J
        journal = _J()

    stats = journal.get_today_stats()
    weekly = journal.get_weekly_stats()
    cl = ChartLibrary()

    realized_d = stats['realized'] * CFG.POINT_VALUE
    open_d = stats['floating'] * CFG.POINT_VALUE
    net_d = stats['net'] * CFG.POINT_VALUE
    weekly_d = weekly['pnl'] * CFG.POINT_VALUE
    closed_count = stats['wins'] + stats['losses']
    wr_str = f"{stats['win_rate']:.0f}%" if closed_count > 0 else "--"
    phase = get_session_phase()

    net_emoji = "\U0001f4c8" if net_d >= 0 else "\U0001f4c9"
    lines = [
        f"*\u23f0 HOURLY UPDATE* \u2502 {now_str} ET",
        f"Session: {phase}",
        "",
        "\u2500\u2500\u2500 Today \u2500\u2500\u2500",
        f"`Trades  ` {stats['total']} ({stats['wins']}W {stats['losses']}L) WR: {wr_str}",
        f"`Realized` *{'+' if realized_d >= 0 else '-'}${abs(realized_d):,.0f}*",
        f"`Open    ` {'+' if open_d >= 0 else '-'}${abs(open_d):,.0f}",
        f"{net_emoji} `Net  ` *{'+' if net_d >= 0 else '-'}${abs(net_d):,.0f}*",
    ]

    # --- Avg win / avg loss / expectancy ---
    if closed_count >= 2:
        today_str = now.strftime("%Y-%m-%d")
        try:
            with journal._conn() as conn:
                rows = conn.execute(
                    f"SELECT pnl, contracts, status FROM trades "
                    f"WHERE timestamp LIKE ? AND status IN {_ts.CLOSED_SQL}",
                    (f"{today_str}%",),
                ).fetchall()
            closed_trades = [dict(r) for r in rows]
            wins_pts = [t["pnl"] for t in closed_trades if t["status"] == _ts.WIN]
            losses_pts = [abs(t["pnl"]) for t in closed_trades if t["status"] in _ts.LOSS_STATUSES]
            avg_w = sum(wins_pts) / len(wins_pts) if wins_pts else 0
            avg_l = sum(losses_pts) / len(losses_pts) if losses_pts else 0
            total_pnl_d = sum(
                t["pnl"] * int(t.get("contracts", 1) or 1) * CFG.POINT_VALUE
                for t in closed_trades
            )
            expectancy_d = total_pnl_d / len(closed_trades) if closed_trades else 0
            lines.append(
                f"`Avg W/L ` +{avg_w:.1f} / -{avg_l:.1f} pts"
            )
            lines.append(
                f"`E[trade]` *{'+' if expectancy_d >= 0 else '-'}${abs(expectancy_d):,.0f}*"
            )
        except Exception as e:
            logger.debug(f"Heartbeat stats computation failed: {e}")

    # --- Open positions ---
    open_trades = journal.get_open_trades()
    if open_trades:
        lines.append("")
        lines.append("\u2500\u2500\u2500 Open Trades \u2500\u2500\u2500")
        for t in open_trades:
            t_cts = int(t.get("contracts", 1) or 1)
            t_pnl = float(t.get("pnl", 0))
            t_pnl_d = t_pnl * CFG.POINT_VALUE * t_cts
            t_entry = float(t.get("price", 0))
            d = "L" if ts.is_long(t.get("verdict", "")) else "S"
            t_emoji = "\U0001f7e2" if t_pnl_d >= 0 else "\U0001f534"
            lines.append(
                f"{t_emoji} *{d} {t_cts}x* @ `{t_entry:.2f}` \u2502 "
                f"*{'+' if t_pnl_d >= 0 else '-'}${abs(t_pnl_d):,.0f}* ({t_pnl:+.1f} pts)"
            )

    # --- Accuracy breakdown ---
    if accuracy_tracker is not None:
        try:
            acc = accuracy_tracker.get_recent_accuracy()
            by_session = acc.get("by_session", {})
            by_dir = acc.get("by_direction", {})

            session_parts = []
            for s, sdata in sorted(by_session.items()):
                if sdata["total"] >= 2:
                    s_wr = sdata["wins"] / sdata["total"] * 100
                    session_parts.append(f"{s}: {s_wr:.0f}% ({sdata['total']})")
            if session_parts:
                lines.append("")
                lines.append("\u2500\u2500\u2500 Accuracy (last 20) \u2500\u2500\u2500")
                for sp in session_parts:
                    lines.append(f"\u25b8 {sp}")

            dir_parts = []
            for d_name, ddata in by_dir.items():
                if ddata["total"] >= 2:
                    d_wr = ddata["wins"] / ddata["total"] * 100
                    dir_parts.append(f"{d_name}: {d_wr:.0f}% ({ddata['total']})")
            if dir_parts:
                lines.append("Direction: " + " \u2502 ".join(dir_parts))

            if acc.get("streak", 0) >= 2:
                lines.append(f"Streak: {acc['streak']} {acc['streak_type']}")
        except Exception as e:
            logger.debug(f"Heartbeat accuracy stats failed: {e}")

    # --- Footer ---
    wk_emoji = "\U0001f4c8" if weekly_d >= 0 else "\U0001f4c9"
    lines.append("")
    lines.append(
        f"{wk_emoji} *Week:* {weekly['total']} trades \u2502 "
        f"{'+' if weekly_d >= 0 else '-'}${abs(weekly_d):,.0f}"
    )

    send_telegram("\n".join(lines))


# =================================================================
# --- TELEGRAM COMMAND LISTENER ---
# =================================================================

class TelegramCommandListener:
    """Polls Telegram for /status, /recap, /pnl, /quiet, /full, /health commands."""

    def __init__(self, journal: Journal, alert_tier: AlertTier = None, health=None):
        self.journal = journal
        self.alert_tier = alert_tier
        self.health = health  # HealthMetrics instance (optional)
        self.last_update_id = 0
        self._running = False

    def start(self):
        self._running = True
        try:
            res = requests.get(
                f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/getUpdates",
                params={"offset": -1, "timeout": 1},
                timeout=5,
            )
            if res.status_code == 200:
                results = res.json().get("result", [])
                if results:
                    self.last_update_id = results[-1]["update_id"]
                    logger.info(f"Telegram: flushed buffered updates, starting from {self.last_update_id}")
        except Exception as e:
            logger.warning(f"Telegram flush failed (non-critical): {e}")
        thread = threading.Thread(target=self._poll_loop, daemon=True)
        thread.start()
        logger.info("Telegram command listener started.")

    def stop(self):
        self._running = False

    def _poll_loop(self):
        from session_utils import get_session_phase
        from charts import ChartLibrary

        backoff = 5
        consecutive_errors = 0
        while self._running:
            try:
                res = requests.get(
                    f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/getUpdates",
                    params={"offset": self.last_update_id + 1, "timeout": 30},
                    timeout=35,
                )
                if res.status_code != 200:
                    time.sleep(backoff)
                    continue

                consecutive_errors = 0
                backoff = 10
                data = res.json()
                for update in data.get("result", []):
                    self.last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip().lower()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != CFG.TELEGRAM_CHAT_ID:
                        continue

                    if text == "/status":
                        self._handle_status(get_session_phase, ChartLibrary)
                    elif text == "/recap":
                        send_daily_recap(self.journal)
                    elif text == "/pnl":
                        self._handle_pnl()
                    elif text == "/trades":
                        self._handle_trades()
                    elif text == "/risk":
                        self._handle_risk()
                    elif text == "/skipped":
                        self._handle_skipped()
                    elif text == "/charts":
                        self._handle_charts(ChartLibrary)
                    elif text == "/quiet":
                        if self.alert_tier:
                            self.alert_tier.quiet_mode = True
                            send_telegram("*Quiet mode ON* -- Only sending updates when signals change.\nUse /full to restore full updates.")
                        else:
                            send_telegram("Alert system not available.")
                    elif text == "/full":
                        if self.alert_tier:
                            self.alert_tier.quiet_mode = False
                            send_telegram("*Full mode ON* -- Sending complete analysis every cycle.")
                        else:
                            send_telegram("Alert system not available.")
                    elif text == "/detail":
                        self._handle_detail()
                    elif text == "/health":
                        self._handle_health()
                    elif text == "/week":
                        self._handle_week()
                    elif text == "/month":
                        self._handle_month()
                    elif text == "/signals":
                        self._handle_signals()
                    elif text == "/shadow":
                        self._handle_shadow()
                    elif text.startswith("/config"):
                        self._handle_config(text)
                    elif text == "/help":
                        send_telegram(
                            "*Commands:*\n"
                            "/status -- Current bot state\n"
                            "/detail -- Full analysis from last cycle\n"
                            "/pnl -- Today's P&L\n"
                            "/trades -- Open advisory positions\n"
                            "/risk -- Risk summary & exposure\n"
                            "/skipped -- Today's skipped trades\n"
                            "/signals -- Signal decomposition summary\n"
                            "/shadow -- Shadow mode comparison stats\n"
                            "/config -- View/set config values\n"
                            "/health -- Bot performance metrics\n"
                            "/week -- Weekly performance roll-up\n"
                            "/month -- Monthly performance roll-up\n"
                            "/recap -- Full day recap\n"
                            "/charts -- Chart archive status\n"
                            "/quiet -- Only alert on changes\n"
                            "/full -- Full analysis every cycle\n"
                            "/help -- This message"
                        )

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.Timeout,
                    ConnectionResetError):
                time.sleep(2)
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 2 or consecutive_errors % 10 == 0:
                    logger.warning(f"Command listener: {consecutive_errors} errors (backoff {backoff}s): {e}")
                time.sleep(backoff)
                backoff = min(backoff + 5, 120)

    def _handle_status(self, get_session_phase, ChartLibrary):
        stats = self.journal.get_today_stats()
        verdict = self.journal.get_last_verdict()
        phase = get_session_phase()
        cl = ChartLibrary()
        net_d = stats['net'] * CFG.POINT_VALUE
        realized_d = stats['realized'] * CFG.POINT_VALUE
        open_d = stats['floating'] * CFG.POINT_VALUE
        closed_count = stats['wins'] + stats['losses']
        wr_str = f"{stats['win_rate']:.0f}%" if closed_count > 0 else "--"
        send_telegram(
            f"*\u2139 Bot Status*\n"
            f"\n"
            f"`Phase   ` {phase}\n"
            f"`Verdict ` {verdict}\n"
            f"\n"
            f"\u2500\u2500\u2500 Today \u2500\u2500\u2500\n"
            f"`Trades  ` {stats['total']} ({stats['wins']}W {stats['losses']}L) WR: {wr_str}\n"
            f"`Realized` {'+' if realized_d >= 0 else '-'}${abs(realized_d):,.0f}\n"
            f"`Open    ` {'+' if open_d >= 0 else '-'}${abs(open_d):,.0f}\n"
            f"`Net     ` *{'+' if net_d >= 0 else '-'}${abs(net_d):,.0f}*\n"
            f"\n"
            f"Charts: {cl.get_status()}"
        )

    def _handle_pnl(self):
        stats = self.journal.get_today_stats()
        realized_d = stats['realized'] * CFG.POINT_VALUE
        floating_d = stats['floating'] * CFG.POINT_VALUE
        net_d = stats['net'] * CFG.POINT_VALUE
        closed_count = stats['wins'] + stats['losses']
        wr_str = f"{stats['win_rate']:.0f}%" if closed_count > 0 else "--"
        send_telegram(
            f"*\U0001f4b0 Advisory P&L*\n"
            f"\n"
            f"`Realized` *{'+' if realized_d >= 0 else '-'}${abs(realized_d):,.0f}*\n"
            f"`Open    ` {'+' if floating_d >= 0 else '-'}${abs(floating_d):,.0f}\n"
            f"`Net     ` *{'+' if net_d >= 0 else '-'}${abs(net_d):,.0f}*\n"
            f"\n"
            f"{stats['total']} trades \u2502 {stats['wins']}W {stats['losses']}L \u2502 WR: {wr_str}"
        )

    def _handle_trades(self):
        open_trades = self.journal.get_open_trades()
        if not open_trades:
            send_telegram("*Open Trades*\nNo advisory trades currently open.")
            return
        lines = [f"*Open Trades ({len(open_trades)})*", ""]
        for t in open_trades:
            verdict = t.get("verdict", "?")
            entry = float(t.get("price", 0))
            target_val = float(t.get("target", 0))
            stop_val = float(t.get("stop", 0))
            conf = t.get("confidence", 0)
            cts = int(t.get("contracts", 1) or 1)
            pnl = float(t.get("pnl", 0))
            trade_ts = t.get("timestamp", "?")
            d = "L" if ts.is_long(verdict) else "S"
            pnl_d = pnl * CFG.POINT_VALUE * cts
            lines.append(f"*{d} {cts}x {verdict}* ({conf}%) -- {trade_ts}")
            lines.append(f"`Entry ` {entry:.2f}")
            lines.append(f"`Target` {target_val:.2f}")
            lines.append(f"`Stop  ` {stop_val:.2f}")
            lines.append(f"`P&L   ` *{'+' if pnl_d >= 0 else '-'}${abs(pnl_d):,.0f}* ({pnl:+.2f} pts)")
            lines.append("")
        send_telegram("\n".join(lines))

    def _handle_detail(self):
        with _detail_msg_lock:
            msg = _latest_detail_msg
            detail_time = _latest_detail_time
        if msg:
            send_telegram(f"_Analysis from {detail_time} ET_\n\n{msg}")
        else:
            send_telegram("No analysis yet -- waiting for first cycle to complete.")

    def _handle_risk(self):
        """Show current risk exposure and distance to daily loss limit."""
        stats = self.journal.get_today_stats()
        open_trades = self.journal.get_open_trades()

        realized_d = stats['realized'] * CFG.POINT_VALUE
        floating_d = stats['floating'] * CFG.POINT_VALUE
        net_d = stats['net'] * CFG.POINT_VALUE
        loss_limit = CFG.MAX_DAILY_LOSS  # negative number like -500

        # Calculate total exposure
        total_cts = sum(int(t.get("contracts", 1) or 1) for t in open_trades)
        long_cts = sum(int(t.get("contracts", 1) or 1) for t in open_trades if ts.is_long(t.get("verdict", "")))
        short_cts = total_cts - long_cts

        # Max single-trade risk (worst open trade's stop distance)
        max_risk_d = 0
        for t in open_trades:
            entry = float(t.get("price", 0))
            stop = float(t.get("stop", 0))
            cts = int(t.get("contracts", 1) or 1)
            dist = abs(entry - stop) * CFG.POINT_VALUE * cts
            max_risk_d = max(max_risk_d, dist)

        # Distance to loss limit
        remaining = abs(loss_limit) + net_d  # e.g., 500 + (-200) = 300 remaining
        pct_today = (net_d / CFG.ACCOUNT_SIZE) * 100

        lines = [
            f"*\U0001f6e1 RISK SUMMARY*",
            "",
            "\u2500\u2500\u2500 Exposure \u2500\u2500\u2500",
            f"`Open     ` {total_cts} ct ({long_cts}L / {short_cts}S)" if total_cts > 0 else "`Open     ` No positions",
        ]
        if max_risk_d > 0:
            lines.append(f"`Max Risk ` ${max_risk_d:,.0f} (largest open)")

        lines.extend([
            "",
            "\u2500\u2500\u2500 P&L \u2500\u2500\u2500",
            f"`Realized ` *{'+' if realized_d >= 0 else '-'}${abs(realized_d):,.0f}*",
            f"`Floating ` {'+' if floating_d >= 0 else '-'}${abs(floating_d):,.0f}",
            f"`Net      ` *{'+' if net_d >= 0 else '-'}${abs(net_d):,.0f}*",
            "",
            "\u2500\u2500\u2500 Limits \u2500\u2500\u2500",
            f"`Loss Cap ` ${abs(loss_limit):,.0f} ({'+' if remaining >= 0 else '-'}${abs(remaining):,.0f} remaining)",
            f"`Account  ` ${CFG.ACCOUNT_SIZE:,.0f} ({pct_today:+.2f}% today)",
        ])

        send_telegram("\n".join(lines))

    def _handle_skipped(self):
        """Show today's skipped trades and their phantom P&L."""
        skipped = self.journal.get_skipped_trades_today()
        if not skipped:
            send_telegram("*Skipped Trades*\nNo trades skipped today.")
            return

        total_phantom = 0
        lines = [f"*Skipped Trades ({len(skipped)})*", ""]
        for t in skipped:
            verdict = t.get("verdict", "?")
            entry = float(t.get("price", 0))
            pnl = float(t.get("pnl", 0))
            reason = t.get("reasoning", "")[:40]
            d = "L" if ts.is_long(verdict) else "S"
            pnl_d = pnl * CFG.POINT_VALUE
            total_phantom += pnl_d
            lines.append(
                f"{d} @ `{entry:.2f}` \u2502 "
                f"*{'+' if pnl_d >= 0 else '-'}${abs(pnl_d):,.0f}* ({pnl:+.2f} pts)"
                f"{f' | {reason}' if reason else ''}"
            )
        lines.append("")
        lines.append(
            f"*Phantom P&L:* {'+' if total_phantom >= 0 else '-'}${abs(total_phantom):,.0f} "
            f"(if all taken)"
        )
        send_telegram("\n".join(lines))

    def _handle_charts(self, ChartLibrary):
        cl = ChartLibrary()
        today_str = now_et().strftime("%Y-%m-%d")
        today_shots = cl.get_day_screenshots(today_str)
        recent = cl.get_recent_days(5)
        recent_lines = "\n".join(
            f"  * {d} -- {len(cl.get_day_screenshots(d))} screenshots"
            for d in recent
        ) if recent else "  None yet"
        send_telegram(
            f"*Chart Library*\n"
            f"Total: {cl.get_total_days()} days | {cl.get_total_screenshots()} screenshots\n"
            f"Today: {len(today_shots)} captures\n\n"
            f"Recent days:\n{recent_lines}"
        )

    def _handle_signals(self):
        """Show signal decomposition summary from SignalLogger."""
        try:
            from signal_logger import SignalLogger
            sl = SignalLogger()
            send_telegram(sl.format_telegram_report(hours=24))
        except Exception as e:
            send_telegram(f"*Signals*\nError: {e}")

    def _handle_shadow(self):
        """Show shadow mode comparison stats."""
        try:
            from shadow_mode import ShadowMode
            sm = ShadowMode(shadow_params=CFG.SHADOW_PARAMS)
            send_telegram(sm.format_telegram_report())
        except Exception as e:
            send_telegram(f"*Shadow*\nError: {e}")

    def _handle_config(self, text: str):
        """View or set hot-reloadable config values.

        /config           → show current values
        /config KEY VALUE → update and persist
        """
        from bot_config import reload_config, save_config_override, _RELOADABLE_FIELDS

        parts = text.strip().split(maxsplit=2)  # ["/config", key?, value?]

        if len(parts) == 1:
            # Show current values
            lines = ["*⚙️ Live Config*", ""]
            for key in sorted(_RELOADABLE_FIELDS.keys()):
                val = getattr(CFG, key, "?")
                lines.append(f"`{key}` = {val}")
            send_telegram("\n".join(lines))
            return

        if len(parts) < 3:
            send_telegram("Usage: `/config KEY VALUE`\nExample: `/config RR_MINIMUM 1.5`")
            return

        key = parts[1].upper()
        raw_value = parts[2]

        if key not in _RELOADABLE_FIELDS:
            send_telegram(
                f"❌ `{key}` is not reloadable.\n\n"
                f"Allowed: {', '.join(sorted(_RELOADABLE_FIELDS.keys()))}"
            )
            return

        expected_type = _RELOADABLE_FIELDS[key]
        try:
            if expected_type == float:
                value = float(raw_value)
            elif expected_type == int:
                value = int(raw_value)
            elif expected_type in (dict, list):
                import json
                value = json.loads(raw_value)
            else:
                value = raw_value
        except (ValueError, TypeError) as e:
            send_telegram(f"❌ Invalid value for `{key}`: {e}")
            return

        old_val = getattr(CFG, key, None)
        setattr(CFG, key, value)
        if save_config_override(key, value):
            send_telegram(
                f"✅ `{key}` updated\n"
                f"Old: `{old_val}`\n"
                f"New: `{value}`\n\n"
                f"_Persisted to config\\_overrides.json_"
            )
        else:
            send_telegram(
                f"⚠️ `{key}` updated in memory but failed to persist to file.\n"
                f"Old: `{old_val}` → New: `{value}`"
            )

    def _handle_health(self):
        """Show bot performance metrics from HealthMetrics."""
        if not self.health:
            send_telegram("*Bot Health*\nHealth metrics not available.")
            return

        # Get 24-hour summary
        summary = self.health.get_summary(hours=24)
        recent = self.health.get_recent_metrics(hours=4)

        lines = [
            "*\U0001f3e5 Bot Health (24h)*",
            "",
            "\u2500\u2500\u2500 Performance \u2500\u2500\u2500",
            f"`Cycles   ` {summary['cycles']}",
            f"`Avg Cycle` {summary['avg_cycle_ms']:,.0f}ms",
            f"`Avg Data ` {summary.get('avg_ibkr_ms', 0):,.0f}ms" if 'avg_ibkr_ms' in summary else "",
            f"`Avg Claude` {summary['avg_claude_ms']:,.0f}ms",
            f"`Avg Fractal` {summary['avg_fractal_ms']:,.0f}ms",
            f"`Errors   ` {summary['total_errors']}",
        ]
        # Remove empty lines from conditional fields
        lines = [l for l in lines if l != ""]

        if recent:
            # Show last 3 cycles
            lines.extend(["", "\u2500\u2500\u2500 Recent Cycles \u2500\u2500\u2500"])
            for m in recent[:3]:
                verdict = m.get('verdict', 'N/A')
                conf = m.get('confidence', 0)
                total = m.get('total_cycle_ms', 0)
                src = m.get('data_source', '?')
                ts_str = m.get('timestamp', '?')
                # Just show HH:MM from timestamp
                if len(ts_str) >= 16:
                    ts_str = ts_str[11:16]
                err_flag = " \u26a0" if m.get('errors', 0) > 0 else ""
                lines.append(f"`{ts_str}` {verdict} @{conf}% | {total:.0f}ms | {src}{err_flag}")

        send_telegram("\n".join(lines))

    def _handle_week(self):
        """Detailed weekly performance roll-up."""
        weekly = self.journal.get_weekly_stats(detailed=True)
        if weekly["total"] == 0:
            send_telegram("*Weekly Performance*\nNo closed trades this week.")
            return
        pnl_d = weekly["pnl"] * CFG.POINT_VALUE
        best_d = weekly.get("best_trade_pts", 0) * CFG.POINT_VALUE
        worst_d = weekly.get("worst_trade_pts", 0) * CFG.POINT_VALUE
        exp_d = weekly.get("expectancy_pts", 0) * CFG.POINT_VALUE
        wr = weekly.get("win_rate", 0)
        lines = [
            "*\U0001f4ca WEEKLY ROLL-UP*",
            "",
            "\u2500\u2500\u2500 Performance \u2500\u2500\u2500",
            f"`Trades  ` {weekly['total']} ({weekly['wins']}W {weekly['losses']}L)",
            f"`Win Rate` {wr:.0f}%",
            f"`P&L     ` *{'+' if pnl_d >= 0 else '-'}${abs(pnl_d):,.0f}*",
            f"`Best    ` +${best_d:,.0f}",
            f"`Worst   ` -${abs(worst_d):,.0f}",
            f"`E[trade]` *{'+' if exp_d >= 0 else '-'}${abs(exp_d):,.0f}*",
        ]
        # Prior week comparison
        prev_pnl = weekly.get("prev_week_pnl")
        if prev_pnl is not None:
            prev_d = prev_pnl * CFG.POINT_VALUE
            prev_total = weekly.get("prev_week_trades", 0)
            delta = pnl_d - prev_d
            lines.append("")
            lines.append(
                f"*Prior Week:* {prev_total} trades \u2502 "
                f"{'+' if prev_d >= 0 else '-'}${abs(prev_d):,.0f} "
                f"(\u0394: {'+' if delta >= 0 else '-'}${abs(delta):,.0f})"
            )
        # Signal performance
        sig_perf = self.journal.get_signal_performance(min_trades=3)
        if sig_perf:
            lines.append("")
            lines.append("\u2500\u2500\u2500 Best Signals \u2500\u2500\u2500")
            for sp in sig_perf[:3]:
                lines.append(f"\u25b8 {sp['combo']}: {sp['win_rate']:.0f}% ({sp['total']} trades)")
        # Signal quality (from signal_scores)
        try:
            sq = self.journal.get_signal_quality_stats(days=7)
            if sq.get("total", 0) >= 3:
                lines.append("")
                lines.append("\u2500\u2500\u2500 Signal Quality \u2500\u2500\u2500")
                lines.append(f"`30m  ` {sq['correct_30m_pct']:.0f}% correct")
                lines.append(f"`60m  ` {sq['correct_60m_pct']:.0f}% correct")
                lines.append(f"`120m ` {sq['correct_120m_pct']:.0f}% correct")
        except Exception as e:
            logger.debug(f"Signal quality stats failed: {e}")
        send_telegram("\n".join(lines))

    def _handle_month(self):
        """Monthly performance roll-up."""
        monthly = self.journal.get_monthly_stats()
        if monthly["total"] == 0:
            send_telegram("*Monthly Performance*\nNo closed trades this month.")
            return
        pnl_d = monthly["pnl"] * CFG.POINT_VALUE
        best_d = monthly.get("best_trade_pts", 0) * CFG.POINT_VALUE
        worst_d = monthly.get("worst_trade_pts", 0) * CFG.POINT_VALUE
        exp_d = monthly.get("expectancy_pts", 0) * CFG.POINT_VALUE
        wr = monthly.get("win_rate", 0)
        month_name = now_et().strftime("%B %Y")
        lines = [
            f"*\U0001f4ca {month_name.upper()} PERFORMANCE*",
            "",
            "\u2500\u2500\u2500 Performance \u2500\u2500\u2500",
            f"`Trades  ` {monthly['total']} ({monthly['wins']}W {monthly['losses']}L)",
            f"`Win Rate` {wr:.0f}%",
            f"`P&L     ` *{'+' if pnl_d >= 0 else '-'}${abs(pnl_d):,.0f}*",
            f"`Best    ` +${best_d:,.0f}",
            f"`Worst   ` -${abs(worst_d):,.0f}",
            f"`E[trade]` *{'+' if exp_d >= 0 else '-'}${abs(exp_d):,.0f}*",
        ]
        send_telegram("\n".join(lines))
