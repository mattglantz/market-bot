"""
Claude AI analysis prompt builder, cycle memory, and alert tier system.

Extracted from market_bot_v26.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from bot_config import logger, now_et, CFG

if TYPE_CHECKING:
    from market_data import MarketData


# =================================================================
# --- CROSS-CYCLE MEMORY (Narrative Buffer) ---
# =================================================================

_CYCLE_MEMORY_FILE = Path("logs/cycle_memory.json")


class CycleMemory:
    """Stores last N cycle summaries so Claude can track developing setups.
    Persists to disk so context survives restarts (#11)."""

    def __init__(self, max_cycles: int = 5):
        self.max_cycles = max_cycles
        self.cycles: list = []
        self._load()

    def _load(self):
        """Load persisted cycle memory from disk."""
        try:
            if _CYCLE_MEMORY_FILE.exists():
                data = json.loads(_CYCLE_MEMORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _required = {"time", "verdict", "confidence"}
                    valid = [c for c in data if isinstance(c, dict) and _required.issubset(c.keys())]
                    if len(valid) < len(data):
                        logger.warning(f"CycleMemory: dropped {len(data) - len(valid)} invalid entries")
                    self.cycles = valid[-self.max_cycles:]
                    logger.info(f"CycleMemory: loaded {len(self.cycles)} cycles from disk")
        except Exception as e:
            logger.warning(f"CycleMemory load failed (starting fresh): {e}")
            self.cycles = []

    def _save(self):
        """Persist cycle memory to disk."""
        try:
            _CYCLE_MEMORY_FILE.parent.mkdir(exist_ok=True)
            _CYCLE_MEMORY_FILE.write_text(
                json.dumps(self.cycles, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"CycleMemory save failed: {e}")

    def record(self, price: float, verdict: str, confidence: int,
               fractal_dir: str, fractal_conf: int, key_signals: dict,
               data: dict = None):
        now = now_et()

        price_change = 0.0
        if self.cycles:
            price_change = price - self.cycles[-1]["price"]

        entry = {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%Y-%m-%d"),
            "price": round(price, 2),
            "price_change": round(price_change, 2),
            "verdict": verdict,
            "confidence": confidence,
            "fractal": f"{fractal_dir} ({fractal_conf}%)",
            "gex": key_signals.get("gex_regime", "N/A"),
            "flow": key_signals.get("flow_bias", "N/A"),
            "mtf": key_signals.get("mtf_alignment", "N/A"),
            "divergence": key_signals.get("divergence", "NONE"),
            "sweep": key_signals.get("sweep", "NONE"),
            "setup": (data or {}).get("setup", "N/A")[:80].replace("AFTER-HOURS ", "").replace("AFTER HOURS ", ""),
        }

        self.cycles.append(entry)
        if len(self.cycles) > self.max_cycles:
            self.cycles.pop(0)
        self._save()

    def get_prompt_text(self) -> str:
        if not self.cycles:
            return ""

        lines = [f"Last {len(self.cycles)} cycles (newest last):"]
        for c in self.cycles:
            chg = f"{c['price_change']:+.2f}" if c["price_change"] != 0 else "--"
            lines.append(
                f"  [{c['time']}] {c['price']} ({chg}) | "
                f"{c['verdict']} @ {c['confidence']}% | "
                f"Fractal: {c['fractal']} | GEX: {c['gex']} | "
                f"MTF: {c['mtf']} | "
                f"{'DIV: ' + c['divergence'] + ' | ' if c['divergence'] not in ('NONE', 'LOW (ALIGNED)') else ''}"
                f"{'SWEEP: ' + c['sweep'] + ' | ' if c['sweep'] != 'NONE' else ''}"
                f"Setup: {c['setup']}"
            )

        if len(self.cycles) >= 2:
            first_p = self.cycles[0]["price"]
            last_p = self.cycles[-1]["price"]
            total_move = last_p - first_p
            verdicts = [c["verdict"] for c in self.cycles]
            same_count = sum(1 for v in verdicts if v == verdicts[-1])
            lines.append(
                f"  SUMMARY: Price {first_p} -> {last_p} ({total_move:+.2f} pts over {len(self.cycles)} cycles). "
                f"Verdict consistency: {same_count}/{len(self.cycles)} '{verdicts[-1]}'"
            )

        return "\n".join(lines)

    def get_last(self) -> dict:
        return self.cycles[-1] if self.cycles else {}


# =================================================================
# --- TIERED ALERT SYSTEM ---
# =================================================================

class AlertTier:
    """Compares current analysis vs previous cycle to determine alert level."""

    def __init__(self):
        self.quiet_mode = False
        self.last_state = {}

    def should_send_full(self, current_state: dict, first_run: bool = False) -> bool:
        if not self.quiet_mode:
            return True

        if first_run or not self.last_state:
            self.last_state = current_state
            return True

        changes = []

        if current_state.get("verdict") != self.last_state.get("verdict"):
            changes.append("verdict_change")

        conf_diff = abs(
            int(current_state.get("confidence", 0)) -
            int(self.last_state.get("confidence", 0))
        )
        if conf_diff >= 15:
            changes.append("confidence_shift")

        curr_div = current_state.get("divergence_alert", False)
        prev_div = self.last_state.get("divergence_alert", False)
        if curr_div and not prev_div:
            changes.append("new_divergence")

        curr_sweep = current_state.get("sweep", "NONE")
        prev_sweep = self.last_state.get("sweep", "NONE")
        if curr_sweep != "NONE" and curr_sweep != prev_sweep:
            changes.append("new_sweep")

        if current_state.get("fractal_dir") != self.last_state.get("fractal_dir"):
            changes.append("fractal_shift")

        if current_state.get("vol_alert") and not self.last_state.get("vol_alert"):
            changes.append("vol_expansion")

        self.last_state = current_state

        return len(changes) > 0

    def format_quiet_message(self, price: float, verdict: str,
                             confidence: int, fractal_dir: str,
                             price_change: float = 0.0) -> str:
        emoji = "G" if "BULL" in verdict else "R" if "BEAR" in verdict else "W"
        chg = f" ({price_change:+.1f})" if price_change != 0 else ""
        return f"[{emoji}] {price:.2f}{chg} | {verdict} @ {confidence}% | Fractal: {fractal_dir} | No change"


# =================================================================
# --- POST-TRADE REVIEW PROMPT ---
# =================================================================

def build_review_prompt(closed_trades: list, current_price: float) -> str:
    """Build a Claude prompt to review today's closed trades."""
    trade_blocks = []
    for t in closed_trades:
        signals = {}
        try:
            signals = json.loads(t.get("signals", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        entry = float(t["price"])
        pnl = float(t["pnl"])
        cts = int(t.get("contracts", 1) or 1)
        pnl_dollars = pnl * CFG.POINT_VALUE * cts
        trade_blocks.append(
            f"Trade #{t['id']}: {t['verdict']} {cts}x @ {entry:.2f}\n"
            f"  Target: {t.get('target', 0):.2f} | Stop: {t.get('stop', 0):.2f}\n"
            f"  Outcome: {t['status']} | P&L: {pnl:+.2f} pts (${pnl_dollars:+,.0f})\n"
            f"  Confidence: {t['confidence']}% | Session: {t.get('session', 'N/A')}\n"
            f"  Signals: {json.dumps(signals)}\n"
            f"  Reasoning: {(t.get('reasoning') or 'N/A')[:200]}"
        )
    trades_text = "\n\n".join(trade_blocks) if trade_blocks else "No trades."

    return (
        "You are reviewing today's ES futures trades for a post-trade journal.\n\n"
        f"Today's closed trades:\n\n{trades_text}\n\n"
        f"Current ES price (close): {current_price:.2f}\n\n"
        "For each trade, provide a 2-3 sentence lesson: what worked, what didn't, "
        "and what to do differently next time. Focus on signal quality and execution.\n"
        "Then provide an overall \"Today I Learned\" summary (2-3 sentences) "
        "synthesizing the day's key takeaway.\n\n"
        "Respond with ONLY this JSON:\n"
        "{\"reviews\": [{\"trade_id\": <id>, \"lesson\": \"<2-3 sentences>\"}, ...], "
        "\"summary\": \"<Today I Learned summary>\"}"
    )


# =================================================================
# --- STATIC SYSTEM PROMPT (cached by Anthropic across calls) ---
# =================================================================

SYSTEM_PROMPT = """You are an expert ES futures intraday analyst. You combine pattern recognition with market microstructure to make high-probability trade calls.

You will receive a triple-screen chart image (10-min, 1-hour, Daily) and comprehensive quantitative data. Analyze them using the rules below.

<signal_rules>
GEX:
- LONG GAMMA: Market pins. Fade breakouts, sell premium, mean-reversion.
- SHORT GAMMA: Moves accelerate. Ride breakouts, buy premium, momentum.
- Near GEX flip level: Regime transition zone -- reduce size, wait for confirmation.

OPTIONS FLOW:
- Large unusual flow (>$1M single strike, vol > 3x OI) often precedes moves. Trade WITH the big money.

TAPE:
- BULLISH tape + large institutional buys = high conviction long.
- Absorption = hidden resting orders defending a level. BUY absorption = support. SELL absorption = resistance.
- Large trade bias (>= 50 lots) carries more weight -- institutions move size.

SWEEPS:
- A confirmed sweep (ran stops then reversed) is a HIGH-PROBABILITY reversal signal.
- BULL TRAP = short signal. BEAR TRAP = long signal.

DIVERGENCE:
- >=4 divergent indicators = HIGH PROBABILITY REVERSAL. Tighten stops or exit. >=6 = EXTREME -- go FLAT.

VOL SHIFT:
- Vol expansion > 2x = prior signals may be invalid. Tighten stops. Vol compression = range-bound.

CYCLE HISTORY:
- Track developing setups across cycles. If you flagged a level 2 cycles ago and price is now testing it, increase conviction. If your last 3 calls were wrong, reduce confidence.
</signal_rules>

<hard_rules>
- EXTREME DIVERGENCE (>=75%): MUST BE FLAT -- reversal imminent
- STRONG SELLING breadth: DO NOT GO LONG
- STRONG BUYING breadth: DO NOT GO SHORT
- VOL EXPANSION > 2.5x: Reassess all signals. Tighten stops 50%.
- LONG GAMMA (GEX): Fade breakouts. SHORT GAMMA: Ride breakouts.
- VWAP OVERBOUGHT: NO NEW LONGS
- VWAP OVERSOLD: NO NEW SHORTS
- LUNCH CHOP: 90%+ confidence or FLAT
- RVOL < 0.5: Skeptical of any directional call
- VIX BACKWARDATION: Extra caution on longs
- MTF CONFLICTED + TICK NEUTRAL: Strong FLAT signal
- ACTIVE LIQUIDITY SWEEP: High-probability reversal
- HEAVY PUT FLOW (P/C > 1.5): Extra caution on longs
- HEAVY CALL FLOW (P/C < 0.65): Extra caution on shorts
- IV SKEW elevated (put/call > 1.3): Hedging demand high -- caution on longs
- CROSS-ASSET DIVERGENCE (ES rising + VIX rising): Warning -- validate with other signals
- FULL BEARISH MTF + HIGH DIVERGENCE: DO NOT GO BULLISH. The pipeline will block it. Go FLAT or BEARISH.
- FULL BULLISH MTF + HIGH DIVERGENCE: DO NOT GO BEARISH. The pipeline will block it. Go FLAT or BULLISH.
- CONTRADICTING FRACTAL: If your verdict opposes the fractal direction (>=60% confidence), you need STRONG supporting evidence (4+ confirming signals). Without it, the pipeline will force FLAT. When fractal and momentum disagree, prefer FLAT.
- Return your TRUE directional conviction. The pipeline applies regime penalties -- do NOT self-censor to FLAT.
- OVERNIGHT/GLOBEX SESSION: ES futures trade nearly 24 hours (6PM-5PM ET). You MUST analyze overnight sessions the same as RTH. Do NOT dismiss signals just because it's outside 9:30AM-4PM. Overnight moves set the next day's gap and opening drive. Provide real directional analysis with targets and stops -- never return "after-hours locked" or similar dismissals.
</hard_rules>

<confidence_calibration>
Your raw confidence is post-processed by a pipeline that SUBTRACTS regime penalties (up to -15 pts) before comparing to a threshold (typically 60%).
  85-100 = Multiple strong signals aligned, high-probability setup with clear R:R
  75-84  = Good directional signal with supporting confirmation
  65-74  = Lean with some confirmation -- DEFAULT for any directional call
  55-64  = Weak lean, likely filtered to FLAT after penalties
  0-54   = Genuinely conflicted -- use FLAT verdict
If you have ANY directional lean with at least one confirming signal, start at 65+. A raw 65 with -5 penalty becomes 60 (barely clears). A raw 58 becomes 53 (wasted).
</confidence_calibration>

Respond with ONLY JSON. No markdown, no code fences, no extra text.
Required keys: reasoning, current_price, verdict (Bullish/Bearish/FLAT), confidence, target, invalidation, setup, action_plan, reversal_reason, risk_reward, fractal_agreement (CONFIRM/CONTRADICT/PARTIAL), day_type_read, mtf_alignment (ALIGNED/CONFLICTED/PARTIAL), sweep_detected (NONE/BULL_TRAP/BEAR_TRAP), gex_read, divergence_flag (NONE/MODERATE/HIGH/EXTREME)."""


# =================================================================
# --- ANALYSIS PROMPT BUILDER ---
# =================================================================

def _format_delta_levels(dal: dict) -> str:
    """Format delta-at-levels breakdown for the Claude analysis prompt."""
    levels = dal.get("levels", [])
    if not levels:
        return "  No level data available"
    active = [lv for lv in levels if lv.get("pattern") != "NEUTRAL"]
    if not active:
        return "  No active patterns at key levels"
    parts = []
    for lv in active[:6]:
        parts.append(
            f"  {lv['name']} ({lv['price']:.0f}, {lv['role']}): "
            f"{lv['pattern']} str={lv['strength']} | "
            f"buy={lv['buy_vol']:.0f} sell={lv['sell_vol']:.0f} net={lv['net_delta']:+.0f} | "
            f"dist={lv['distance']:+.1f}pts"
        )
    return "\n".join(parts)


def _format_mtf_details(mtf: dict) -> str:
    """Format MTF detail line with ROC, ADX, and strength for the Claude prompt."""
    details = mtf.get("details", {})
    if not details:
        return ""
    parts = []
    for k, label in [("5m", "5m"), ("1h", "1H"), ("daily", "Daily")]:
        d = details.get(k, {})
        if d.get("direction", "N/A") == "N/A":
            continue
        parts.append(
            f"{label}: ROC={d.get('norm_roc', 0):+.1f}ATR "
            f"ADX={d.get('adx', 0):.0f} "
            f"Str={d.get('strength', 0):.1f}"
        )
    return "  " + " | ".join(parts) if parts else ""


def build_analysis_prompt(md: MarketData, metrics: dict, prev_verdict: str, pnl_str: str, accuracy_str: str = "", cycle_memory_text: str = "", calibration_str: str = "") -> str:
    """Claude-optimized prompt with fractal recognition + full indicator suite."""

    fractal = metrics.get("fractal", {})
    fractal_prompt = fractal.get("prompt_text", "\nFRACTAL: No data available.\n")
    proj = fractal.get("projection", None)
    if proj is None:
        class _EmptyProj:
            direction = "NEUTRAL"; confidence = 0; bullish_pct = 50; bearish_pct = 50
            expected_close_vs_current = 0; upside_target = 0; downside_target = 0; match_count = 0
        proj = _EmptyProj()
    gap = metrics.get("gap", {})
    opening = metrics.get("opening_type", {})
    rvol = metrics.get("rvol", {})
    vix_ts = metrics.get("vix_term", {})
    mtf = metrics.get("mtf_momentum", {})
    vpoc_mig = metrics.get("vpoc_migration", {})
    tick = metrics.get("tick_proxy", {})
    regime = metrics.get("regime", {})
    tod = metrics.get("tod_context", "")
    cross_corr = metrics.get("cross_corr", {})
    iv_skew = metrics.get("iv_skew", {})
    tf = mtf.get("timeframes", {})
    avwaps = metrics.get("anchored_vwaps", {})
    sweeps = metrics.get("liq_sweeps", {})

    sweep_str = sweeps.get('signal', 'No sweeps')
    cluster_str = sweeps.get('cluster_signal', 'N/A')

    prompt = f"""Data source: {metrics.get('data_source', 'yfinance')}.

<fractal_analysis>
{fractal_prompt}
</fractal_analysis>

<session_context>
CURRENT STATE:
  Price: {md.current_price:.2f} | Session: {metrics['session']} | {pnl_str}
  Day of Week: {now_et().strftime('%A')}

GAP ANALYSIS:
  {gap.get('summary', 'N/A')}
  Gap: {gap.get('gap_size', 0):+.1f} pts ({gap.get('gap_pct', 0):+.3f}%) | Status: {gap.get('fill_status', 'N/A')}

OPENING TYPE (Market Profile / Dalton):
  Type: {opening.get('type', 'N/A')} | Bias: {opening.get('bias', 'N/A')}
  Detail: {opening.get('description', 'N/A')}

RELATIVE VOLUME: {rvol.get('rvol', 'N/A')}x -- {rvol.get('status', 'N/A')}

VIX REGIME:
  VIX: {vix_ts.get('vix', 'N/A')} | VIX9D: {vix_ts.get('vix9d', 'N/A')} | Term Structure: {vix_ts.get('structure', 'N/A')}
  Signal: {vix_ts.get('signal', 'N/A')}
  Regime: {regime.get('regime', 'N/A')} | FLAT Threshold: {regime.get('flat_threshold', 60)}%
  Adjustments: {regime.get('summary', 'Standard')}
</session_context>

<dealer_gamma_exposure>
  GEX Regime: {metrics.get('gex_regime', {}).get('regime', 'N/A')}
  GEX Flip Level: {metrics.get('gex_regime', {}).get('flip_level', 'N/A')} ({metrics.get('gex_regime', {}).get('distance_to_flip', 'N/A')} pts from price)
  Playbook: {metrics.get('gex_regime', {}).get('playbook', 'N/A')}
  Signal: {metrics.get('gex_regime', {}).get('signal', 'N/A')}
</dealer_gamma_exposure>

<options_flow>
  Flow Bias: {metrics.get('flow_data', {}).get('flow_bias', 'N/A')} | Confidence: {metrics.get('flow_data', {}).get('flow_confidence', 'N/A')}
  Weighted P/C: {metrics.get('flow_data', {}).get('weighted_pc_ratio', 'N/A')} (Raw P/C: {metrics.get('flow_data', {}).get('pc_ratio', 'N/A')}) | Vol vs Avg: {metrics.get('flow_data', {}).get('vol_vs_avg', 'N/A')}x
  Call Vol: {metrics.get('flow_data', {}).get('total_call_vol', 0):,} (${metrics.get('flow_data', {}).get('call_notional_m', 0):.1f}M) | Put Vol: {metrics.get('flow_data', {}).get('total_put_vol', 0):,} (${metrics.get('flow_data', {}).get('put_notional_m', 0):.1f}M)
  Unusual Activity: {metrics.get('flow_data', {}).get('signal', 'N/A')}
</options_flow>

{f"""<tape_flow>
{metrics.get("tape_text", "")}
</tape_flow>""" if metrics.get("tape_text") else ""}

<multi_timeframe_momentum>
  5m: {tf.get('5m', 'N/A')} | 1H: {tf.get('1h', 'N/A')} | Daily: {tf.get('daily', 'N/A')}
  ALIGNMENT: {mtf.get('alignment', 'N/A')} (Score: {mtf.get('score', 0)})
  {_format_mtf_details(mtf)}
  Signal: {mtf.get('signal', 'N/A')}
</multi_timeframe_momentum>

<vpoc_migration>
  POC Migration: {vpoc_mig.get('migration', 'N/A')} -- {vpoc_mig.get('trend', 'N/A')}
  Signal: {vpoc_mig.get('signal', 'N/A')}
  Naked POCs (untested prior magnets): {', '.join(f"{p['poc']} ({p['distance']:+.1f}pts, {p['date']})" for p in vpoc_mig.get('naked_pocs', [])) or 'None nearby'}
</vpoc_migration>

<tick_proxy>
  Breadth TICK Proxy: {tick.get('tick_proxy', 50):.0f}% upticking | {tick.get('extreme', 'N/A')}
  Cumulative: {tick.get('cumulative', 0):+.2f} | Bullish Extremes (1hr): {tick.get('bullish_extremes', 0)} | Bearish: {tick.get('bearish_extremes', 0)}
  Signal: {tick.get('signal', 'N/A')}
</tick_proxy>

<liquidity_sweeps>
  Active Sweep: {sweeps.get('active_sweep', 'NONE')}
  Signal: {sweep_str}
  Stop Clusters: {cluster_str}
  Recent Swing Highs: {', '.join(f"{s['price']:.2f}" for s in sweeps.get('swing_highs', [])) or 'N/A'}
  Recent Swing Lows: {', '.join(f"{s['price']:.2f}" for s in sweeps.get('swing_lows', [])) or 'N/A'}
</liquidity_sweeps>

<reference_levels>
  Prior Day: H {metrics.get('prior', {}).get('prev_high', 'N/A')} | L {metrics.get('prior', {}).get('prev_low', 'N/A')} | C {metrics.get('prior', {}).get('prev_close', 'N/A')} | Range {metrics.get('prior', {}).get('prev_range', 'N/A')}
  VWAP: {metrics.get('vwap_val', 'N/A')} ({metrics.get('vwap_status', 'N/A')})
    Bands: +2SD {metrics.get('vwap_levels', {}).get('upper_2','N/A')} | +1SD {metrics.get('vwap_levels', {}).get('upper_1','N/A')} | -1SD {metrics.get('vwap_levels', {}).get('lower_1','N/A')} | -2SD {metrics.get('vwap_levels', {}).get('lower_2','N/A')}
  Anchored VWAPs:
    Weekly: {avwaps.get('weekly_vwap', 'N/A')} | Monthly: {avwaps.get('monthly_vwap', 'N/A')} | {avwaps.get('swing_label', 'Swing')}: {avwaps.get('swing_vwap', 'N/A')}
    Convergence: {avwaps.get('convergence', 'N/A')}
    Signal: {avwaps.get('signal', 'N/A')}
  Volume POC: {metrics.get('vpoc', {}).get('poc', 'N/A')} | VAH: {metrics.get('vpoc', {}).get('vah', 'N/A')} | VAL: {metrics.get('vpoc', {}).get('val', 'N/A')}
  Initial Balance: H {metrics.get('ib', {}).get('ib_high', 'N/A')} | L {metrics.get('ib', {}).get('ib_low', 'N/A')} | {metrics.get('ib', {}).get('ib_status', 'N/A')}
  Gamma Walls (0DTE): Call {metrics.get('g_call', 'N/A')} ({metrics.get('gamma_detail', {}).get('distance_to_call', '?')} pts away) | Put {metrics.get('g_put', 'N/A')} ({metrics.get('gamma_detail', {}).get('distance_to_put', '?')} pts away)
    Secondary: Call2 {metrics.get('gamma_detail', {}).get('call_wall_2', 'N/A')} | Put2 {metrics.get('gamma_detail', {}).get('put_wall_2', 'N/A')}
    Net Gamma: {metrics.get('gamma_detail', {}).get('net_gamma', 'N/A')} | Source: {metrics.get('gamma_detail', {}).get('source', 'N/A')}
</reference_levels>

<market_internals>
  Breadth (Mag 7): {metrics.get('breadth', 'N/A')}
  Cumulative Delta: {metrics.get('cum_delta_bias', 'N/A')} (Raw: {metrics.get('cum_delta_val', 0):+,.0f})
  Fractal Bias (1H 50MA): {metrics.get('structure', {}).get('fractal', 'N/A')}
  VSA: {metrics.get('structure', {}).get('vsa', 'N/A')}
  Wick: {metrics.get('structure', {}).get('wick', 'N/A')}
  ATR(14, 15m): {metrics.get('structure', {}).get('atr', 'N/A')}
</market_internals>

<delta_at_levels>
  Order Flow at Key Levels ({metrics.get('delta_levels', {}).get('source', 'N/A')} data):
  Aggregate Bias: {metrics.get('delta_levels', {}).get('net_bias', 'N/A')} ({metrics.get('delta_levels', {}).get('bias_score', 0):+d})
  Strongest: {metrics.get('delta_levels', {}).get('strongest_signal', 'N/A')}
  Per-Level Breakdown:
{_format_delta_levels(metrics.get('delta_levels', {}))}
  INTERPRETATION: ABSORPTION at support = level defended (bullish). BREAKOUT with delta = real move.
  EXHAUSTION = no conviction at level (likely reversal). REJECTION = brief breach, delta flipped.
</delta_at_levels>

<cross_asset_correlation>
  Regime: {cross_corr.get('regime', 'N/A')}
  ES-DXY Corr: {cross_corr.get('es_dxy_corr', 'N/A')} | ES-VIX: {cross_corr.get('es_vix_corr', 'N/A')} | ES-TNX: {cross_corr.get('es_tnx_corr', 'N/A')}
  Divergences: {', '.join(cross_corr.get('divergences', [])) or 'None'}
  Signal: {cross_corr.get('signal', 'N/A')}
</cross_asset_correlation>

<iv_skew>
  ATM IV: {iv_skew.get('atm_iv', 'N/A')} | Put/Call Skew: {iv_skew.get('put_call_skew', 'N/A')}
  Skew Signal: {iv_skew.get('skew_signal', 'N/A')}
  Term Slope: {iv_skew.get('term_slope', 'N/A')} | Term Signal: {iv_skew.get('term_signal', 'N/A')}
  Source: {iv_skew.get('source', 'N/A')}
</iv_skew>

<vol_regime_shift>
  Status: {metrics.get('vol_shift', {}).get('shift', 'N/A')}
  Current 30m RVol: {metrics.get('vol_shift', {}).get('current_vol', 0):.1f}% | Prior: {metrics.get('vol_shift', {}).get('prior_vol', 0):.1f}% | Ratio: {metrics.get('vol_shift', {}).get('vol_ratio', 1.0):.2f}x
  Expected Hourly Range: +/-{metrics.get('vol_shift', {}).get('expected_hourly_range', 0):.1f} pts
  Signal: {metrics.get('vol_shift', {}).get('signal', 'N/A')}
</vol_regime_shift>

<divergence_detector>
  Severity: {metrics.get('divergence', {}).get('severity', 'N/A')}
  Score: {metrics.get('divergence', {}).get('score', 0)}/{metrics.get('divergence', {}).get('max_score', 0)} indicators diverge from price ({metrics.get('divergence', {}).get('price_direction', 'N/A')})
  Divergent: {', '.join(metrics.get('divergence', {}).get('divergent', [])) or 'None'}
  Aligned: {', '.join(metrics.get('divergence', {}).get('aligned', [])) or 'None'}
  Signal: {metrics.get('divergence', {}).get('signal', 'N/A')}
</divergence_detector>

<bot_performance>
{accuracy_str}
Time-of-Day Performance: {tod}
Regime Confidence Modifier: {regime.get('confidence_mod', 0):+d}%
{calibration_str if calibration_str else "No calibration data yet."}
</bot_performance>

<cycle_history>
{cycle_memory_text if cycle_memory_text else "First cycle -- no history yet."}
</cycle_history>

<analysis_framework>
Follow these steps in order using the data sections above. Do not skip any step.

1. FRACTAL (baseline): Start with the fractal projection. Only override with strong contradictory evidence.
2. DIVERGENCE (safety gate): If EXTREME (>=75%) -> FLAT. If HIGH (>=50%) -> reduce confidence 20%.
3. MOMENTUM + FLOW: Check MTF alignment, TICK, options flow. Full alignment = highest conviction.
4. GEX + VOL: Apply GEX playbook. If vol expansion > 2x, prior signals may be invalid.
5. CROSS-ASSET + IV: Check cross-asset regime and IV skew for confirmation or divergence.
6. SESSION: Opening type + gap + RVOL + sweeps. Open Drive + high RVOL = trend. Auction + low RVOL = range.
7. CHART + INTERNALS: Read the triple-screen. Breadth, delta, tape. Aligned = boost. Divergent = reduce.
8. LEVELS: Use reference levels, fractal targets, GEX flip for R:R >= 2:1.
9. VERDICT: Synthesize. Apply regime modifier ({regime.get('confidence_mod', 0):+d}%). FLAT when signals conflict.
</analysis_framework>

Previous verdict: {prev_verdict}. If flipping, explain why.
Current price for JSON: {md.current_price}"""

    return prompt
