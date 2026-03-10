"""
Confidence pipeline engine — extracted from market_bot_v26.py.

Contains:
- AccuracyTracker: tracks the bot's prediction accuracy and adjusts confidence
- calc_regime_adjustment(): adjusts confidence based on VIX/RVOL/MTF regime
- calc_signal_confluence(): detects when multiple independent signals agree
- apply_confidence_pipeline(): orchestrates the full confidence pipeline

Usage:
    from confidence_engine import (
        AccuracyTracker,
        calc_regime_adjustment,
        calc_signal_confluence,
        apply_confidence_pipeline,
    )
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

import trade_status as ts
from bot_config import CFG

if TYPE_CHECKING:
    from journal import Journal

logger = logging.getLogger("MarketBot")

_TIMESTAMP_FMTS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse a trade timestamp string, trying common formats."""
    for fmt in _TIMESTAMP_FMTS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


# =================================================================
# --- SELF-ACCURACY TRACKER ---
# =================================================================

class AccuracyTracker:
    """
    Tracks the bot's own prediction accuracy and adjusts confidence dynamically.

    FIXED (v25.3 confidence pipeline fix):
    - Penalty decays over time: if no trade in 2+ hours, halve; 4+ hours, zero
    - Forced FLATs count as "no data", not as continuing a losing streak
    - Cap penalty at -10 (was -15)
    - Track when last trade was logged to detect stale streaks
    """

    def __init__(self, journal: 'Journal', now_fn: Optional[Callable[[], datetime]] = None):
        self.journal = journal
        self._last_trade_time = None  # Track staleness
        self._cache: dict = {}
        self._cache_time: float = 0.0
        if now_fn is not None:
            self._now_fn = now_fn
        else:
            # Lazy import to avoid circular dependency
            self._now_fn = None

    def _now(self) -> datetime:
        if self._now_fn is not None:
            return self._now_fn()
        # Lazy import on first call
        from bot_config import now_et
        self._now_fn = now_et
        return now_et()

    def clear_cache(self):
        """Reset the accuracy cache."""
        self._cache = {}
        self._cache_time = 0.0

    def get_recent_accuracy(self, lookback: int = 20) -> dict:
        """Get accuracy stats for the last N closed trades."""
        if self._cache and (time.time() - self._cache_time) < CFG.ACCURACY_CACHE_TTL:
            return self._cache
        try:
            with self.journal._conn() as conn:
                rows = conn.execute(
                    "SELECT verdict, status, pnl, confidence, session, timestamp FROM trades "
                    f"WHERE status IN {ts.DECIDED_SQL} "
                    "ORDER BY id DESC LIMIT ?", (lookback,)
                ).fetchall()

            if not rows:
                return {"total": 0, "win_rate": 50, "streak": 0, "streak_type": "N/A",
                        "avg_win": 0, "avg_loss": 0, "confidence_adjustment": 0,
                        "by_session": {}, "by_direction": {}}

            trades = [dict(r) for r in rows]
            wins = [t for t in trades if t["status"] == ts.WIN]
            losses = [t for t in trades if t["status"] in ts.LOSS_STATUSES]

            win_rate = len(wins) / len(trades) * 100 if trades else 50
            avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0
            avg_loss = float(np.mean([abs(t["pnl"]) for t in losses])) if losses else 0

            # Current streak
            streak = 0
            streak_type = trades[0]["status"] if trades else "N/A"
            for t in trades:
                if t["status"] == streak_type or (
                    streak_type in ts.LOSS_STATUSES and
                    t["status"] in ts.LOSS_STATUSES
                ):
                    streak += 1
                else:
                    break

            # --- FIX: Decay streak penalty based on staleness ---
            hours_since_last = 0
            try:
                last_ts = _parse_timestamp(trades[0]["timestamp"])
                if last_ts:
                    hours_since_last = (self._now().replace(tzinfo=None) - last_ts).total_seconds() / 3600
            except Exception as e:
                logger.debug(f"Streak staleness parse failed: {e}")

            # --- FIX: Capped, decaying confidence adjustment ---
            # Require minimum 10 trades before applying penalties.
            # With < 10 trades the sample is too small — a few early losses
            # create a death spiral where the bot never trades again.
            conf_adj = 0
            if len(trades) >= 10:
                if streak >= 4 and streak_type in ts.LOSS_STATUSES:
                    conf_adj = -7
                elif streak >= 3 and streak_type in ts.LOSS_STATUSES:
                    conf_adj = -5
                elif streak >= 4 and streak_type == ts.WIN:
                    conf_adj = 5
                elif win_rate < 35:
                    conf_adj = -5
                elif win_rate > 65:
                    conf_adj = 5
                else:
                    conf_adj = 0
            elif len(trades) >= 5:
                # Small sample — only apply light adjustments
                if streak >= 4 and streak_type in ts.LOSS_STATUSES:
                    conf_adj = -3
                elif streak >= 4 and streak_type == ts.WIN:
                    conf_adj = 3

            # Decay: if no trade in 2+ hours, halve the penalty.
            # If no trade in 3+ hours (new session context), zero it out.
            # FIX (v26.2): Lowered full-reset from 4h to 3h so a pre-market
            # loss streak (e.g., 08:46) doesn't cripple RTH signals starting ~12:00.
            if conf_adj < 0 and hours_since_last > 3:
                conf_adj = 0  # New session — fresh start
            elif conf_adj < 0 and hours_since_last > 1.5:
                conf_adj = -((-conf_adj) // 2)  # Stale streak — halve penalty (round toward zero)

            # --- FIX: Hard cap at -7 (was -10) ---
            conf_adj = max(conf_adj, -7)

            # Accuracy by session phase
            by_session = {}
            for t in trades:
                s = t.get("session", "Unknown")
                if s not in by_session:
                    by_session[s] = {"wins": 0, "total": 0}
                by_session[s]["total"] += 1
                if t["status"] == ts.WIN:
                    by_session[s]["wins"] += 1

            # Accuracy by direction
            by_dir = {"BULLISH": {"wins": 0, "total": 0}, "BEARISH": {"wins": 0, "total": 0}}
            for t in trades:
                v = t.get("verdict", "").upper()
                if "BULL" in v:
                    by_dir["BULLISH"]["total"] += 1
                    if t["status"] == ts.WIN:
                        by_dir["BULLISH"]["wins"] += 1
                elif "BEAR" in v:
                    by_dir["BEARISH"]["total"] += 1
                    if t["status"] == ts.WIN:
                        by_dir["BEARISH"]["wins"] += 1

            result = {
                "total": len(trades),
                "win_rate": round(win_rate, 1),
                "streak": streak,
                "streak_type": "WINNING" if streak_type == ts.WIN else "LOSING",
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "confidence_adjustment": conf_adj,
                "hours_since_last_trade": round(hours_since_last, 1),
                "by_session": by_session,
                "by_direction": by_dir,
            }
            self._cache = result
            self._cache_time = time.time()
            return result
        except Exception as e:
            logger.warning(f"Accuracy tracker failed: {e}")
            return {"total": 0, "win_rate": 50, "streak": 0, "streak_type": "N/A",
                    "avg_win": 0, "avg_loss": 0, "confidence_adjustment": 0,
                    "by_session": {}, "by_direction": {}}

    def get_calibration_correction(self, raw_confidence: int) -> int:
        """
        Apply automatic calibration correction based on historical accuracy.

        If the bot says 70% confidence but historically wins only 55% at that level,
        the correction derate is applied: (actual_wr - expected_wr) / 2.

        Guards against single-session poisoning:
        - Requires trades from 2+ distinct dates to activate
        - Halved if last trade > 4 hours ago (stale session)

        Returns the corrected confidence (may be higher or lower than raw).
        """
        cal = self.get_calibration_table()
        if not cal:
            return raw_confidence

        # Find the bucket for this confidence
        bucket = f"{(raw_confidence // 10) * 10}-{(raw_confidence // 10) * 10 + 9}"
        bucket_data = cal.get(bucket)

        if not bucket_data or bucket_data["total"] < 10:
            return raw_confidence  # Not enough data for this bucket

        # --- FIX: Require session diversity ---
        # A single bad session (e.g., 8 losses on one morning) can poison a bucket.
        # Require trades from at least 2 distinct dates before applying correction.
        distinct_dates = bucket_data.get("distinct_dates", 0)
        if distinct_dates < 2:
            return raw_confidence

        actual_wr = bucket_data["win_rate"]
        # Expected win rate should roughly match confidence (50 conf = 50% wr, 70 conf = 70% wr)
        # But our thresholds mean we only trade above 60%, so normalize:
        expected_wr = min(raw_confidence, 85)  # Cap expected at 85% (nobody wins 95%)

        # Correction: half the gap between expected and actual
        correction = int((actual_wr - expected_wr) / 2)

        # Cap correction at ±10
        correction = max(-10, min(10, correction))

        # --- FIX: Staleness decay for calibration ---
        # If last trade was 4+ hours ago, this penalty may reflect an old session.
        # Halve it so a fresh session isn't crippled by stale data.
        if correction < 0:
            try:
                acc = self.get_recent_accuracy()
                hours_stale = acc.get("hours_since_last_trade", 0)
                if hours_stale > 8:
                    correction = 0  # Very stale — don't penalize
                elif hours_stale > 4:
                    correction = -((-correction) // 2)  # Halve penalty
            except Exception:
                pass

        corrected = raw_confidence + correction
        if correction != 0:
            logger.info(
                f"Calibration correction: {raw_confidence}% -> {corrected}% "
                f"(bucket {bucket}: {actual_wr:.0f}% actual, correction {correction:+d})"
            )

        return max(0, min(100, corrected))

    def adjust_confidence(self, raw_confidence: int) -> int:
        """Apply accuracy-based adjustment AND calibration correction to raw confidence.

        FIX (v26.2): Cap the COMBINED penalty at -10.
        Previously calibration (-10) + streak (-7) could stack to -17,
        far exceeding the intended -10 max documented in the pipeline.
        """
        # Step 1: Calibration correction (historical accuracy per bucket)
        calibrated = self.get_calibration_correction(raw_confidence)
        # Step 2: Streak/performance adjustment
        acc = self.get_recent_accuracy()
        adjusted = calibrated + acc["confidence_adjustment"]

        # Step 3: Cap COMBINED accuracy deduction at -10
        # Prevents death spiral where calibration + streak compound beyond intent
        if adjusted < raw_confidence - 10:
            adjusted = raw_confidence - 10
            logger.info(
                f"Accuracy cap: combined penalty capped at -10 "
                f"(calibration={calibrated - raw_confidence:+d}, "
                f"streak={acc['confidence_adjustment']:+d})"
            )

        return max(0, min(100, adjusted))

    def get_accuracy_context(self) -> str:
        """Formatted string for inclusion in the AI prompt."""
        acc = self.get_recent_accuracy()
        if acc["total"] == 0:
            return "No trade history yet — use standard confidence."

        parts = [
            f"Last {acc['total']} trades: {acc['win_rate']:.0f}% win rate.",
            f"Current streak: {acc['streak']} {acc['streak_type']}.",
            f"Avg win: +{acc['avg_win']:.1f} | Avg loss: -{acc['avg_loss']:.1f}.",
        ]

        if acc["confidence_adjustment"] != 0:
            parts.append(f"Confidence adjustment: {acc['confidence_adjustment']:+d}% (based on recent performance).")

        # Session-specific insight
        for s, data in acc.get("by_session", {}).items():
            if data["total"] >= 3:
                wr = data["wins"] / data["total"] * 100
                if wr < 30:
                    parts.append(f"WARNING: Only {wr:.0f}% accuracy during {s} — be extra cautious.")
                elif wr > 70:
                    parts.append(f"Strong track record during {s} ({wr:.0f}% accuracy).")

        # Time-of-day insight
        tod = self.get_time_of_day_accuracy()
        current_hour = self._now().hour
        hour_key = f"{current_hour:02d}"
        if hour_key in tod and tod[hour_key]["total"] >= 3:
            wr = tod[hour_key]["win_rate"]
            if wr < 35:
                parts.append(f"⚠️ TIME WARNING: Only {wr:.0f}% accuracy at {current_hour}:00 hour — historically weak.")
            elif wr > 65:
                parts.append(f"Strong accuracy at {current_hour}:00 hour ({wr:.0f}%).")

        # Calibration insight
        cal = self.get_calibration_table()
        if cal:
            over_conf = [(b, d) for b, d in cal.items() if d["total"] >= 5 and d["win_rate"] < 40]
            if over_conf:
                worst = min(over_conf, key=lambda x: x[1]["win_rate"])
                parts.append(
                    f"CALIBRATION WARNING: When you say {worst[0]}% confidence, "
                    f"actual win rate is only {worst[1]['win_rate']:.0f}%. Recalibrate."
                )

        return " ".join(parts)

    def get_calibration_table(self) -> dict:
        """
        Build a calibration table: for each raw-confidence bucket,
        compute the actual win rate.  Returns e.g.:
        {"50-59": {"total": 8, "wins": 3, "win_rate": 37.5, "distinct_dates": 3},
         "60-69": {"total": 15, "wins": 9, "win_rate": 60.0, "distinct_dates": 5}, ...}

        FIX (v26.2): Track distinct_dates per bucket to prevent single-session
        poisoning. A bucket with 15 trades all from one bad morning is unreliable.
        """
        try:
            with self.journal._conn() as conn:
                rows = conn.execute(
                    "SELECT confidence, status, timestamp FROM trades "
                    f"WHERE status IN {ts.DECIDED_SQL} "
                    "ORDER BY id DESC LIMIT 200"
                ).fetchall()
            if not rows:
                return {}

            buckets = {}
            for r in rows:
                conf = int(r["confidence"])
                bucket = f"{(conf // 10) * 10}-{(conf // 10) * 10 + 9}"
                if bucket not in buckets:
                    buckets[bucket] = {"total": 0, "wins": 0, "dates": set()}
                buckets[bucket]["total"] += 1
                if r["status"] == ts.WIN:
                    buckets[bucket]["wins"] += 1
                # Track distinct dates for session diversity check
                try:
                    ts_str = r["timestamp"] or ""
                    date_part = ts_str[:10]  # "YYYY-MM-DD"
                    if len(date_part) == 10:
                        buckets[bucket]["dates"].add(date_part)
                except Exception:
                    pass

            for k in buckets:
                t = buckets[k]["total"]
                w = buckets[k]["wins"]
                buckets[k]["win_rate"] = round(w / t * 100, 1) if t > 0 else 0
                buckets[k]["distinct_dates"] = len(buckets[k].pop("dates", set()))
            return buckets
        except Exception as e:
            logger.warning(f"Calibration table failed: {e}")
            return {}

    def get_calibration_context(self) -> str:
        """Formatted calibration data for inclusion in Claude prompt."""
        cal = self.get_calibration_table()
        if not cal:
            return "No calibration data yet."

        parts = ["CONFIDENCE CALIBRATION (your raw confidence vs actual outcomes):"]
        for bucket in sorted(cal.keys()):
            d = cal[bucket]
            if d["total"] >= 3:  # Only show buckets with meaningful sample
                emoji = "\U0001f7e2" if d["win_rate"] >= 55 else ("\U0001f534" if d["win_rate"] < 40 else "\u26aa")
                parts.append(
                    f"  {emoji} {bucket}%: {d['win_rate']:.0f}% actual win rate ({d['total']} trades)"
                )

        if len(parts) <= 1:
            return "Calibration data insufficient (need 3+ trades per bucket)."

        # Add guidance
        over_conf = [b for b, d in cal.items() if d["total"] >= 5 and d["win_rate"] < 40]
        under_conf = [b for b, d in cal.items() if d["total"] >= 5 and d["win_rate"] > 65]
        if over_conf:
            parts.append(f"  WARNING: Overconfident at {', '.join(over_conf)}% — lower your confidence in this range.")
        if under_conf:
            parts.append(f"  STRONG: Underconfident at {', '.join(under_conf)}% — you can be bolder here.")

        return "\n".join(parts)

    def get_time_of_day_accuracy(self) -> dict:
        """
        Track win rate by hour of day (e.g., 10am, 11am, etc.).
        The bot's accuracy at 10:15am may differ wildly from 2:30pm.
        """
        try:
            with self.journal._conn() as conn:
                rows = conn.execute(
                    "SELECT timestamp, status FROM trades "
                    f"WHERE status IN {ts.DECIDED_SQL} "
                    "ORDER BY id DESC LIMIT 100"
                ).fetchall()

            if not rows:
                return {}

            by_hour = {}
            for r in rows:
                try:
                    parsed_ts = _parse_timestamp(r["timestamp"])
                    if parsed_ts is None:
                        continue
                    hour_key = f"{parsed_ts.hour:02d}"
                    if hour_key not in by_hour:
                        by_hour[hour_key] = {"wins": 0, "total": 0}
                    by_hour[hour_key]["total"] += 1
                    if r["status"] == ts.WIN:
                        by_hour[hour_key]["wins"] += 1
                except Exception:
                    continue

            # Calculate win rates
            for k in by_hour:
                t = by_hour[k]["total"]
                w = by_hour[k]["wins"]
                by_hour[k]["win_rate"] = round(w / t * 100, 1) if t > 0 else 50

            return by_hour

        except Exception as e:
            logger.warning(f"Time-of-day accuracy failed: {e}")
            return {}

    def get_time_of_day_context(self) -> str:
        """Compact summary of time-of-day performance for the prompt."""
        tod = self.get_time_of_day_accuracy()
        if not tod:
            return "No time-of-day data yet."

        parts = []
        for hour in sorted(tod.keys()):
            d = tod[hour]
            if d["total"] >= 2:
                emoji = "🟢" if d["win_rate"] >= 60 else ("🔴" if d["win_rate"] < 40 else "⚪")
                parts.append(f"{emoji} {hour}:00 → {d['win_rate']:.0f}% ({d['total']} trades)")

        return " | ".join(parts) if parts else "Insufficient data."


# =================================================================
# --- ADAPTIVE REGIME CONFIDENCE ---
# =================================================================

def calc_regime_adjustment(vix_term: dict, rvol: dict, day_type: dict, mtf: dict) -> dict:
    """
    Adjusts confidence thresholds based on the current market regime.

    FIXED (v25.3 confidence pipeline fix):
    - Confidence penalties capped at -15 total (was unbounded, could reach -35)
    - flat_threshold uses a single assignment, not cascading max() ratchet
    - Penalties apply to EITHER confidence OR threshold, never both (no double squeeze)
    - Threshold maxes at 70 (was 75), since accuracy tracker already penalizes
    """
    try:
        vix_val = vix_term.get("vix", 0)
        structure = vix_term.get("structure", "N/A")
        rvol_val = rvol.get("rvol", 1.0)
        trend_prob = day_type.get("trend_probability", 50)
        mtf_score = mtf.get("score", 0)

        adjustments = []
        confidence_mod = 0
        # --- FIX: Single flat_threshold assignment based on VIX regime only ---
        # Other factors affect confidence_mod, NOT the threshold.
        # This eliminates the double-squeeze and the ratchet.
        if vix_val > 27:
            flat_threshold = 65      # Crisis-level vol
            regime = "HIGH VOLATILITY"
        elif vix_val > 20:
            flat_threshold = 62      # Elevated — slightly higher than normal
            regime = "ELEVATED VOLATILITY"
        elif vix_val > 14:
            flat_threshold = 58      # Normal
            regime = "NORMAL"
        elif vix_val > 0:
            flat_threshold = 55      # Low vol
            regime = "LOW VOLATILITY"
        else:
            flat_threshold = 60
            regime = "N/A"

        # --- Confidence modifiers (apply to score, NOT threshold) ---

        # VIX level — threshold already handles regime, so NO confidence penalty
        # for VIX > 20 or > 27. Only reward low-vol grinds where threshold is lowered.
        # FIX: Removes double-squeeze where VIX>20 raised threshold +3 AND penalized conf -3.
        if vix_val > 27:
            adjustments.append(f"VIX {vix_val:.0f} (HIGH): Threshold at 65%.")
        elif vix_val > 20:
            adjustments.append(f"VIX {vix_val:.0f} (ELEVATED): Threshold at 60%.")
        elif vix_val < 14 and vix_val > 0:
            confidence_mod += 3
            adjustments.append("VIX <14: +3 conf. Low vol grind.")

        # VIX term structure — softened; threshold already accounts for VIX regime
        if "STEEP BACKWARDATION" in str(structure):
            confidence_mod -= 3
            adjustments.append("Steep backwardation: -3 conf (extreme fear).")
        elif "BACKWARDATION" in str(structure):
            confidence_mod -= 1
            adjustments.append("Backwardation: -1 conf (elevated fear).")

        # RVOL — only penalize truly dead volume
        if rvol_val < 0.3:
            confidence_mod -= 5
            adjustments.append(f"RVOL {rvol_val:.1f}x: Extremely low volume. -5 conf.")
        elif rvol_val < 0.5:
            confidence_mod -= 3
            adjustments.append(f"RVOL {rvol_val:.1f}x: Low volume. -3 conf.")
        elif rvol_val > 2.0:
            confidence_mod += 3
            adjustments.append(f"RVOL {rvol_val:.1f}x: Heavy institutional activity. +3 conf.")

        # MTF alignment — reward alignment, no penalty for conflict
        # (choppy markets almost always show mixed MTF; penalizing it stacks with other penalties)
        if abs(mtf_score) == 100:
            confidence_mod += 5
            adjustments.append("Full MTF alignment: +5 conf.")
        elif abs(mtf_score) >= 60:
            confidence_mod += 3
            adjustments.append("Mostly aligned MTF: +3 conf.")
        elif mtf_score == 0 and mtf.get("alignment") == "MIXED / CONFLICTED":
            adjustments.append("Conflicting MTF: informational only, no penalty.")

        # Day type — informational only, no penalty
        if trend_prob >= 70:
            adjustments.append(f"Trend day likely ({trend_prob}%): Trade with momentum.")
        elif day_type.get("range_probability", 0) >= 70:
            adjustments.append(f"Range day likely: Fade extremes, smaller targets.")

        # --- FIX: Cap total penalty at -15 (was unbounded, could reach -35) ---
        confidence_mod = max(confidence_mod, -15)

        return {
            "regime": regime,
            "confidence_mod": confidence_mod,
            "flat_threshold": flat_threshold,
            "adjustments": adjustments,
            "summary": " | ".join(adjustments) if adjustments else "Standard regime.",
        }

    except Exception as e:
        logger.warning(f"Regime adjustment failed: {e}", exc_info=True)
        return {"regime": "N/A", "confidence_mod": 0, "flat_threshold": 60,
                "adjustments": [], "summary": "Error"}


# =================================================================
# --- Signal confluence detection — conviction floor ---
# =================================================================

def calc_signal_confluence(metrics: dict) -> dict:
    """
    NEW FUNCTION (v25.3): Detect when multiple independent signals agree.
    Returns a conviction floor that post-hoc penalties cannot breach.

    When 3+ independent signals agree on direction, that's a genuine
    high-probability setup. Regime penalties should reduce size, not
    prevent the trade entirely.
    """
    try:
        direction_votes = {"BULLISH": 0, "BEARISH": 0}
        signals_checked = 0
        confirming = []

        # 1. Fractal projection (strongest single signal — double weight)
        fractal = metrics.get("fractal", {})
        proj = fractal.get("projection", None)
        if proj and proj.confidence >= 65:
            signals_checked += 1
            # Scale weight 1.0-2.0 by confidence (65%→1.0, 100%→2.0)
            fractal_weight = round(1.0 + min(1.0, (proj.confidence - 50) / 50), 2)
            if "BULL" in proj.direction:
                direction_votes["BULLISH"] += fractal_weight
                confirming.append(f"Fractal {proj.direction} ({proj.confidence}%, wt={fractal_weight})")
            elif "BEAR" in proj.direction:
                direction_votes["BEARISH"] += fractal_weight
                confirming.append(f"Fractal {proj.direction} ({proj.confidence}%, wt={fractal_weight})")

        # 2. Options flow bias
        flow = metrics.get("flow_data", {})
        # FlowScanner may return key as "flow_bias" or "bias" — check both
        flow_bias = (flow.get("flow_bias") or flow.get("bias") or "").upper()
        if "BULLISH" in flow_bias or "CALL" in flow_bias:
            direction_votes["BULLISH"] += 1
            confirming.append(f"Flow: {flow_bias}")
            signals_checked += 1
        elif "BEARISH" in flow_bias or "PUT" in flow_bias:
            direction_votes["BEARISH"] += 1
            confirming.append(f"Flow: {flow_bias}")
            signals_checked += 1

        # 3. Liquidity sweep (high-probability reversal signal)
        sweeps = metrics.get("liq_sweeps", {})
        active_sweep = sweeps.get("active_sweep", "NONE")
        if active_sweep == "BEAR TRAP":
            direction_votes["BULLISH"] += 1
            confirming.append("Bear Trap (sweep reversal)")
            signals_checked += 1
        elif active_sweep == "BULL TRAP":
            direction_votes["BEARISH"] += 1
            confirming.append("Bull Trap (sweep reversal)")
            signals_checked += 1

        # 4. MTF momentum alignment
        mtf = metrics.get("mtf_momentum", {})
        mtf_align = mtf.get("alignment", "")
        if "FULL BULLISH" in mtf_align or "MOSTLY BULLISH" in mtf_align:
            direction_votes["BULLISH"] += 1
            confirming.append(f"MTF: {mtf_align}")
            signals_checked += 1
        elif "FULL BEARISH" in mtf_align or "MOSTLY BEARISH" in mtf_align:
            direction_votes["BEARISH"] += 1
            confirming.append(f"MTF: {mtf_align}")
            signals_checked += 1

        # 5. TICK proxy (institutional breadth)
        tick = metrics.get("tick_proxy", {})
        tick_extreme = tick.get("extreme", "")
        if "BULLISH" in tick_extreme:
            direction_votes["BULLISH"] += 1
            confirming.append(f"TICK: {tick_extreme}")
            signals_checked += 1
        elif "BEARISH" in tick_extreme:
            direction_votes["BEARISH"] += 1
            confirming.append(f"TICK: {tick_extreme}")
            signals_checked += 1

        # 6. Cumulative delta
        delta_bias = str(metrics.get("cum_delta_bias", ""))
        if "NET BUYERS" in delta_bias and "RISING" in delta_bias:
            direction_votes["BULLISH"] += 1
            confirming.append("Delta: Net Buyers Rising")
            signals_checked += 1
        elif "NET SELLERS" in delta_bias and "FALLING" in delta_bias:
            direction_votes["BEARISH"] += 1
            confirming.append("Delta: Net Sellers Falling")
            signals_checked += 1

        # 7. Opening type bias
        opening = metrics.get("opening_type", {})
        open_bias = opening.get("bias", "").upper()
        if open_bias in ("BULLISH", "STRONGLY BULLISH"):
            direction_votes["BULLISH"] += 1
            confirming.append(f"Open: {opening.get('type', '')} ({open_bias})")
            signals_checked += 1
        elif open_bias in ("BEARISH", "STRONGLY BEARISH"):
            direction_votes["BEARISH"] += 1
            confirming.append(f"Open: {opening.get('type', '')} ({open_bias})")
            signals_checked += 1

        # 8. Delta at key levels (order flow at structure)
        dal = metrics.get("delta_levels", {})
        dal_bias = dal.get("net_bias", "NEUTRAL")
        dal_score = abs(dal.get("bias_score", 0))
        if dal_bias == "BULLISH" and dal_score >= 20:
            # Scale weight by bias strength: 20-50 → 1.0, 50-100 → up to 1.5
            dal_wt = round(min(1.5, 1.0 + (dal_score - 20) / 160), 2)
            direction_votes["BULLISH"] += dal_wt
            confirming.append(f"Delta@Levels: {dal_bias} ({dal_score}%, wt={dal_wt})")
            signals_checked += 1
        elif dal_bias == "BEARISH" and dal_score >= 20:
            dal_wt = round(min(1.5, 1.0 + (dal_score - 20) / 160), 2)
            direction_votes["BEARISH"] += dal_wt
            confirming.append(f"Delta@Levels: {dal_bias} ({dal_score}%, wt={dal_wt})")
            signals_checked += 1

        # --- Determine confluence ---
        bull_score = direction_votes["BULLISH"]
        bear_score = direction_votes["BEARISH"]
        dominant_dir = "BULLISH" if bull_score > bear_score else "BEARISH" if bear_score > bull_score else "NEUTRAL"
        dominant_score = max(bull_score, bear_score)

        if dominant_score >= 5:
            conviction_floor = 75
            confluence_level = "STRONG"
        elif dominant_score >= 4:
            conviction_floor = 68
            confluence_level = "MODERATE"
        elif dominant_score >= 3:
            conviction_floor = 63
            confluence_level = "WEAK"
        else:
            conviction_floor = 0
            confluence_level = "NONE"

        return {
            "direction": dominant_dir,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "signals_checked": signals_checked,
            "confirming": confirming,
            "conviction_floor": conviction_floor,
            "confluence_level": confluence_level,
            "summary": (
                f"{confluence_level} confluence: {dominant_score} signals -> "
                f"{dominant_dir} (floor: {conviction_floor}%)"
                if dominant_score >= 3
                else f"Low confluence ({dominant_score} signals). No floor protection."
            ),
        }

    except Exception as e:
        logger.warning(f"calc_signal_confluence error: {e}")
        return {
            "direction": "NEUTRAL", "bull_score": 0, "bear_score": 0,
            "signals_checked": 0, "confirming": [], "conviction_floor": 0,
            "confluence_level": "NONE", "summary": f"Error: {e}",
        }


# =================================================================
# --- PATCHED CONFIDENCE PIPELINE (replaces inline code in main) ---
# =================================================================

def apply_confidence_pipeline(data, metrics, accuracy_tracker, regime, md,
                              position_suggestion_fn=None, news_info=None):
    """
    FIXED confidence pipeline (v25.3). Replaces the inline code in main().

    Old: raw -> accuracy(-15) -> regime(-35) -> ratcheted threshold(75) -> FLAT
    New: raw -> accuracy(-10,decayed) -> regime(-15,capped) -> floor -> stable threshold -> verdict
    Max penalty: -25 (was -70). Floor protects confirmed multi-signal setups.
    """
    raw_conf = int(data.get("confidence", 0))
    verdict = data.get("verdict", "FLAT").upper()
    original_verdict = verdict  # preserve for decomposition logging

    # Step 1: Accuracy adjustment (capped at -10, decays with staleness)
    conf = accuracy_tracker.adjust_confidence(raw_conf)
    acc_adj = conf - raw_conf

    # Step 2: Regime adjustment (capped at -15)
    regime_mod = regime.get("confidence_mod", 0)
    if regime_mod != 0:
        conf = max(0, min(100, conf + regime_mod))

    # Step 3: Signal confluence floor (NEW — the key fix)
    confluence = calc_signal_confluence(metrics)
    floor = confluence.get("conviction_floor", 0)

    # Step 4: Regime flat threshold — must be defined before floor threshold-lift check
    flat_threshold = regime.get("flat_threshold", 60)

    floor_applied = False
    confluence_override = False  # track if we flipped FLAT → directional

    # Step 3a: CONFLUENCE OVERRIDE — if Claude returned FLAT but 4+ signals
    # strongly agree on a direction, override to that direction.
    # This prevents the AI from perpetually punting when fractal conflicts.
    #
    # Regime gate:
    #   HIGH VOL (flat_threshold >= 65): NEVER override FLAT — respect caution
    #   ELEVATED (flat_threshold >= 62): Require STRONG confluence (5+ signals)
    #   NORMAL/LOW (< 62):              MODERATE or STRONG (4+ signals)
    if verdict in ("FLAT", "NEUTRAL") and floor > 0:
        confl_level = confluence.get("confluence_level", "NONE")
        confl_dir = confluence.get("direction", "NEUTRAL")

        # Regime-aware quality gate
        _override_allowed = False
        if flat_threshold >= 65:
            # HIGH VOL — don't override FLAT at all
            if confl_level in ("STRONG", "MODERATE") and confl_dir != "NEUTRAL":
                logger.info(
                    f"CONFLUENCE OVERRIDE BLOCKED (HIGH VOL regime, flat_threshold={flat_threshold}) | "
                    f"{confl_level} {confl_dir} ({len(confluence.get('confirming', []))} signals)"
                )
        elif flat_threshold >= 62:
            # ELEVATED — require STRONG only
            _override_allowed = (confl_level == "STRONG" and confl_dir != "NEUTRAL")
            if not _override_allowed and confl_level == "MODERATE" and confl_dir != "NEUTRAL":
                logger.info(
                    f"CONFLUENCE OVERRIDE BLOCKED (ELEVATED regime) — need STRONG, got MODERATE | "
                    f"{confl_dir} ({len(confluence.get('confirming', []))} signals)"
                )
        else:
            # NORMAL / LOW — allow MODERATE or STRONG
            _override_allowed = (confl_level in ("STRONG", "MODERATE") and confl_dir != "NEUTRAL")

        if _override_allowed:
            old_verdict = verdict
            old_conf = conf
            verdict = f"LEAN {confl_dir}"
            data["verdict"] = verdict
            conf = max(conf, floor)
            confluence_override = True
            floor_applied = True
            logger.info(
                f"CONFLUENCE OVERRIDE: {old_verdict} {old_conf}% -> {verdict} {conf}% | "
                f"{confl_level} confluence ({len(confluence['confirming'])} signals: "
                f"{', '.join(confluence['confirming'])})"
            )

    # Step 3b: For directional verdicts, apply floor if direction agrees
    if floor > 0 and not confluence_override and verdict not in ("FLAT", "NEUTRAL"):
        # Broaden direction check — handle BULLISH / LONG / BUY and BEARISH / SHORT / SELL
        verdict_bull = any(k in verdict for k in ("BULL", "LONG", "BUY"))
        verdict_bear = any(k in verdict for k in ("BEAR", "SHORT", "SELL"))
        confluence_bull = confluence["direction"] == "BULLISH"
        confluence_bear = confluence["direction"] == "BEARISH"

        direction_agrees = (verdict_bull and confluence_bull) or (verdict_bear and confluence_bear)

        if direction_agrees and conf < floor:
            old_conf = conf
            conf = floor
            floor_applied = True

        # FIX: If floor was applied but conf is STILL below flat_threshold
        # (happens in high-VIX regimes where threshold > floor), lift to threshold
        # so confluence setups with 3+ agreeing signals are never silently FLATted.
        if floor_applied and conf < flat_threshold:
            confluence_level = confluence.get("confluence_level", "NONE")
            if confluence_level in ("STRONG", "MODERATE", "WEAK"):
                conf = flat_threshold
                logger.info(
                    f"CONFLUENCE THRESHOLD LIFT: {confluence_level} setup — "
                    f"floor {floor}% < threshold {flat_threshold}%, lifting to threshold."
                )

    # Step 5: Position sizing (uses regime threshold, not hardcoded 70)
    if position_suggestion_fn is None:
        from session_utils import position_suggestion
        position_suggestion_fn = position_suggestion
    contracts, pos_str = position_suggestion_fn(conf, flat_threshold)

    # Step 5b: Event-aware position sizing — halve contracts near high-impact news
    if news_info:
        _approaching, _event_str, _event_dt, _impact = news_info
        if _approaching and _impact == "high" and _event_dt:
            from bot_config import now_et as _now_et
            _minutes_until = (_event_dt - _now_et()).total_seconds() / 60
            if 0 < _minutes_until <= 30 and contracts > 1:
                _orig = contracts
                contracts = max(1, contracts // 2)
                pos_str = f"{contracts} ct (NEWS HALVED from {_orig})"
                logger.info(
                    f"NEWS POSITION REDUCTION: {_event_str} in {_minutes_until:.0f}min "
                    f"-- contracts {_orig} -> {contracts}"
                )

    # Step 6a: DIRECTIONAL CONFLICT GUARD (Fix #1 — 2026-03-06 loss review)
    # If MTF is fully opposed to verdict AND divergence is HIGH+, force FLAT.
    # This prevents buying into a falling knife when every timeframe is bearish.
    mtf_align = metrics.get("mtf_momentum", {}).get("alignment", "")
    div_severity = metrics.get("divergence", {}).get("severity", "")
    div_high = any(k in div_severity for k in ("HIGH", "EXTREME"))
    verdict_bull = any(k in verdict for k in ("BULL", "LONG", "BUY"))
    verdict_bear = any(k in verdict for k in ("BEAR", "SHORT", "SELL"))

    mtf_conflict = False
    if verdict_bull and "FULL BEARISH" in mtf_align and div_high:
        mtf_conflict = True
    elif verdict_bear and "FULL BULLISH" in mtf_align and div_high:
        mtf_conflict = True

    if mtf_conflict and verdict not in ("FLAT", "NEUTRAL"):
        logger.info(
            f"DIRECTIONAL CONFLICT GUARD: {verdict} blocked -- "
            f"MTF is {mtf_align} + divergence {div_severity}. Forcing FLAT."
        )
        data["setup"] = (
            f"Blocked by conflict guard: {verdict} vs {mtf_align} + {div_severity}"
        )
        verdict = "FLAT"
        data["verdict"] = "FLAT"

    # Step 6b: FRACTAL CONTRADICTION GUARD (Fix #3 — 2026-03-06 loss review)
    # If verdict opposes fractal direction, require MODERATE+ confluence (4+ signals)
    # to proceed. Otherwise force FLAT. Prevents chasing momentum against fractal.
    _fractal_proj = metrics.get("fractal", {}).get("projection", None)
    if verdict not in ("FLAT", "NEUTRAL") and _fractal_proj:
        fractal_dir = getattr(_fractal_proj, "direction", "")
        fractal_conf_val = getattr(_fractal_proj, "confidence", 0)
        frac_bull = "BULL" in fractal_dir
        frac_bear = "BEAR" in fractal_dir

        contradicts_fractal = (
            (verdict_bull and frac_bear and fractal_conf_val >= 60) or
            (verdict_bear and frac_bull and fractal_conf_val >= 60)
        )
        if contradicts_fractal:
            confl_level = confluence.get("confluence_level", "NONE")
            if confl_level not in ("STRONG", "MODERATE"):
                logger.info(
                    f"FRACTAL CONTRADICTION GUARD: {verdict} contradicts "
                    f"Fractal {fractal_dir} ({fractal_conf_val}%) with only "
                    f"{confl_level} confluence. Forcing FLAT."
                )
                data["setup"] = (
                    f"Blocked: {verdict} contradicts Fractal {fractal_dir} "
                    f"({fractal_conf_val}%) -- need MODERATE+ confluence, got {confl_level}"
                )
                verdict = "FLAT"
                data["verdict"] = "FLAT"

    # Step 6c: FLAT override (only if below threshold AND no confluence floor)
    forced_flat = False
    if verdict not in ("FLAT", "NEUTRAL") and conf < flat_threshold and not floor_applied:
        forced_flat = True
        verdict = "FLAT"
        data["verdict"] = "FLAT"
        data["setup"] = (
            f"Forced FLAT: {conf}% < {flat_threshold}% threshold "
            f"(raw {raw_conf}, acc {acc_adj:+d}, regime {regime_mod:+d})"
        )

    if floor_applied:
        logger.info(
            f"CONFLUENCE FLOOR: {confluence['confluence_level']} -- "
            f"{len(confluence['confirming'])} signals: {', '.join(confluence['confirming'])}. "
            f"Conf raised {old_conf}% -> {conf}% (floor {floor}%)"
        )
    if forced_flat:
        logger.info(
            f"Forced FLAT: {conf}% below {flat_threshold}% "
            f"(raw {raw_conf} + acc {acc_adj:+d} + regime {regime_mod:+d})"
        )
    elif acc_adj != 0 or regime_mod != 0:
        logger.info(
            f"Confidence: {raw_conf} -> {conf} "
            f"(acc {acc_adj:+d}, regime {regime_mod:+d}"
            f"{f', floor {floor}' if floor_applied else ''})"
        )

    data["confidence"] = conf

    decomposition = {
        "raw_verdict": original_verdict,
        "raw_confidence": raw_conf,
        "accuracy_adj": acc_adj,
        "regime_adj": regime_mod,
        "confluence_level": confluence.get("confluence_level", "NONE"),
        "floor_applied": floor_applied,
        "confluence_override": confluence_override,
        "forced_flat": forced_flat,
        "mtf_conflict": mtf_conflict,
    }
    return data, verdict, conf, contracts, pos_str, confluence, decomposition
