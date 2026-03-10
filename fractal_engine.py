"""
FRACTAL ENGINE v2.2 -- Advanced Intraday Pattern Recognition for ES Futures
===========================================================================
Think of this like facial recognition, but for trading days.

v2.2 additions: rvol_bucket wired into ContextFilter, open_type context filter
(Drive vs Auction), TNX trend cross-asset confirmation.

v2.1 fixes: was_correct outcome tracking, match count confidence scaling,
VIX bucket granularity (split at 25), NQ cross-instrument confirmation.

TIER 1 (Massive impact):
  - Bulk historical data loader (IBKR backfill -> 500-2000+ days)
  - Context pre-filter (VIX bucket, gap direction, RVOL, day-of-week,
    opening type, TNX trend)
  - Recency-weighted bar scoring (later bars count more)

TIER 2 (Strong improvement):
  - Candle microstructure features (wicks, body ratios, bar sequences)
  - Outcome-weighted scoring (learn from past prediction accuracy)
  - Multi-resolution matching (1m + 5m + 15m + NQ simultaneously)

TIER 3 (Refinement):
  - Segmented matching (open drive, AM trend, lunch, PM, close)
  - Volatility-adjusted normalization (ATR-based instead of min-max)

Same API as v1 -- drop-in replacement.
"""

import logging
import sqlite3
import json
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, time as dtime, date as dateclass
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from bot_config import now_et, CFG

logger = logging.getLogger("MarketBot")


# =================================================================
# --- DTW (Dynamic Time Warping) — Optimized NumPy ---
# =================================================================

def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """DTW distance with Sakoe-Chiba band constraint for speed."""
    n, m = len(s1), len(s2)
    window = max(10, abs(n - m) + 5)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            d = (s1[i - 1] - s2[j - 1]) ** 2
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return np.sqrt(cost[n, m]) / max(n, m)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 for constant arrays instead of NaN."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    try:
        c = np.corrcoef(a, b)[0, 1]
        return 0.0 if np.isnan(c) else float(c)
    except Exception:
        return 0.0


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation. Returns 0.0 for degenerate cases."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        wm_a = np.average(a, weights=w)
        wm_b = np.average(b, weights=w)
        cov = np.average((a - wm_a) * (b - wm_b), weights=w)
        std_a = np.sqrt(np.average((a - wm_a) ** 2, weights=w))
        std_b = np.sqrt(np.average((b - wm_b) ** 2, weights=w))
        if std_a < 1e-12 or std_b < 1e-12:
            return 0.0
        c = cov / (std_a * std_b)
        return 0.0 if np.isnan(c) else float(c)
    except Exception:
        return 0.0


# =================================================================
# --- SHARED HELPERS ---
# =================================================================

def _rvol_bucket(vol_total: float, avg_volume: float) -> str:
    """Classify relative volume into a bucket."""
    if avg_volume <= 0:
        return "NORMAL"
    rv = vol_total / avg_volume
    if rv < 0.5:   return "VERY_LOW"
    if rv < 0.8:   return "LOW"
    if rv < 1.5:   return "NORMAL"
    if rv < 2.0:   return "HIGH"
    return "VERY_HIGH"


def _classify_open_type(open_type: str) -> str:
    """Map a free-text open_type string to an internal bucket."""
    if not open_type:
        return "AUCTION"
    ot = str(open_type).upper()
    if "DRIVE" in ot and ("UP" in ot or "BUY" in ot or "BULL" in ot):
        return "DRIVE_UP"
    if "DRIVE" in ot and ("DOWN" in ot or "SELL" in ot or "BEAR" in ot):
        return "DRIVE_DOWN"
    if "DRIVE" in ot:
        return "DRIVE_UP"  # Generic drive — default up
    return "AUCTION"


# =================================================================
# --- SESSION FILTER ---
# =================================================================

_DAY_START = dtime(3, 0)
_DAY_END   = dtime(16, 0)
_NIGHT_START = dtime(18, 0)
_NIGHT_END   = dtime(3, 0)


def _filter_bars_by_session(df, session_type):
    """Filter a DataFrame of bars to DAY (3AM-4PM) or NIGHT (6PM-3AM) session.

    Returns a copy with reset integer index so bar_idx logic still works.
    Returns original df unchanged if session_type is None.
    """
    if session_type is None or df.empty:
        return df
    idx = df.index
    if hasattr(idx, 'time'):
        times = idx.time
    elif 'timestamp' in df.columns:
        times = pd.to_datetime(df['timestamp']).dt.time
    else:
        return df  # Can't determine times

    if session_type == "DAY":
        mask = np.array([(t >= _DAY_START and t < _DAY_END) for t in times])
    elif session_type == "NIGHT":
        mask = np.array([(t >= _NIGHT_START or t < _NIGHT_END) for t in times])
    else:
        return df

    filtered = df[mask]
    if filtered.empty:
        return filtered
    # Reset to integer index so bar_idx-based logic (segments, etc.) still works
    filtered = filtered.copy()
    filtered.index = pd.RangeIndex(len(filtered))
    return filtered


# =================================================================
# --- DAY SIGNATURE v2: Enhanced Fingerprint ---
# =================================================================

@dataclass
class DaySignature:
    """All features extracted from one trading day's price action."""
    date: object
    price_shape: np.ndarray
    volume_shape: np.ndarray
    range_expansion: np.ndarray
    momentum_curve: np.ndarray
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    day_range: float
    day_return_pct: float
    bar_count: int
    volume_total: float
    volatility: float
    full_closes: np.ndarray
    full_highs: np.ndarray
    full_lows: np.ndarray
    full_volumes: np.ndarray
    full_times: np.ndarray
    # v2 Context
    vix_at_open: float = 0.0
    gap_pct: float = 0.0
    day_of_week: int = 0
    rvol_bucket: str = "NORMAL"
    open_type_bucket: str = "AUCTION"   # DRIVE_UP | DRIVE_DOWN | AUCTION
    tnx_trend: str = "NEUTRAL"          # UP | DOWN | NEUTRAL
    # v2 Microstructure
    wick_ratios: np.ndarray = field(default_factory=lambda: np.array([]))
    body_directions: np.ndarray = field(default_factory=lambda: np.array([]))
    bar_range_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    # v2 ATR-normalized
    atr_norm_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    # Segment shapes
    segment_shapes: Dict[str, np.ndarray] = field(default_factory=dict)


SEGMENTS = {
    "open_drive": (0, 6),
    "am_trend":   (6, 24),
    "lunch":      (24, 42),
    "pm_trend":   (42, 66),
    "close":      (66, 78),
}


def extract_signature(day_df: pd.DataFrame, max_bars: int = None,
                      prev_close: float = 0.0, vix_open: float = 0.0,
                      avg_volume: float = 0.0) -> Optional[DaySignature]:
    """Extract enhanced fingerprint from one day's 5-min OHLCV data."""
    if day_df.empty or len(day_df) < 6:
        return None

    df = day_df.iloc[:max_bars] if max_bars else day_df
    full_df = day_df

    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    opens_arr = df["Open"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    c_min, c_max = closes.min(), closes.max()
    if c_max == c_min:
        return None

    # v1 features
    price_shape = (closes - c_min) / (c_max - c_min)
    v_max = volumes.max()
    volume_shape = volumes / v_max if v_max > 0 else np.zeros_like(volumes)
    running_high = np.maximum.accumulate(highs)
    running_low = np.minimum.accumulate(lows)
    running_range = running_high - running_low
    final_range = running_range[-1]
    range_expansion = running_range / final_range if final_range > 0 else np.zeros_like(running_range)
    first_open = opens_arr[0]
    momentum_curve = (closes - first_open) / first_open * 100 if first_open > 0 else np.zeros_like(closes)

    # TIER 2: Candle microstructure
    bar_ranges = highs - lows
    bodies = np.abs(closes - opens_arr)
    upper_wicks = highs - np.maximum(closes, opens_arr)
    lower_wicks = np.minimum(closes, opens_arr) - lows
    safe_ranges = np.where(bar_ranges > 0, bar_ranges, 1.0)
    wick_ratios = (upper_wicks - lower_wicks) / safe_ranges
    body_threshold = bar_ranges * 0.1
    body_directions = np.where(bodies < body_threshold, 0,
                               np.where(closes > opens_arr, 1, -1)).astype(float)
    avg_bar_range = bar_ranges.mean() if bar_ranges.mean() > 0 else 1.0
    bar_range_curve = bar_ranges / avg_bar_range

    # TIER 3: ATR-normalized price curve
    atr_14 = pd.Series(bar_ranges).rolling(14, min_periods=1).mean().values
    atr_14 = np.where(atr_14 > 0, atr_14, 1.0)
    atr_norm_curve = (closes - first_open) / atr_14

    # TIER 3: Segment shapes
    segment_shapes = {}
    for seg_name, (start, end) in SEGMENTS.items():
        actual_end = min(end, len(closes))
        if actual_end > start and actual_end - start >= 3:
            seg_closes = closes[start:actual_end]
            seg_min, seg_max = seg_closes.min(), seg_closes.max()
            if np.isfinite(seg_min) and np.isfinite(seg_max) and seg_max > seg_min:
                segment_shapes[seg_name] = (seg_closes - seg_min) / (seg_max - seg_min)
            else:
                segment_shapes[seg_name] = np.zeros(actual_end - start)

    # Context
    # Opening type: classify first 6 bars (30 min) as drive vs auction
    open_type_bucket = "AUCTION"
    if len(closes) >= 6:
        first_6_high = highs[:6].max()
        first_6_low  = lows[:6].min()
        first_6_range = first_6_high - first_6_low
        if first_6_range > 0:
            first_close_rel = (closes[5] - closes[0]) / first_6_range
            # Drive: price moves >70% of early range in one direction without reversal
            reversals = sum(
                1 for i in range(1, 6)
                if (closes[i] - closes[i-1]) * (closes[5] - closes[0]) < 0
            )
            if first_close_rel > 0.65 and reversals <= 1:
                open_type_bucket = "DRIVE_UP"
            elif first_close_rel < -0.65 and reversals <= 1:
                open_type_bucket = "DRIVE_DOWN"

    gap_pct = ((first_open - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
    day_date = day_df.index[0].date() if hasattr(day_df.index[0], 'date') else day_df.index[0]
    dow = day_date.weekday() if isinstance(day_date, dateclass) else 0
    vol_total = float(volumes.sum())
    rvol_bucket = _rvol_bucket(vol_total, avg_volume)

    full_closes = full_df["Close"].values.astype(float)
    full_highs = full_df["High"].values.astype(float)
    full_lows = full_df["Low"].values.astype(float)
    full_volumes = full_df["Volume"].values.astype(float)
    try:
        times = np.array([(t - full_df.index[0]).total_seconds() / 60 for t in full_df.index])
    except Exception:
        times = np.arange(len(full_df)) * 5.0

    return DaySignature(
        date=day_date, price_shape=price_shape, volume_shape=volume_shape,
        range_expansion=range_expansion, momentum_curve=momentum_curve,
        open_price=float(first_open), close_price=float(full_closes[-1]),
        high_price=float(full_highs.max()), low_price=float(full_lows.min()),
        day_range=float(full_highs.max() - full_lows.min()),
        day_return_pct=float((full_closes[-1] - first_open) / first_open * 100) if first_open > 0 else 0.0,
        bar_count=len(full_df), volume_total=vol_total,
        volatility=float((highs.max() - lows.min()) / closes[-1]) if closes[-1] > 0 else 0.0,
        full_closes=full_closes, full_highs=full_highs, full_lows=full_lows,
        full_volumes=full_volumes, full_times=times,
        vix_at_open=vix_open, gap_pct=gap_pct, day_of_week=dow,
        rvol_bucket=rvol_bucket, open_type_bucket=open_type_bucket,
        wick_ratios=wick_ratios,
        body_directions=body_directions, bar_range_curve=bar_range_curve,
        atr_norm_curve=atr_norm_curve, segment_shapes=segment_shapes,
    )


# =================================================================
# --- CONTEXT PRE-FILTER (TIER 1) ---
# =================================================================

class ContextFilter:
    @staticmethod
    def filter(today: DaySignature, candidates: list, strict: bool = False) -> list:
        filtered = []
        today_vix_b  = _vix_bucket(today.vix_at_open)
        today_gap_b  = _gap_bucket(today.gap_pct)
        today_dow    = today.day_of_week
        today_rvol   = today.rvol_bucket
        today_otype  = today.open_type_bucket
        today_tnx    = today.tnx_trend

        for date_str, day_df, hist_context in candidates:
            h_vix   = hist_context.get("vix", 0)
            h_gap   = hist_context.get("gap_pct", 0)
            h_dow   = hist_context.get("dow", 0)
            h_rvol  = hist_context.get("rvol_bucket", "NORMAL")
            h_otype = hist_context.get("open_type", "AUCTION")
            h_tnx   = hist_context.get("tnx_trend", "NEUTRAL")
            bonus = 0.0
            passes = True

            # VIX regime match — always reject wildly different regimes (3+ buckets apart)
            h_vix_b = _vix_bucket(h_vix)
            if today.vix_at_open > 0 and h_vix > 0:
                vix_gap = abs(h_vix_b - today_vix_b)
                if h_vix_b == today_vix_b: bonus += 0.04
                elif vix_gap == 1: bonus += 0.01
                elif vix_gap >= 3: passes = False   # e.g. VIX 14 vs VIX 35 — different universe
                elif strict: passes = False          # 2-bucket gap only rejected in strict mode

            # Gap direction match
            h_gap_b = _gap_bucket(h_gap)
            if h_gap_b == today_gap_b: bonus += 0.03
            elif np.sign(h_gap_b) == np.sign(today_gap_b) and h_gap_b != 0: bonus += 0.01

            # Day-of-week match
            if h_dow == today_dow: bonus += 0.02
            elif today_dow in (0, 4) and h_dow in (0, 4): bonus += 0.01

            # RVOL regime match (was computed but never used — now wired in)
            _rvol_order = {"VERY_LOW": 0, "LOW": 1, "NORMAL": 2, "HIGH": 3, "VERY_HIGH": 4}
            if h_rvol == today_rvol:
                bonus += 0.02  # Same volume environment = same participant behavior
            elif h_rvol in _rvol_order and today_rvol in _rvol_order and \
                    abs(_rvol_order[h_rvol] - _rvol_order[today_rvol]) == 1:
                bonus += 0.01  # Adjacent bucket

            # Opening type match: an Open Drive day matches best against other Drive days
            if h_otype == today_otype:
                bonus += 0.03  # Same session structure = most predictive forward path
            elif (h_otype in ("DRIVE_UP","DRIVE_DOWN") and
                  today_otype in ("DRIVE_UP","DRIVE_DOWN")):
                bonus += 0.01  # Both are drive days, even if direction differs

            # TNX trend match: same bond market regime adds cross-asset confirmation
            if today_tnx != "NEUTRAL" and h_tnx != "NEUTRAL":
                if h_tnx == today_tnx: bonus += 0.02

            if passes:
                filtered.append((date_str, day_df, bonus))
        return filtered

def _vix_bucket(vix):
    # Returns -1 for unknown (guards in ContextFilter prevent -1 from scoring)
    if vix <= 0: return -1
    if vix < 15: return 0
    if vix < 20: return 1
    if vix < 25: return 2   # Elevated
    if vix < 32: return 3   # High (split from old single 20-30 bucket)
    return 4                # Crisis

def _gap_bucket(gap_pct):
    if gap_pct < -0.3: return -2
    if gap_pct < -0.05: return -1
    if gap_pct < 0.05: return 0
    if gap_pct < 0.3: return 1
    return 2


# =================================================================
# --- SIMILARITY SCORING v2 ---
# =================================================================

def score_similarity(today, hist, hist_partial, outcome_bonus=0.0, context_bonus=0.0):
    scores = {}
    n_bars = min(len(today.price_shape), len(hist_partial.price_shape))
    if n_bars < 6:
        return {"composite": 0.0}

    # Recency weights (TIER 1)
    recency = np.exp(np.linspace(0, 1.1, n_bars))
    recency /= recency.sum()

    # 1. Price Shape (recency-weighted)
    t_p = today.price_shape[:n_bars]
    h_p = hist_partial.price_shape[:n_bars]
    scores["price_corr"] = max(0, _weighted_corr(t_p, h_p, recency))

    # 2. DTW
    try:
        scores["dtw"] = max(0, 1.0 - dtw_distance(t_p, h_p) * 2)
    except Exception:
        scores["dtw"] = 0.0

    # 3. Momentum (recency-weighted)
    n_m = min(len(today.momentum_curve), len(hist_partial.momentum_curve))
    if n_m >= 6:
        rec_m = np.exp(np.linspace(0, 1.1, n_m)); rec_m /= rec_m.sum()
        scores["momentum"] = max(0, _weighted_corr(
            today.momentum_curve[:n_m], hist_partial.momentum_curve[:n_m], rec_m))
    else:
        scores["momentum"] = 0.0

    # 4. Volume Shape
    n_v = min(len(today.volume_shape), len(hist_partial.volume_shape))
    if n_v >= 6:
        scores["vol_shape"] = max(0, _safe_corr(today.volume_shape[:n_v], hist_partial.volume_shape[:n_v]))
    else: scores["vol_shape"] = 0.0

    # 5. Range Expansion
    n_r = min(len(today.range_expansion), len(hist_partial.range_expansion))
    if n_r >= 6:
        scores["range_pattern"] = max(0, _safe_corr(today.range_expansion[:n_r], hist_partial.range_expansion[:n_r]))
    else: scores["range_pattern"] = 0.0

    # 6. Volatility Match
    if today.volatility > 0 and np.isfinite(today.volatility) and np.isfinite(hist_partial.volatility):
        scores["vol_match"] = max(0, 1.0 - abs(today.volatility - hist_partial.volatility) / today.volatility)
    else:
        scores["vol_match"] = 0.5

    # 7. TIER 2: Wick Pattern
    n_w = min(len(today.wick_ratios), len(hist_partial.wick_ratios))
    if n_w >= 6:
        scores["wick_pattern"] = max(0, _safe_corr(today.wick_ratios[:n_w], hist_partial.wick_ratios[:n_w]))
    else: scores["wick_pattern"] = 0.0

    # 8. TIER 2: Body Direction Sequence
    n_b = min(len(today.body_directions), len(hist_partial.body_directions))
    if n_b >= 6:
        scores["body_sequence"] = float(np.mean(today.body_directions[:n_b] == hist_partial.body_directions[:n_b]))
    else:
        scores["body_sequence"] = 0.0

    # 9. TIER 2: Bar Range Curve
    n_br = min(len(today.bar_range_curve), len(hist_partial.bar_range_curve))
    if n_br >= 6:
        scores["bar_range_match"] = max(0, _safe_corr(today.bar_range_curve[:n_br], hist_partial.bar_range_curve[:n_br]))
    else: scores["bar_range_match"] = 0.0

    # 10. TIER 3: ATR-Normalized Curve
    n_a = min(len(today.atr_norm_curve), len(hist_partial.atr_norm_curve))
    if n_a >= 6:
        scores["atr_norm"] = max(0, _safe_corr(today.atr_norm_curve[:n_a], hist_partial.atr_norm_curve[:n_a]))
    else: scores["atr_norm"] = 0.0

    # 11. TIER 3: Segment Matching
    _seg_weights = {"open_drive": 1.0, "am_trend": 1.2, "lunch": 0.8, "pm_trend": 1.5, "close": 2.0}
    seg_scores = []
    seg_w = []
    for seg_name in SEGMENTS:
        t_seg = today.segment_shapes.get(seg_name)
        h_seg = hist_partial.segment_shapes.get(seg_name)
        if t_seg is not None and h_seg is not None:
            n_s = min(len(t_seg), len(h_seg))
            if n_s >= 3:
                seg_scores.append(max(0, _safe_corr(t_seg[:n_s], h_seg[:n_s])))
                seg_w.append(_seg_weights.get(seg_name, 1.0))

    if seg_scores:
        scores["segment_match"] = float(np.average(seg_scores, weights=seg_w))
    else:
        scores["segment_match"] = 0.0

    # Weighted Composite
    weights = {
        "price_corr": 0.18, "dtw": 0.15, "momentum": 0.12, "atr_norm": 0.10,
        "wick_pattern": 0.08, "body_sequence": 0.08, "bar_range_match": 0.06,
        "segment_match": 0.06, "vol_shape": 0.06, "range_pattern": 0.06, "vol_match": 0.05,
    }
    composite = sum(scores.get(k, 0) * w for k, w in weights.items())
    composite += context_bonus + outcome_bonus
    scores["composite"] = min(0.99, composite)
    scores["context_bonus"] = context_bonus
    scores["outcome_bonus"] = outcome_bonus
    return scores


# =================================================================
# --- MULTI-RESOLUTION MATCHER (TIER 2) ---
# =================================================================

class MultiResolutionMatcher:
    @staticmethod
    def score(today_5m, hist_5m, today_1m=None, hist_1m=None,
              today_15m=None, hist_15m=None, today_nq=None, hist_nq=None):
        bonus = 1.0

        # 15m ES agreement
        if today_15m is not None and hist_15m is not None:
            n = min(len(today_15m), len(hist_15m))
            if n >= 4:
                corr = _safe_corr(today_15m[:n], hist_15m[:n])
                if corr > 0.85: bonus += 0.08
                elif corr > 0.70: bonus += 0.04

        # 1m ES agreement
        if today_1m is not None and hist_1m is not None:
            n = min(len(today_1m), len(hist_1m))
            if n >= 20:
                step = max(1, n // 50)
                corr = _safe_corr(today_1m[:n:step], hist_1m[:n:step])
                if corr > 0.80: bonus += 0.07
                elif corr > 0.65: bonus += 0.03

        # NQ cross-confirmation: today's NQ intraday shape vs historical ES 15m shape.
        # ES and NQ are tightly correlated, so if today's NQ matches a historical ES
        # pattern the signal is confirmed across instruments — not just intraday noise.
        if today_nq is not None and hist_15m is not None:
            n = min(len(today_nq), len(hist_15m))
            if n >= 4:
                corr = _safe_corr(today_nq[:n], hist_15m[:n])
                if corr > 0.85: bonus += 0.06   # Strong cross-instrument confirmation
                elif corr > 0.70: bonus += 0.03

        return bonus


# =================================================================
# --- OUTCOME TRACKER (TIER 2) ---
# =================================================================

class OutcomeTracker:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fractal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_date TEXT NOT NULL, today_date TEXT NOT NULL,
                    predicted_dir TEXT, actual_dir TEXT,
                    composite_score REAL, was_correct INTEGER DEFAULT 0,
                    move_pts REAL DEFAULT 0.0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fo_match ON fractal_outcomes(match_date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fo_today ON fractal_outcomes(today_date)")
            conn.commit()

    def record_prediction(self, match_dates, predicted_dir, scores):
        today = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            for md, sc in zip(match_dates, scores):
                conn.execute(
                    "INSERT INTO fractal_outcomes (match_date, today_date, predicted_dir, composite_score) VALUES (?,?,?,?)",
                    (str(md), today, predicted_dir, sc))
            conn.commit()

    def record_actual(self, actual_dir, move_pts):
        """
        FIX v2.1: was_correct now compares actual_dir to the stored predicted_dir
        per row. Previous code checked (actual_dir != "FLAT") which marked any
        trending day as correct regardless of whether the prediction matched.
        """
        today = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, predicted_dir FROM fractal_outcomes "
                "WHERE today_date=? AND actual_dir IS NULL",
                (today,)).fetchall()
            for row in rows:
                was_correct = int(
                    actual_dir not in ("FLAT", "NEUTRAL") and
                    actual_dir == row["predicted_dir"]
                )
                conn.execute(
                    "UPDATE fractal_outcomes SET actual_dir=?, was_correct=?, move_pts=? WHERE id=?",
                    (actual_dir, was_correct, move_pts, row["id"]))
            conn.commit()

    def get_day_bonus(self, match_date):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT was_correct FROM fractal_outcomes WHERE match_date=? AND actual_dir IS NOT NULL",
                (str(match_date),)).fetchall()
        if len(rows) < 3: return 0.0
        rate = sum(r["was_correct"] for r in rows) / len(rows)
        if rate >= 0.80: return 0.05
        elif rate >= 0.70: return 0.03
        elif rate >= 0.60: return 0.01
        elif rate < 0.35: return -0.03
        return 0.0


# =================================================================
# --- FORWARD PROJECTION ---
# =================================================================

@dataclass
class ForwardProjection:
    direction: str; confidence: int; expected_close_vs_current: float
    projected_high: float; projected_low: float; upside_target: float
    downside_target: float; expected_close_price: float
    bullish_pct: float; bearish_pct: float; avg_move: float
    max_seen_up: float; max_seen_down: float; match_count: int; summary: str


def build_projection(matches, current_price, current_bar_idx):
    rc, rh, rl, wt = [], [], [], []
    today = now_et().date()
    for sig, scores in matches:
        if sig.bar_count <= current_bar_idx: continue
        fc = sig.full_closes[current_bar_idx:]
        fh = sig.full_highs[current_bar_idx:]
        fl = sig.full_lows[current_bar_idx:]
        if len(fc) < 2: continue
        mp = sig.full_closes[current_bar_idx - 1] if current_bar_idx > 0 else sig.full_closes[0]
        if mp <= 0: continue
        scale = current_price / mp if mp > 0 else 1.0
        rc.append((fc[-1] - mp) * scale)
        rh.append((fh.max() - mp) * scale)
        rl.append((fl.min() - mp) * scale)
        # Recency bonus: matches from last 6 months get up to 20% weight boost
        composite = scores["composite"]
        try:
            match_date = sig.date if isinstance(sig.date, dateclass) else datetime.strptime(str(sig.date), "%Y-%m-%d").date()
            days_ago = (today - match_date).days
            recency_mult = 1.0 + 0.2 * max(0, 1 - days_ago / 180)  # 1.2x for today, 1.0x at 6mo+
        except Exception:
            recency_mult = 1.0
        wt.append(composite * recency_mult)

    if len(rc) == 0:
        return ForwardProjection("NEUTRAL", 0, 0, 0, 0, current_price, current_price,
                                 current_price, 50, 50, 0, 0, 0, 0, "No data.")

    wt = np.array(wt)
    wt_total = wt.sum()
    if wt_total == 0:
        return ForwardProjection("NEUTRAL", 0, 0, 0, 0, current_price, current_price,
                                 current_price, 50, 50, 0, 0, 0, 0, "Zero composite scores.")
    best_raw_composite = float(wt.max())  # save before normalization
    wt /= wt_total
    rc, rh, rl = np.array(rc), np.array(rh), np.array(rl)
    ec = float(np.average(rc, weights=wt))
    eh = float(np.average(rh, weights=wt))
    el = float(np.average(rl, weights=wt))
    am = float(np.average(np.abs(rc), weights=wt))
    bc = (rc > 0).sum(); brc = (rc < 0).sum(); total = len(rc)
    bp = bc / total * 100; brp = brc / total * 100
    mu = float(rh.max()); md = float(rl.min())

    if bp >= 75: d, c = "BULLISH", min(95, int(50 + bp / 2))
    elif brp >= 75: d, c = "BEARISH", min(95, int(50 + brp / 2))
    elif bp >= 60: d, c = "LEAN BULLISH", min(75, int(40 + bp / 3))
    elif brp >= 60: d, c = "LEAN BEARISH", min(75, int(40 + brp / 3))
    else: d, c = "NEUTRAL / CHOP", max(20, int(50 - abs(bp - 50)))

    # Cap direction to "LEAN" variant when only 1-2 matches (single data point ≠ conviction)
    if len(rc) <= 2 and d in ("BULLISH", "BEARISH"):
        d = "LEAN " + d

    # Scale by best composite quality AND match count
    # v2.3: less punishing formula — old version crushed confidence via multiplicative dampeners
    quality_scale = min(1.0, best_raw_composite / 0.65)   # 0.65 — sharper quality gate
    count_scale = min(1.0, len(rc) / 3)    # Full confidence at 3+ projection-eligible matches
    # Blended scaling: average of quality and count, not multiplicative
    # This prevents one low factor from destroying the other
    combined_scale = 0.5 * quality_scale + 0.5 * count_scale
    c = int(c * combined_scale)
    s = f"{total} matches. {bp:.0f}% up, {brp:.0f}% down. Avg: {am:.1f}pts. Exp: {ec:+.1f}pts."

    return ForwardProjection(d, c, round(ec, 2), round(eh, 2), round(el, 2),
                             round(current_price + eh, 2), round(current_price + el, 2),
                             round(current_price + ec, 2), round(bp, 1), round(brp, 1),
                             round(am, 2), round(mu, 2), round(md, 2), total, s)


# =================================================================
# --- DAY CACHE v2 ---
# =================================================================

class DayCache:
    def __init__(self, db_path=Path("fractal_cache.db")):
        self.db_path = db_path
        self._init()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init(self):
        with self._conn() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS daily_bars (
                date TEXT, bar_idx INTEGER, timestamp TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (date, bar_idx))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_bars_date ON daily_bars(date)")
            conn.execute("""CREATE TABLE IF NOT EXISTS day_context (
                date TEXT PRIMARY KEY, prev_close REAL DEFAULT 0,
                vix_open REAL DEFAULT 0, gap_pct REAL DEFAULT 0,
                dow INTEGER DEFAULT 0, avg_volume REAL DEFAULT 0,
                bar_count INTEGER DEFAULT 0, day_return REAL DEFAULT 0,
                rvol_bucket TEXT DEFAULT 'NORMAL',
                open_type TEXT DEFAULT 'AUCTION',
                tnx_trend TEXT DEFAULT 'NEUTRAL')""")
            # Migrations for DBs created before these columns existed
            for _col, _def in [("rvol_bucket","'NORMAL'"), ("open_type","'AUCTION'"), ("tnx_trend","'NEUTRAL'")]:
                try:
                    conn.execute(f"ALTER TABLE day_context ADD COLUMN {_col} TEXT DEFAULT {_def}")
                except Exception:
                    pass  # Column already exists
            conn.execute("""CREATE TABLE IF NOT EXISTS bars_15m (
                date TEXT, bar_idx INTEGER, timestamp TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (date, bar_idx))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS bars_1m_compressed (
                date TEXT PRIMARY KEY, closes BLOB, bar_count INTEGER)""")
            conn.commit()

    def has_date(self, date_str):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM daily_bars WHERE date=?", (date_str,)).fetchone()[0] > 0

    def store_day(self, date_str, day_df, prev_close=0.0, vix_open=0.0, avg_volume=0.0):
        if day_df.empty or self.has_date(date_str): return
        with self._conn() as conn:
            for i, (idx, row) in enumerate(day_df.iterrows()):
                conn.execute("INSERT OR IGNORE INTO daily_bars VALUES (?,?,?,?,?,?,?,?)",
                    (date_str, i, str(idx), float(row["Open"]), float(row["High"]),
                     float(row["Low"]), float(row["Close"]), float(row["Volume"])))
            o = day_df["Open"].values; c = day_df["Close"].values
            gp = ((o[0] - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
            dd = day_df.index[0]
            dw = dd.weekday() if hasattr(dd, 'weekday') else 0
            dr = ((c[-1] - o[0]) / o[0] * 100) if o[0] > 0 else 0.0
            # Compute rvol_bucket and open_type for storage
            rv_bucket = "NORMAL"
            vol_total = float(day_df["Volume"].sum())
            if avg_volume > 0:
                rv = vol_total / avg_volume
                if rv < 0.5: rv_bucket = "VERY_LOW"
                elif rv < 0.8: rv_bucket = "LOW"
                elif rv < 1.5: rv_bucket = "NORMAL"
                elif rv < 2.0: rv_bucket = "HIGH"
                else: rv_bucket = "VERY_HIGH"
            ot_bucket = "AUCTION"
            if len(day_df) >= 6:
                ot_c = day_df["Close"].values[:6].astype(float)
                ot_h = day_df["High"].values[:6].astype(float)
                ot_l = day_df["Low"].values[:6].astype(float)
                ot_range = ot_h.max() - ot_l.min()
                if ot_range > 0:
                    rel = (ot_c[-1] - ot_c[0]) / ot_range
                    revs = sum(1 for i in range(1, 6) if (ot_c[i]-ot_c[i-1])*(ot_c[-1]-ot_c[0]) < 0)
                    if rel > 0.65 and revs <= 1: ot_bucket = "DRIVE_UP"
                    elif rel < -0.65 and revs <= 1: ot_bucket = "DRIVE_DOWN"
            conn.execute(
                "INSERT OR REPLACE INTO day_context VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (date_str, prev_close, vix_open, gp, dw, avg_volume,
                 len(day_df), round(dr, 4), rv_bucket, ot_bucket, "NEUTRAL"))
            conn.commit()

    def store_15m(self, date_str, df_15m):
        if df_15m.empty: return
        with self._conn() as conn:
            for i, (idx, row) in enumerate(df_15m.iterrows()):
                conn.execute("INSERT OR IGNORE INTO bars_15m VALUES (?,?,?,?,?,?,?,?)",
                    (date_str, i, str(idx), float(row["Open"]), float(row["High"]),
                     float(row["Low"]), float(row["Close"]), float(row["Volume"])))
            conn.commit()

    def store_1m_compressed(self, date_str, closes_1m):
        if len(closes_1m) < 30: return
        blob = closes_1m.astype(np.float32).tobytes()
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO bars_1m_compressed VALUES (?,?,?)",
                (date_str, blob, len(closes_1m)))
            conn.commit()

    def get_1m_shape(self, date_str):
        with self._conn() as conn:
            row = conn.execute("SELECT closes, bar_count FROM bars_1m_compressed WHERE date=?", (date_str,)).fetchone()
        if not row: return None
        c = np.frombuffer(row[0], dtype=np.float32)
        mn, mx = c.min(), c.max()
        return (c - mn) / (mx - mn) if mx > mn else None

    def get_15m_shape(self, date_str):
        with self._conn() as conn:
            rows = conn.execute("SELECT close FROM bars_15m WHERE date=? ORDER BY bar_idx", (date_str,)).fetchall()
        if not rows or len(rows) < 4: return None
        c = np.array([r[0] for r in rows])
        mn, mx = c.min(), c.max()
        return (c - mn) / (mx - mn) if mx > mn else None

    def get_context(self, date_str):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM day_context WHERE date=?", (date_str,)).fetchone()
        if not row:
            return {"vix": 0, "gap_pct": 0, "dow": 0, "prev_close": 0,
                    "avg_volume": 0, "rvol_bucket": "NORMAL",
                    "open_type": "AUCTION", "tnx_trend": "NEUTRAL"}
        d = dict(row)
        return {
            "vix":         d.get("vix_open", 0),
            "gap_pct":     d.get("gap_pct", 0),
            "dow":         d.get("dow", 0),
            "prev_close":  d.get("prev_close", 0),
            "avg_volume":  d.get("avg_volume", 0),
            "rvol_bucket": d.get("rvol_bucket", "NORMAL"),
            "open_type":   d.get("open_type", "AUCTION"),
            "tnx_trend":   d.get("tnx_trend", "NEUTRAL"),
        }

    def get_all_dates(self):
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT date FROM daily_bars ORDER BY date").fetchall()]

    def get_day(self, date_str):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close, volume FROM daily_bars WHERE date=? ORDER BY bar_idx",
                (date_str,)).fetchall()
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df.index = pd.to_datetime(df["Timestamp"], format="mixed"); df.drop(columns=["Timestamp"], inplace=True)
        return df

    def get_day_count(self):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(DISTINCT date) FROM daily_bars").fetchone()[0]


# =================================================================
# --- BULK BACKFILL (TIER 1) ---
# =================================================================

def backfill_from_ibkr(ibkr, cache, target_days=2000):
    current_count = cache.get_day_count()
    if current_count >= target_days:
        logger.info(f"Fractal cache: {current_count} days (target {target_days}). Skip backfill.")
        return current_count

    logger.info(f"Fractal backfill: {current_count} → {target_days} days...")
    if not ibkr or not ibkr.connected:
        logger.warning("Backfill: IBKR not connected.")
        return current_count

    import time as _time
    total_fetched = 0
    chunks = (target_days - current_count) // 40 + 1
    zero_chunks = 0

    # Prefer continuous futures (ES_CONT) for deep history — goes back 2+ years
    # Fall back to front month (ES) which only has ~8 months
    contract = ibkr._contracts.get("ES_CONT") or ibkr._contracts.get("ES")
    if not contract:
        logger.warning("Backfill: No ES contract available.")
        return current_count
    is_cont = "ES_CONT" in ibkr._contracts
    contract_label = "ES continuous" if is_cont else getattr(contract, 'localSymbol', 'ES')
    max_chunks = 50 if is_cont else 15  # Continuous = 5+ years, front month = ~8 months
    logger.info(f"  Using {contract_label} (max {max_chunks} chunks)")

    try:
        from ib_insync import util
    except ImportError:
        logger.warning("Backfill: ib_insync not available — cannot backfill")
        return current_count

    for chunk_i in range(min(chunks, max_chunks)):
        try:
            end_dt = now_et().replace(tzinfo=None) - timedelta(days=60 * chunk_i)
            # ContFuture doesn't allow endDateTime — use empty string for latest data
            end_str = "" if (is_cont and chunk_i == 0) else end_dt.strftime("%Y%m%d-%H:%M:%S")
            logger.info(f"  Chunk {chunk_i+1}: 60D ending {end_str[:8] if end_str else 'latest'}...")

            # For continuous contracts, IBKR doesn't allow endDateTime at all
            # Use front-month ES for chunked backfill if continuous fails on chunk > 0
            fetch_contract = contract
            if is_cont and chunk_i > 0:
                # Fall back to front-month for historical chunks with endDateTime
                fetch_contract = ibkr._contracts.get("ES") or contract
                end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")

            bars = ibkr.ib.reqHistoricalData(
                fetch_contract, endDateTime=end_str, durationStr="60 D",
                barSizeSetting="5 mins", whatToShow="TRADES",
                useRTH=False, formatDate=1)
            if not bars:
                logger.info(f"  No more data at chunk {chunk_i+1}.")
                break

            df = util.df(bars)
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")
            df["Date"] = df.index.date

            prev_close = 0.0
            chunk_count = 0
            for d in sorted(df["Date"].unique()):
                date_str = str(d)
                day_df = df[df["Date"] == d]
                if not day_df.empty and len(day_df) >= 20:
                    if not cache.has_date(date_str):
                        cache.store_day(date_str, day_df, prev_close=prev_close)
                        chunk_count += 1; total_fetched += 1
                    prev_close = float(day_df["Close"].iloc[-1])

            logger.info(f"  Chunk {chunk_i+1}: +{chunk_count} days")

            # Early exit: only start counting consecutive empties after we've pushed
            # past the already-cached recent data (first 6 chunks = ~360 calendar days)
            if chunk_count == 0:
                if chunk_i >= 6:
                    zero_chunks += 1
                    if zero_chunks >= 2:
                        logger.info(f"  2 consecutive empty chunks past cached range — all available data fetched.")
                        break
                else:
                    logger.debug(f"  Chunk {chunk_i+1} empty but still in cached range, continuing...")
            else:
                zero_chunks = 0

            _time.sleep(2)
        except Exception as e:
            logger.warning(f"  Chunk {chunk_i+1} failed: {e}")
            _time.sleep(5)

    final = cache.get_day_count()
    logger.info(f"Backfill done: {final} days (+{total_fetched} new)")
    return final


# =================================================================
# --- FRACTAL ENGINE v2.0 ---
# =================================================================

class FractalEngine:
    """
    v2.0 — drop-in replacement for v1.
    Usage: engine.analyze(es_5m, current_price) OR
           engine.analyze(es_5m, current_price, es_1m=..., es_15m=..., vix=...)
    """

    def __init__(self, top_n=5, min_score=0.70, min_bars=12,
                 cache_path=Path("fractal_cache.db")):
        self.top_n = top_n
        self.min_score = min_score
        self.min_bars = min_bars
        self.cache = DayCache(cache_path)
        self.outcome_tracker = OutcomeTracker(cache_path)
        self.multi_res = MultiResolutionMatcher()
        self._backfill_done = False
        logger.info(f"FractalEngine v2.0 initialized. Cache has {self.cache.get_day_count()} days.")

    def backfill(self, ibkr, target_days=1000):
        if self._backfill_done: return
        self._backfill_done = True
        count = backfill_from_ibkr(ibkr, self.cache, target_days)
        return count

    def analyze(self, es_5m, current_price, es_1m=None, es_15m=None, vix=None,
               prev_day_close=0.0, nq_15m=None, tnx=None, open_type=None,
               target_date=None):
        """
        open_type: string from calc_opening_type() e.g. 'OPEN DRIVE UP', 'OPEN AUCTION'
        tnx: DataFrame with bond yield data (TNX 15m), used for trend context
        target_date: override for datetime.now().date() (used by backtester)
        """
        try:
            start_time = time.time()
            df = es_5m.copy()
            if df.empty: return self._empty_result("No 5-min data")
            # Defensive: ensure DatetimeIndex survives .copy() / concat
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.DatetimeIndex(df.index)
                except (ValueError, TypeError):
                    # Mixed tz-aware/naive timestamps — normalize via UTC
                    df.index = pd.to_datetime(df.index, utc=True)
            if df.index.tz is not None: df.index = df.index.tz_convert("America/New_York")
            df["Date"] = df.index.date

            today = target_date or now_et().date()
            today_df = df[df["Date"] == today]
            if len(today_df) < self.min_bars:
                return self._empty_result(f"Only {len(today_df)} bars, need {self.min_bars}")

            # Detect session type for filtered matching
            if target_date:
                session_type = None  # Backtester — no session filtering
            else:
                _now_t = now_et().time()
                if dtime(9, 30) <= _now_t < dtime(16, 0):
                    session_type = "DAY"
                elif _now_t >= dtime(18, 0) or _now_t < dtime(3, 0):
                    session_type = "NIGHT"
                else:
                    session_type = None  # 4-6 PM gap

            # Filter today's bars to session window
            if session_type:
                today_df_session = _filter_bars_by_session(today_df, session_type)
                if len(today_df_session) >= self.min_bars:
                    today_df_match = today_df_session
                else:
                    today_df_match = today_df
                    session_type = None  # Not enough bars yet, use full pool
            else:
                today_df_match = today_df

            self._update_cache(df, today, vix)
            if es_15m is not None and not es_15m.empty: self._cache_15m(es_15m, today)
            if es_1m is not None and not es_1m.empty: self._cache_1m(es_1m, today)

            # VIX at open
            vix_open = 0.0
            if vix is not None and not vix.empty:
                try:
                    vt = vix[vix.index.date == today] if hasattr(vix.index, 'date') else vix
                    vix_open = float(vt["Close"].iloc[0]) if not vt.empty else float(vix["Close"].iloc[-1])
                except Exception: vix_open = float(vix["Close"].iloc[-1]) if not vix.empty else 0.0

            # Prev close for gap
            if prev_day_close == 0:
                yd = df[df["Date"] < today]
                if not yd.empty: prev_day_close = float(yd["Close"].iloc[-1])

            # Avg volume
            avg_vol = 0.0
            dates = sorted(df["Date"].unique())
            if len(dates) > 5:
                vols = [df[df["Date"]==d]["Volume"].sum() for d in dates[-21:-1] if len(df[df["Date"]==d]) > 20]
                if vols: avg_vol = np.mean(vols)

            today_sig = extract_signature(today_df_match, prev_close=prev_day_close, vix_open=vix_open, avg_volume=avg_vol)
            if today_sig is None: return self._empty_result("Flat market?")
            cbc = len(today_df_match)

            # Compute TNX trend for today (UP / DOWN / NEUTRAL)
            tnx_trend_today = "NEUTRAL"
            if tnx is not None and not tnx.empty and len(tnx) >= 5:
                try:
                    # Handle both DataFrame (has "Close" column) and Series (is the Close column)
                    tnx_vals = (tnx["Close"] if hasattr(tnx, "columns") and "Close" in tnx.columns else tnx).values.astype(float)
                    tnx_trend_today = "UP" if tnx_vals[-1] > tnx_vals[-5] * 1.001 else                                       "DOWN" if tnx_vals[-1] < tnx_vals[-5] * 0.999 else "NEUTRAL"
                except Exception:
                    pass
            today_sig.tnx_trend = tnx_trend_today

            # Map open_type string from main bot to internal bucket
            today_sig.open_type_bucket = _classify_open_type(open_type)

            # Store today's TNX + open_type in cache so future cycles can compare
            try:
                with self.cache._conn() as _cc:
                    _cc.execute(
                        "UPDATE day_context SET tnx_trend=?, open_type=? WHERE date=?",
                        (tnx_trend_today, today_sig.open_type_bucket, str(today)))
                    _cc.commit()
            except Exception:
                pass

            # Multi-res today shapes
            today_1m_shape = self._extract_today_shape(es_1m, today, 30)
            today_15m_shape = self._extract_today_shape(es_15m, today, 4)
            today_nq_shape = self._extract_today_shape(nq_15m, today, 4)

            all_hist = self._gather_historical_days(df, today)
            if not all_hist: return self._empty_result("No historical days")

            # Context filter
            candidates = [(ds, hdf, self.cache.get_context(ds)) for ds, hdf in all_hist]
            filtered = ContextFilter.filter(today_sig, candidates, strict=True)

            # Score with session filter
            scored = self._score_candidates(
                today_sig, cbc, filtered, session_type, start_time,
                today_1m_shape, today_15m_shape, today_nq_shape)
            scored.sort(key=lambda x: x[1]["composite"], reverse=True)
            top = scored[:self.top_n]

            # Fallback: if session filter produced no matches, retry with full pool
            if not top and session_type is not None:
                logger.info(f"Fractal session filter ({session_type}) found no matches — falling back to full pool")
                session_type = None
                today_sig = extract_signature(today_df, prev_close=prev_day_close, vix_open=vix_open, avg_volume=avg_vol)
                if today_sig is None:
                    return self._empty_result("Flat market?")
                today_sig.tnx_trend = tnx_trend_today
                today_sig.open_type_bucket = _classify_open_type(open_type)
                cbc = len(today_df)
                scored = self._score_candidates(
                    today_sig, cbc, filtered, None, start_time,
                    today_1m_shape, today_15m_shape, today_nq_shape)
                scored.sort(key=lambda x: x[1]["composite"], reverse=True)
                top = scored[:self.top_n]

            if not top:
                return self._empty_result(f"No days above {self.min_score:.0%} ({len(filtered)} scanned)")

            proj = build_projection(top, current_price, cbc)

            # Record predictions
            try:
                self.outcome_tracker.record_prediction(
                    [str(s.date) for s, _ in top], proj.direction,
                    [sc["composite"] for _, sc in top])
            except Exception as e:
                logger.warning(f"Failed to record fractal prediction: {e}")

            # Match details
            details = []
            for sig, sc in top:
                if sig.bar_count > cbc:
                    mp = sig.full_closes[cbc - 1]
                    mv = sig.full_closes[-1] - mp
                    oc = "RALLIED" if mv > 0 else "SOLD OFF"
                else: mv, oc = 0, "N/A"
                details.append({
                    "date": str(sig.date),
                    "composite_score": round(sc["composite"]*100, 1),
                    "price_corr": round(sc.get("price_corr",0)*100, 1),
                    "dtw_score": round(sc.get("dtw",0)*100, 1),
                    "momentum_score": round(sc.get("momentum",0)*100, 1),
                    "wick_score": round(sc.get("wick_pattern",0)*100, 1),
                    "atr_score": round(sc.get("atr_norm",0)*100, 1),
                    "segment_score": round(sc.get("segment_match",0)*100, 1),
                    "context_bonus": round(sc.get("context_bonus",0)*100, 1),
                    "outcome_bonus": round(sc.get("outcome_bonus",0)*100, 1),
                    "multi_res": round(sc.get("multi_res_mult",1.0), 3),
                    "outcome": oc, "remaining_move": round(mv, 2),
                    "day_return_pct": round(sig.day_return_pct, 2),
                })

            pt = self._build_prompt(details, proj, today_sig, cbc, len(filtered), session_type)

            elapsed = time.time() - start_time
            _session_label = session_type or "FULL"
            logger.info(f"Fractal analysis ({_session_label}) completed in {elapsed:.1f}s ({len(top)} days matched)")

            return {
                "status": "OK", "match_count": len(top),
                "total_days_scanned": len(filtered),
                "cached_days": self.cache.get_day_count(),
                "top_matches": details, "projection": proj,
                "today_bars": cbc,
                "today_volatility": round(today_sig.volatility * 100, 3),
                "today_return_pct": round(today_sig.day_return_pct, 2),
                "prompt_text": pt,
                "session_filter": _session_label,
            }
        except Exception as e:
            logger.error(f"FractalEngine v2 failed: {e}", exc_info=True)
            return self._empty_result(f"Engine error: {e}")

    def _score_candidates(self, today_sig, cbc, filtered, session_type, start_time,
                          today_1m_shape, today_15m_shape, today_nq_shape):
        """Score filtered candidates, optionally applying session-based bar filtering."""
        scored = []
        n_scored = 0
        for date_str, hist_df, ctx_bonus in filtered:
            if time.time() - start_time > CFG.FRACTAL_TIME_BUDGET:
                logger.warning(f"Fractal time budget exceeded ({CFG.FRACTAL_TIME_BUDGET:.1f}s) — "
                               f"stopped after {n_scored}/{len(filtered)} candidates")
                break
            if session_type:
                hist_df = _filter_bars_by_session(hist_df, session_type)
            if len(hist_df) < cbc:
                continue
            ctx = self.cache.get_context(date_str)
            hp = extract_signature(hist_df, max_bars=cbc, prev_close=ctx.get("prev_close", 0),
                                   vix_open=ctx.get("vix", 0), avg_volume=ctx.get("avg_volume", 0))
            hf = extract_signature(hist_df, prev_close=ctx.get("prev_close", 0),
                                   vix_open=ctx.get("vix", 0), avg_volume=ctx.get("avg_volume", 0))
            if hp is None or hf is None:
                continue

            ob = self.outcome_tracker.get_day_bonus(date_str)
            sc = score_similarity(today_sig, hf, hp, outcome_bonus=ob, context_bonus=ctx_bonus)

            h1m = self.cache.get_1m_shape(date_str)
            h15m = self.cache.get_15m_shape(date_str)
            mrm = self.multi_res.score(today_sig, hf, today_1m_shape, h1m, today_15m_shape, h15m,
                                       today_nq_shape, h15m)
            sc["composite"] = min(0.99, sc["composite"] * mrm)
            sc["multi_res_mult"] = mrm

            if sc["composite"] >= self.min_score:
                scored.append((hf, sc))
            n_scored += 1
        return scored

    def record_outcome(self, actual_dir, move_pts):
        try: self.outcome_tracker.record_actual(actual_dir, move_pts)
        except Exception as e: logger.warning(f"Outcome record failed: {e}")

    def _extract_today_shape(self, df, today, min_len):
        if df is None or df.empty: return None
        try:
            td = df[df.index.date == today] if hasattr(df.index[0], 'date') else df
            if len(td) < min_len: return None
            c = td["Close"].values; mn, mx = c.min(), c.max()
            return (c - mn) / (mx - mn) if mx > mn else None
        except Exception: return None

    def _update_cache(self, df, today, vix=None):
        dates = sorted(df["Date"].unique())
        prev_close = 0.0
        for d in dates:
            if d == today: continue
            day_df = df[df["Date"] == d]
            if len(day_df) < 20:
                if not day_df.empty: prev_close = float(day_df["Close"].iloc[-1])
                continue
            if not self.cache.has_date(str(d)):
                vv = 0.0
                if vix is not None and not vix.empty:
                    try:
                        vd = vix[vix.index.date == d] if hasattr(vix.index, 'date') else pd.DataFrame()
                        if not vd.empty: vv = float(vd["Close"].iloc[0])
                    except Exception as e:
                        logger.debug(f"VIX lookup for {d} failed: {e}")
                self.cache.store_day(str(d), day_df, prev_close=prev_close, vix_open=vv)
            prev_close = float(day_df["Close"].iloc[-1])

    def _cache_15m(self, es_15m, today):
        try:
            df = es_15m.copy()
            if df.index.tz is not None: df.index = df.index.tz_convert("America/New_York")
            df["Date"] = df.index.date
            for d in df["Date"].unique():
                if d == today: continue
                dd = df[df["Date"]==d]
                if len(dd) >= 4: self.cache.store_15m(str(d), dd)
        except Exception as e:
            logger.debug(f"_cache_15m failed: {e}")

    def _cache_1m(self, es_1m, today):
        try:
            df = es_1m.copy()
            if df.index.tz is not None: df.index = df.index.tz_convert("America/New_York")
            df["Date"] = df.index.date
            for d in df["Date"].unique():
                if d == today: continue
                dd = df[df["Date"]==d]
                if len(dd) >= 30: self.cache.store_1m_compressed(str(d), dd["Close"].values)
        except Exception as e:
            logger.debug(f"_cache_1m failed: {e}")

    def _gather_historical_days(self, df, today):
        days = {}
        for d in df["Date"].unique():
            if d == today: continue
            dd = df[df["Date"]==d]
            if len(dd) >= 20: days[str(d)] = dd
        for ds in self.cache.get_all_dates():
            if ds not in days and ds != str(today):
                c = self.cache.get_day(ds)
                if not c.empty and len(c) >= 20: days[ds] = c
        return list(days.items())

    def _build_prompt(self, matches, proj, today_sig, bar_count, days_scanned, session_type=None):
        lines = []
        for i, m in enumerate(matches[:5], 1):
            lines.append(
                f"  #{i}  {m['date']}  |  Score: {m['composite_score']:.0f}%  "
                f"(Price:{m['price_corr']:.0f} DTW:{m['dtw_score']:.0f} "
                f"Mom:{m['momentum_score']:.0f} Wick:{m['wick_score']:.0f} "
                f"ATR:{m['atr_score']:.0f} Seg:{m['segment_score']:.0f}"
                f"{' MR:'+str(m['multi_res']) if m['multi_res']>1.0 else ''}"
                f"{' Ctx:+'+str(m['context_bonus']) if m['context_bonus']>0 else ''}"
                f"{' Track:+'+str(m['outcome_bonus']) if m['outcome_bonus']>0 else ''}"
                f")  |  After: {m['outcome']} {m['remaining_move']:+.1f}  "
                f"|  Day: {m['day_return_pct']:+.2f}%")
        mt = "\n".join(lines) if lines else "  No matches."
        _sess = f"Session: {session_type} | " if session_type else ""

        return f"""
FRACTAL PATTERN RECOGNITION v2.0 (PRIMARY SIGNAL):
{_sess}Matching: 11 features + context filter + multi-res + outcome tracking
Scanned {days_scanned} days with {bar_count} bars.

TOP MATCHES:
{mt}

PROJECTION (weighted):
  Direction: {proj.direction} ({proj.confidence}%)
  {proj.bullish_pct:.0f}% rallied | {proj.bearish_pct:.0f}% sold off
  Expected: {proj.expected_close_vs_current:+.1f} pts
  Range: +{proj.projected_high:.1f} / {proj.projected_low:.1f}
  Targets: Up {proj.upside_target:.2f} | Down {proj.downside_target:.2f}
  Close: {proj.expected_close_price:.2f}
  Avg move: {proj.avg_move:.1f} | Max: +{proj.max_seen_up:.1f} / {proj.max_seen_down:.1f}

CRITICAL: Fractal is the PRIMARY signal.
{proj.bullish_pct:.0f}% rallied → favor BULLISH. {proj.bearish_pct:.0f}% sold off → favor BEARISH.
Near 50/50 → favor FLAT.
"""

    def _empty_result(self, reason):
        return {
            "status": reason, "match_count": 0, "total_days_scanned": 0,
            "cached_days": self.cache.get_day_count(), "top_matches": [],
            "projection": ForwardProjection("NEUTRAL", 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, reason),
            "today_bars": 0, "today_volatility": 0, "today_return_pct": 0,
            "prompt_text": f"\nFRACTAL PATTERN RECOGNITION: {reason}\n",
        }


# =================================================================
# --- TELEGRAM FORMATTING ---
# =================================================================

def format_fractal_telegram(result):
    proj = result["projection"]
    if result["match_count"] == 0:
        return f"🔮 *Fractal:* {result['status']}"

    d = proj.direction
    emoji = "🟢" if "BULLISH" in d else "🔴" if "BEARISH" in d else "⚪"

    top3 = result["top_matches"][:3]
    ml = "\n".join(
        f"  • {m['date']} ({m['composite_score']:.0f}%"
        f"{' MR' if m.get('multi_res',1.0)>1.0 else ''}"
        f"{' Ctx' if m.get('context_bonus',0)>0 else ''}"
        f") → {m['outcome']} {m['remaining_move']:+.1f}"
        for m in top3)

    return (
        f"🔮 *FRACTAL v2.0*\n"
        f"Scanned: {result['total_days_scanned']} | Cache: {result['cached_days']}\n"
        f"Top:\n{ml}\n\n"
        f"{emoji} *{proj.direction}* ({proj.confidence}%)\n"
        f"• {proj.bullish_pct:.0f}% up | {proj.bearish_pct:.0f}% down\n"
        f"• Exp: {proj.expected_close_vs_current:+.1f} pts\n"
        f"• Range: +{proj.projected_high:.1f} / {proj.projected_low:.1f}\n"
        f"• ↑ {proj.upside_target:.2f}  ↓ {proj.downside_target:.2f}")
