"""
Backtest harness for ES Futures trading strategy.

Replays historical days from fractal_cache.db through the decision pipeline
without requiring IBKR, Claude API, or Telegram.

Usage:
    python backtest.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--verbose] [--min-bars 12]
                       [--any-bar] [--entry-bar N] [--walk-forward]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fractal_engine import FractalEngine, DayCache, ForwardProjection, build_projection
from confidence_engine import (
    AccuracyTracker,
    apply_confidence_pipeline,
    calc_regime_adjustment,
)
import trade_status as ts


# ─── Backtest Result ───────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_days: int = 0
    days_with_signal: int = 0
    days_flat: int = 0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    stops: int = 0
    time_exits: int = 0
    eod_exits: int = 0
    total_pnl: float = 0.0
    gross_wins: float = 0.0
    gross_losses: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    trade_log: list = field(default_factory=list)
    signal_log: list = field(default_factory=list)


# ─── Mock Journal for AccuracyTracker ──────────────────────────────

class BacktestJournal:
    """In-memory SQLite journal that accumulates backtest trades for streak tracking."""

    def __init__(self):
        self._db = sqlite3.connect(":memory:")
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(f"""
            CREATE TABLE trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                price       REAL NOT NULL,
                verdict     TEXT NOT NULL,
                confidence  INTEGER NOT NULL DEFAULT 0,
                target      REAL, stop REAL,
                status      TEXT NOT NULL DEFAULT '{ts.OPEN}',
                pnl         REAL NOT NULL DEFAULT 0.0,
                contracts   INTEGER NOT NULL DEFAULT 1,
                reasoning   TEXT, session TEXT,
                fractal_recorded INTEGER DEFAULT NULL,
                oca_group   TEXT DEFAULT NULL
            )
        """)
        self._db.commit()

    def _conn(self):
        return self._db

    def add_trade(self, timestamp, price, verdict, confidence, target, stop, status, pnl, contracts=1):
        self._db.execute(
            "INSERT INTO trades (timestamp, price, verdict, confidence, target, stop, status, pnl, contracts, session) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, price, verdict, confidence, target, stop, status, pnl, max(contracts, 1), "BACKTEST"),
        )
        self._db.commit()


# ─── Position Suggestion (standalone, no CFG dependency) ───────────

def backtest_position_fn(confidence: int, flat_threshold: int = 60):
    """Position sizing for backtest: 1 contract if above threshold, 0 otherwise."""
    if confidence < flat_threshold:
        return 0, f"NO TRADE ({confidence}% < {flat_threshold}%)"
    return 1, f"1 ct (backtest)"


# ─── Fractal Projection -> Verdict ─────────────────────────────────

def projection_to_verdict(proj: ForwardProjection):
    """Convert fractal projection to (verdict, confidence)."""
    if "BULL" in proj.direction.upper():
        return "BULLISH", proj.confidence
    elif "BEAR" in proj.direction.upper():
        return "BEARISH", proj.confidence
    return "FLAT", proj.confidence


# ─── Slippage Model ────────────────────────────────────────────────

def apply_slippage(entry_price: float, is_long: bool, vix: float = 15.0) -> float:
    """Apply realistic slippage to entry price based on volatility regime."""
    from bot_config import CFG
    if vix > 25:
        slip = CFG.BACKTEST_SLIPPAGE_HIGH_VOL
    else:
        slip = CFG.BACKTEST_SLIPPAGE_NORMAL
    # Slippage is adverse: longs enter higher, shorts enter lower
    return entry_price + slip if is_long else entry_price - slip


# ─── Trade Simulation ──────────────────────────────────────────────

def simulate_trade(day_df: pd.DataFrame, entry_bar_idx: int, verdict: str,
                   entry_price: float, target: float, stop: float,
                   slippage: float = 0.0, max_bars: int = 24) -> dict:
    """Walk forward through bars checking target/stop/time-exit.

    max_bars=24 means 24 × 5min = 2 hours (matching the bot's time exit).
    Returns dict with outcome, pnl, bars_held.
    """
    # Apply slippage to entry price (adverse fill)
    entry_price = entry_price + slippage
    is_long = "BULL" in verdict.upper()
    remaining = day_df.iloc[entry_bar_idx:]

    for i, (idx, bar) in enumerate(remaining.iterrows()):
        if i == 0:
            continue  # Skip entry bar

        if i >= max_bars:
            # Time exit
            close = float(bar["Close"])
            pnl = (close - entry_price) if is_long else (entry_price - close)
            # Cap at stop if worse
            stop_pnl = (stop - entry_price) if is_long else (entry_price - stop)
            if pnl < 0 and stop_pnl <= 0 and pnl < stop_pnl:
                pnl = stop_pnl
                return {"outcome": ts.STOP, "pnl": round(pnl, 2), "bars_held": i,
                        "exit_price": stop}
            return {"outcome": ts.TIME_EXIT, "pnl": round(pnl, 2), "bars_held": i,
                    "exit_price": close}

        high = float(bar["High"])
        low = float(bar["Low"])

        if is_long:
            if high >= target:
                return {"outcome": ts.WIN, "pnl": round(target - entry_price, 2),
                        "bars_held": i, "exit_price": target}
            if low <= stop:
                return {"outcome": ts.STOP, "pnl": round(stop - entry_price, 2),
                        "bars_held": i, "exit_price": stop}
        else:
            if low <= target:
                return {"outcome": ts.WIN, "pnl": round(entry_price - target, 2),
                        "bars_held": i, "exit_price": target}
            if high >= stop:
                return {"outcome": ts.STOP, "pnl": round(entry_price - stop, 2),
                        "bars_held": i, "exit_price": stop}

    # End of day — close at last bar
    if len(remaining) > 1:
        last_close = float(remaining["Close"].iloc[-1])
        pnl = (last_close - entry_price) if is_long else (entry_price - last_close)
        return {"outcome": ts.TIME_EXIT, "pnl": round(pnl, 2),
                "bars_held": len(remaining) - 1, "exit_price": last_close}

    return {"outcome": ts.TIME_EXIT, "pnl": 0.0, "bars_held": 0, "exit_price": entry_price}


# ─── Main Backtest Loop ───────────────────────────────────────────

def run_backtest(cache_path: Path = Path("fractal_cache.db"),
                 start_date: str | None = None,
                 end_date: str | None = None,
                 min_entry_bars: int = 12,
                 verbose: bool = False,
                 any_bar: bool = False,
                 entry_bar: int | None = None,
                 walk_forward: bool = False) -> BacktestResult:
    """Run the backtest across cached historical days."""

    cache = DayCache(cache_path)
    engine = FractalEngine(cache_path=cache_path)
    journal = BacktestJournal()
    result = BacktestResult()

    all_cached_dates = cache.get_all_dates()
    all_dates = all_cached_dates[:]
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    if not all_dates:
        print("No cached dates found.")
        return result

    # Determine fixed entry bar (--entry-bar overrides default min_entry_bars)
    fixed_entry_bar = entry_bar if entry_bar is not None else min_entry_bars

    print(f"Backtest: {len(all_dates)} days from {all_dates[0]} to {all_dates[-1]}")
    if any_bar:
        print(f"Entry mode: ANY-BAR (scan from bar {min_entry_bars} onward)")
    else:
        print(f"Entry mode: FIXED bar {fixed_entry_bar} (= {fixed_entry_bar * 5} min into session)")
    if walk_forward:
        print("Walk-forward: ON (no future peeking — only prior days used for matching)")
    print("=" * 70)

    cumulative_pnl = 0.0

    for day_idx, date_str in enumerate(all_dates):
        result.total_days += 1
        day_df = cache.get_day(date_str)
        context = cache.get_context(date_str)

        if day_df.empty or len(day_df) < min_entry_bars + 5:
            if verbose:
                print(f"  {date_str}: SKIP (only {len(day_df)} bars)")
            continue

        # Build multi-day DataFrame for fractal engine
        # Include up to 60 prior days for matching
        if walk_forward:
            # Walk-forward: only use days strictly before the current test date
            # Use full cache (all_cached_dates) so --start/--end don't starve the lookback pool
            prior_dates = [d for d in all_cached_dates if d < date_str][-60:]
        else:
            prior_dates = all_cached_dates[max(0, all_cached_dates.index(date_str) - 60):all_cached_dates.index(date_str)]
        frames = []
        for pd_str in prior_dates:
            pdf = cache.get_day(pd_str)
            if not pdf.empty:
                # Normalize fixed-offset tz -> named tz so concat preserves DatetimeIndex
                if hasattr(pdf.index, 'tz') and pdf.index.tz is not None:
                    pdf.index = pdf.index.tz_convert('America/New_York')
                frames.append(pdf)

        # Normalize the target day too
        if hasattr(day_df.index, 'tz') and day_df.index.tz is not None:
            day_df = day_df.copy()
            day_df.index = day_df.index.tz_convert('America/New_York')
        frames.append(day_df)

        if len(frames) < 10:
            if verbose:
                print(f"  {date_str}: SKIP (< 10 days of history)")
            continue

        combined_df = pd.concat(frames).sort_index()

        # Simulation uses original day_df (un-remapped)
        sim_df = day_df

        # Parse target date for engine's target_date parameter
        target_dt = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Build minimal regime from cached VIX
        vix_val = context.get("vix", 15)
        regime = calc_regime_adjustment(
            vix_term={"vix": vix_val, "structure": "CONTANGO"},
            rvol={"rvol": 1.0},
            day_type={"trend_probability": 50},
            mtf={"score": 0},
        )

        # Determine which bars to try for entry
        if any_bar:
            entry_bars_to_try = list(range(min_entry_bars, len(sim_df) - 5))
        else:
            entry_bars_to_try = [fixed_entry_bar]

        trade_taken_today = False

        for try_bar in entry_bars_to_try:
            if trade_taken_today:
                break

            # Run fractal engine at this bar
            entry_price = float(sim_df["Close"].iloc[try_bar - 1])
            try:
                fractal_result = engine.analyze(
                    es_5m=combined_df,
                    current_price=entry_price,
                    prev_day_close=context.get("prev_close", 0),
                    target_date=target_dt,
                )
            except Exception as e:
                if verbose and try_bar == entry_bars_to_try[0]:
                    print(f"  {date_str}: FRACTAL ERROR: {e}")
                continue

            proj = fractal_result.get("projection")
            if proj is None or proj.direction == "NEUTRAL":
                if not any_bar:
                    result.days_flat += 1
                    if verbose:
                        print(f"  {date_str}: NEUTRAL (no fractal signal)")
                continue

            if not trade_taken_today:
                result.days_with_signal += 1

            # Convert projection to verdict
            verdict, raw_conf = projection_to_verdict(proj)

            # Build minimal metrics for confluence (fractal-only, others neutral)
            metrics = {
                "fractal": {"projection": proj},
                "flow_data": {},
                "liq_sweeps": {"active_sweep": "NONE"},
                "mtf_momentum": {"alignment": "MIXED / CONFLICTED"},
                "tick_proxy": {"extreme": ""},
                "cum_delta_bias": "",
                "opening_type": {"bias": "NEUTRAL"},
            }

            # Use simulated time for accuracy tracker
            day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=10, minute=0)
            tracker = AccuracyTracker(journal, now_fn=lambda dt=day_dt: dt)

            data = {"confidence": raw_conf, "verdict": verdict}
            data, final_verdict, conf, contracts, pos_str, confluence, _decomp = apply_confidence_pipeline(
                data, metrics, tracker, regime, md=None,
                position_suggestion_fn=backtest_position_fn,
            )

            if final_verdict == "FLAT" or contracts == 0:
                if not any_bar:
                    result.days_flat += 1
                    if verbose:
                        print(f"  {date_str}: FLAT ({verdict} {raw_conf}% -> {conf}% below {regime['flat_threshold']}%)")
                continue

            # Derive target and stop from fractal projection
            if "BULL" in final_verdict:
                target_price = proj.upside_target
                stop_price = max(proj.downside_target, entry_price - 6.0)
            else:
                target_price = proj.downside_target
                stop_price = min(proj.upside_target, entry_price + 6.0)

            # Sanity check
            if "BULL" in final_verdict and target_price <= entry_price:
                target_price = entry_price + 4.0
            if "BEAR" in final_verdict and target_price >= entry_price:
                target_price = entry_price - 4.0

            # Apply slippage to entry price
            is_long = "BULL" in final_verdict.upper()
            slipped_entry = apply_slippage(entry_price, is_long, vix=vix_val)
            slippage_delta = slipped_entry - entry_price

            # Simulate trade using the remapped frame
            trade_result = simulate_trade(
                sim_df, entry_bar_idx=try_bar, verdict=final_verdict,
                entry_price=slipped_entry, target=target_price, stop=stop_price,
                slippage=0.0,  # already applied via slipped_entry
                max_bars=24,
            )

            pnl = trade_result["pnl"]
            outcome = trade_result["outcome"]
            result.trades_taken += 1
            result.total_pnl += pnl
            cumulative_pnl += pnl
            trade_taken_today = True

            if pnl > 0:
                result.wins += 1
                result.gross_wins += pnl
            elif pnl < 0:
                result.losses += 1
                result.gross_losses += abs(pnl)

            if outcome == ts.STOP:
                result.stops += 1
            elif outcome == ts.TIME_EXIT:
                result.time_exits += 1

            # Track drawdown
            if cumulative_pnl > result.peak_pnl:
                result.peak_pnl = cumulative_pnl
            dd = result.peak_pnl - cumulative_pnl
            if dd > result.max_drawdown:
                result.max_drawdown = dd

            # Log trade to journal (for accuracy tracker in future days)
            status = ts.WIN if pnl > 0 else (ts.STOP if outcome == ts.STOP else ts.LOSS)
            journal.add_trade(
                timestamp=f"{date_str} 10:00",
                price=slipped_entry,
                verdict=final_verdict,
                confidence=conf,
                target=target_price,
                stop=stop_price,
                status=status,
                pnl=pnl,
            )

            # Record signal data for offline analysis (#9)
            result.signal_log.append({
                "date": date_str,
                "entry_bar": try_bar,
                "fractal_direction": proj.direction,
                "fractal_confidence": proj.confidence,
                "fractal_match_count": proj.match_count,
                "raw_verdict": verdict,
                "raw_confidence": raw_conf,
                "final_verdict": final_verdict,
                "final_confidence": conf,
                "regime": regime.get("regime_label", ""),
                "flat_threshold": regime.get("flat_threshold", 0),
                "vix": vix_val,
                "slippage": round(slippage_delta, 4),
                "contracts": contracts,
                "entry_price": round(slipped_entry, 2),
                "target_price": round(target_price, 2),
                "stop_price": round(stop_price, 2),
                "outcome": outcome,
                "pnl": pnl,
            })

            result.trade_log.append({
                "date": date_str,
                "verdict": final_verdict,
                "conf": conf,
                "entry": slipped_entry,
                "target": round(target_price, 2),
                "stop": round(stop_price, 2),
                "outcome": outcome,
                "pnl": pnl,
                "bars_held": trade_result["bars_held"],
                "exit_price": trade_result["exit_price"],
                "cum_pnl": round(cumulative_pnl, 2),
            })

            if verbose:
                emoji = "W" if pnl > 0 else "L"
                bar_info = f"bar {try_bar}" if any_bar else ""
                print(
                    f"  {date_str}: {emoji} {final_verdict} @ {slipped_entry:.2f} "
                    f"-> {outcome} {pnl:+.2f} pts "
                    f"(conf {conf}%, {trade_result['bars_held']} bars) "
                    f"[cum: {cumulative_pnl:+.2f}]"
                    f"{' ' + bar_info if bar_info else ''}"
                )

        # If any_bar mode and no trade was taken on any scanned bar, count as flat
        if any_bar and not trade_taken_today:
            result.days_flat += 1

    return result


# ─── Output Formatting ─────────────────────────────────────────────

def print_fractal_report(signal_log: list):
    """Print detailed fractal signal accuracy breakdown."""
    if not signal_log:
        print("\n  No signals to analyze for fractal report.")
        return

    print()
    print("=" * 70)
    print("FRACTAL SIGNAL ACCURACY REPORT")
    print("=" * 70)

    # 1. Signal distribution
    directions = {}
    for s in signal_log:
        d = s.get("fractal_direction", "UNKNOWN")
        directions[d] = directions.get(d, 0) + 1

    print(f"\n  Signal Distribution ({len(signal_log)} total signals):")
    for d in sorted(directions.keys()):
        pct = directions[d] / len(signal_log) * 100
        print(f"    {d:20s}: {directions[d]:4d} ({pct:.1f}%)")

    # 2. Direction accuracy
    print(f"\n  Direction Accuracy:")
    for label, match_fn in [
        ("ALL BULLISH",  lambda d: "BULL" in d),
        ("  BULLISH",    lambda d: d == "BULLISH"),
        ("  LEAN BULL",  lambda d: d == "LEAN BULLISH"),
        ("ALL BEARISH",  lambda d: "BEAR" in d),
        ("  BEARISH",    lambda d: d == "BEARISH"),
        ("  LEAN BEAR",  lambda d: d == "LEAN BEARISH"),
    ]:
        subset = [s for s in signal_log if match_fn(s.get("fractal_direction", ""))]
        if not subset:
            continue
        wins = sum(1 for s in subset if s["pnl"] > 0)
        acc = wins / len(subset) * 100
        avg_pnl = sum(s["pnl"] for s in subset) / len(subset)
        print(f"    {label:20s}: {wins:3d}/{len(subset):3d} ({acc:.1f}%), "
              f"avg P&L: {avg_pnl:+.2f} pts")

    # 3. Accuracy by confidence bucket
    print(f"\n  Accuracy by Confidence Bucket:")
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 101)]:
        label = f"{lo}-{min(hi-1, 100)}%"
        subset = [s for s in signal_log if lo <= s.get("fractal_confidence", 0) < hi]
        if not subset:
            print(f"    {label:10s}: no signals")
            continue
        wins = sum(1 for s in subset if s["pnl"] > 0)
        acc = wins / len(subset) * 100
        avg_pnl = sum(s["pnl"] for s in subset) / len(subset)
        print(f"    {label:10s}: {wins:3d}/{len(subset):3d} ({acc:.1f}%), "
              f"avg P&L: {avg_pnl:+.2f} pts")

    # 4. Accuracy by match count
    print(f"\n  Accuracy by Match Count:")
    match_counts = sorted(set(s.get("fractal_match_count", 0) for s in signal_log))
    for mc in match_counts:
        subset = [s for s in signal_log if s.get("fractal_match_count", 0) == mc]
        if not subset:
            continue
        wins = sum(1 for s in subset if s["pnl"] > 0)
        acc = wins / len(subset) * 100
        avg_pnl = sum(s["pnl"] for s in subset) / len(subset)
        mc_label = f"{mc} match" if mc == 1 else f"{mc} matches"
        print(f"    {mc_label:12s}: {wins:3d}/{len(subset):3d} ({acc:.1f}%), "
              f"avg P&L: {avg_pnl:+.2f} pts")

    # 5. Target distance vs actual move
    print(f"\n  Target vs Actual:")
    bull_signals = [s for s in signal_log if "BULL" in s.get("fractal_direction", "")]
    bear_signals = [s for s in signal_log if "BEAR" in s.get("fractal_direction", "")]

    if bull_signals:
        avg_td = np.mean([s["target_price"] - s["entry_price"] for s in bull_signals])
        avg_actual = np.mean([s["pnl"] for s in bull_signals])
        print(f"    BULL avg target: +{avg_td:.2f} pts, actual avg: {avg_actual:+.2f} pts")
    if bear_signals:
        avg_td = np.mean([s["entry_price"] - s["target_price"] for s in bear_signals])
        avg_actual = np.mean([s["pnl"] for s in bear_signals])
        print(f"    BEAR avg target: +{avg_td:.2f} pts, actual avg: {avg_actual:+.2f} pts")

    print()
    print("=" * 70)


def print_results(result: BacktestResult):
    """Print formatted backtest results."""
    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"  Total days scanned:    {result.total_days}")
    print(f"  Days with signal:      {result.days_with_signal}")
    print(f"  Days FLAT/skipped:     {result.days_flat}")
    print(f"  Trades taken:          {result.trades_taken}")
    print()

    if result.trades_taken == 0:
        print("  No trades to analyze.")
        return

    win_rate = result.wins / result.trades_taken * 100
    avg_win = result.gross_wins / result.wins if result.wins else 0
    avg_loss = result.gross_losses / result.losses if result.losses else 0
    profit_factor = result.gross_wins / result.gross_losses if result.gross_losses > 0 else float("inf")

    # Sharpe (simplified: daily P&L std)
    daily_pnls = [t["pnl"] for t in result.trade_log]
    sharpe = 0.0
    if len(daily_pnls) > 1:
        mean_pnl = np.mean(daily_pnls)
        std_pnl = np.std(daily_pnls, ddof=1)
        if std_pnl > 0:
            sharpe = (mean_pnl / std_pnl) * np.sqrt(252)

    print(f"  Win Rate:              {win_rate:.1f}% ({result.wins}W / {result.losses}L)")
    print(f"  Stops:                 {result.stops}")
    print(f"  Time Exits:            {result.time_exits}")
    print()
    print(f"  Total P&L:             {result.total_pnl:+.2f} pts (${result.total_pnl * 50:+,.0f})")
    print(f"  Avg Win:               +{avg_win:.2f} pts")
    print(f"  Avg Loss:              -{avg_loss:.2f} pts")
    print(f"  Profit Factor:         {profit_factor:.2f}")
    print(f"  Sharpe Ratio (ann):    {sharpe:.2f}")
    print(f"  Max Drawdown:          {result.max_drawdown:.2f} pts (${result.max_drawdown * 50:,.0f})")
    print()

    # Direction breakdown
    bull_trades = [t for t in result.trade_log if "BULL" in t["verdict"]]
    bear_trades = [t for t in result.trade_log if "BEAR" in t["verdict"]]
    if bull_trades:
        bull_wins = sum(1 for t in bull_trades if t["pnl"] > 0)
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        print(f"  BULLISH:  {len(bull_trades)} trades, {bull_wins}/{len(bull_trades)} wins "
              f"({bull_wins/len(bull_trades)*100:.0f}%), P&L: {bull_pnl:+.2f}")
    if bear_trades:
        bear_wins = sum(1 for t in bear_trades if t["pnl"] > 0)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        print(f"  BEARISH:  {len(bear_trades)} trades, {bear_wins}/{len(bear_trades)} wins "
              f"({bear_wins/len(bear_trades)*100:.0f}%), P&L: {bear_pnl:+.2f}")

    print()
    print("=" * 70)


# ─── Walk-Forward Analysis ────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """Results for a single train/test window."""
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    is_result: BacktestResult   # in-sample
    oos_result: BacktestResult  # out-of-sample


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward analysis results."""
    windows: list = field(default_factory=list)
    aggregate_is: BacktestResult = field(default_factory=BacktestResult)
    aggregate_oos: BacktestResult = field(default_factory=BacktestResult)
    degradation_ratio: float = 0.0   # OOS_winrate / IS_winrate (1.0 = no overfit)
    oos_sharpe: float = 0.0
    is_sharpe: float = 0.0
    robustness_score: float = 0.0


def _calc_sharpe(pnls: list) -> float:
    """Annualized Sharpe ratio from a list of trade P&Ls."""
    if len(pnls) < 2:
        return 0.0
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1)
    if std_pnl == 0:
        return 0.0
    return float((mean_pnl / std_pnl) * np.sqrt(252))


def _merge_results(results: list) -> BacktestResult:
    """Merge multiple BacktestResult objects into one aggregate."""
    merged = BacktestResult()
    cumulative_pnl = 0.0

    for r in results:
        merged.total_days += r.total_days
        merged.days_with_signal += r.days_with_signal
        merged.days_flat += r.days_flat
        merged.trades_taken += r.trades_taken
        merged.wins += r.wins
        merged.losses += r.losses
        merged.stops += r.stops
        merged.time_exits += r.time_exits
        merged.eod_exits += r.eod_exits
        merged.total_pnl += r.total_pnl
        merged.gross_wins += r.gross_wins
        merged.gross_losses += r.gross_losses
        merged.trade_log.extend(r.trade_log)
        merged.signal_log.extend(r.signal_log)

        # Recalculate max drawdown across all windows
        for t in r.trade_log:
            cumulative_pnl += t["pnl"]
            if cumulative_pnl > merged.peak_pnl:
                merged.peak_pnl = cumulative_pnl
            dd = merged.peak_pnl - cumulative_pnl
            if dd > merged.max_drawdown:
                merged.max_drawdown = dd

    return merged


def run_walk_forward(cache_path: Path = Path("fractal_cache.db"),
                     start_date: str | None = None,
                     end_date: str | None = None,
                     train_days: int = 30,
                     test_days: int = 10,
                     min_entry_bars: int = 12) -> WalkForwardReport:
    """
    Run rolling walk-forward analysis.

    Splits the date range into rolling windows:
    - Train window: walk_forward=False (sees all cached data for matching)
    - Test window: walk_forward=True (only prior days for matching)

    Returns a WalkForwardReport with per-window and aggregate statistics.
    """
    # Get all dates from cache
    cache = DayCache(cache_path)
    all_dates = cache.get_all_dates()

    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    if len(all_dates) < train_days + test_days:
        print(f"Not enough days ({len(all_dates)}) for walk-forward "
              f"(need {train_days} train + {test_days} test)")
        return WalkForwardReport()

    report = WalkForwardReport()
    window_num = 0

    print(f"\nWalk-Forward Analysis")
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
    print(f"  Train window: {train_days} days, Test window: {test_days} days")
    print("=" * 70)

    # Rolling windows
    offset = 0
    while offset + train_days + test_days <= len(all_dates):
        window_num += 1
        train_start = all_dates[offset]
        train_end = all_dates[offset + train_days - 1]
        test_start = all_dates[offset + train_days]
        test_end_idx = min(offset + train_days + test_days - 1, len(all_dates) - 1)
        test_end = all_dates[test_end_idx]

        print(f"\n--- Window {window_num} ---")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")

        # In-sample: full data access (walk_forward=False)
        is_result = run_backtest(
            cache_path=cache_path,
            start_date=train_start,
            end_date=train_end,
            min_entry_bars=min_entry_bars,
            verbose=False,
            walk_forward=False,
        )

        # Out-of-sample: walk-forward mode (only prior data)
        oos_result = run_backtest(
            cache_path=cache_path,
            start_date=test_start,
            end_date=test_end,
            min_entry_bars=min_entry_bars,
            verbose=False,
            walk_forward=True,
        )

        window = WalkForwardWindow(
            window_num=window_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            is_result=is_result,
            oos_result=oos_result,
        )
        report.windows.append(window)

        # Print window summary
        is_wr = (is_result.wins / is_result.trades_taken * 100) if is_result.trades_taken > 0 else 0
        oos_wr = (oos_result.wins / oos_result.trades_taken * 100) if oos_result.trades_taken > 0 else 0
        print(f"  IS:  {is_result.trades_taken} trades, {is_wr:.0f}% WR, {is_result.total_pnl:+.1f} pts")
        print(f"  OOS: {oos_result.trades_taken} trades, {oos_wr:.0f}% WR, {oos_result.total_pnl:+.1f} pts")

        # Advance by test_days (non-overlapping test windows)
        offset += test_days

    if not report.windows:
        print("No complete windows could be formed.")
        return report

    # Aggregate results
    report.aggregate_is = _merge_results([w.is_result for w in report.windows])
    report.aggregate_oos = _merge_results([w.oos_result for w in report.windows])

    # Calculate aggregate metrics
    is_wr = (report.aggregate_is.wins / report.aggregate_is.trades_taken * 100
             if report.aggregate_is.trades_taken > 0 else 0)
    oos_wr = (report.aggregate_oos.wins / report.aggregate_oos.trades_taken * 100
              if report.aggregate_oos.trades_taken > 0 else 0)

    report.degradation_ratio = oos_wr / is_wr if is_wr > 0 else 0.0

    is_pnls = [t["pnl"] for t in report.aggregate_is.trade_log]
    oos_pnls = [t["pnl"] for t in report.aggregate_oos.trade_log]
    report.is_sharpe = _calc_sharpe(is_pnls)
    report.oos_sharpe = _calc_sharpe(oos_pnls)

    # Robustness: composite score (0-100)
    # Factors: degradation ratio (40%), OOS profitability (30%), OOS Sharpe (30%)
    deg_score = min(report.degradation_ratio, 1.0) * 40  # max 40 pts
    profit_score = 30.0 if report.aggregate_oos.total_pnl > 0 else max(0, 15 + report.aggregate_oos.total_pnl)
    sharpe_score = min(max(report.oos_sharpe, 0) / 2.0, 1.0) * 30  # max 30 pts at Sharpe 2.0
    report.robustness_score = min(deg_score + profit_score + sharpe_score, 100)

    return report


def print_walk_forward_report(report: WalkForwardReport):
    """Print formatted walk-forward analysis report."""
    print()
    print("=" * 70)
    print("WALK-FORWARD ANALYSIS REPORT")
    print("=" * 70)

    if not report.windows:
        print("  No windows to report.")
        return

    # Per-window table
    print(f"\n  {'Win':>4s}  {'─── In-Sample ───':>20s}{'':>4s}{'─── Out-of-Sample ───':>22s}")
    print(f"  {'#':>4s}  {'Trades':>7s} {'WR%':>5s} {'P&L':>8s}  {'Trades':>7s} {'WR%':>5s} {'P&L':>8s}")
    print(f"  {'─'*4}  {'─'*7} {'─'*5} {'─'*8}  {'─'*7} {'─'*5} {'─'*8}")

    for w in report.windows:
        is_wr = (w.is_result.wins / w.is_result.trades_taken * 100
                 if w.is_result.trades_taken > 0 else 0)
        oos_wr = (w.oos_result.wins / w.oos_result.trades_taken * 100
                  if w.oos_result.trades_taken > 0 else 0)
        print(f"  {w.window_num:>4d}  {w.is_result.trades_taken:>7d} {is_wr:>5.0f} {w.is_result.total_pnl:>+8.1f}"
              f"  {w.oos_result.trades_taken:>7d} {oos_wr:>5.0f} {w.oos_result.total_pnl:>+8.1f}")

    # Aggregate
    is_r = report.aggregate_is
    oos_r = report.aggregate_oos
    is_wr = (is_r.wins / is_r.trades_taken * 100) if is_r.trades_taken > 0 else 0
    oos_wr = (oos_r.wins / oos_r.trades_taken * 100) if oos_r.trades_taken > 0 else 0

    print(f"\n  ─── Aggregate ───")
    print(f"  In-Sample:      {is_r.trades_taken} trades, {is_wr:.1f}% WR, "
          f"{is_r.total_pnl:+.1f} pts (${is_r.total_pnl * 50:+,.0f})")
    print(f"  Out-of-Sample:  {oos_r.trades_taken} trades, {oos_wr:.1f}% WR, "
          f"{oos_r.total_pnl:+.1f} pts (${oos_r.total_pnl * 50:+,.0f})")

    print(f"\n  ─── Overfit Detection ───")
    print(f"  Degradation Ratio:   {report.degradation_ratio:.2f}  "
          f"(OOS WR / IS WR — 1.00 = no degradation)")
    print(f"  IS Sharpe (ann):     {report.is_sharpe:.2f}")
    print(f"  OOS Sharpe (ann):    {report.oos_sharpe:.2f}")
    print(f"  IS Max Drawdown:     {is_r.max_drawdown:.1f} pts (${is_r.max_drawdown * 50:,.0f})")
    print(f"  OOS Max Drawdown:    {oos_r.max_drawdown:.1f} pts (${oos_r.max_drawdown * 50:,.0f})")
    print(f"  Robustness Score:    {report.robustness_score:.0f}/100")

    # Interpretation
    print(f"\n  ─── Interpretation ───")
    if report.degradation_ratio >= 0.85:
        print(f"  ✓ Minimal overfitting — OOS performance holds up well")
    elif report.degradation_ratio >= 0.65:
        print(f"  ⚠ Moderate degradation — some overfitting likely")
    else:
        print(f"  ✗ Significant degradation — strategy may be overfit to IS data")

    if report.aggregate_oos.total_pnl > 0:
        print(f"  ✓ OOS is profitable ({oos_r.total_pnl:+.1f} pts)")
    else:
        print(f"  ✗ OOS is not profitable ({oos_r.total_pnl:+.1f} pts)")

    if report.oos_sharpe > 0.5:
        print(f"  ✓ OOS Sharpe > 0.5 — acceptable risk-adjusted returns")
    elif report.oos_sharpe > 0:
        print(f"  ⚠ OOS Sharpe is low ({report.oos_sharpe:.2f}) — marginal edge")
    else:
        print(f"  ✗ OOS Sharpe is negative — no consistent edge detected")

    print()
    print("=" * 70)


# ─── CLI Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest ES fractal trading strategy")
    parser.add_argument("--cache", default="fractal_cache.db", help="Path to fractal cache DB")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-bars", type=int, default=12,
                        help="Minimum 5m bars before entry (default: 12 = 1 hour)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-trade details")
    parser.add_argument("--any-bar", action="store_true",
                        help="Scan from min-bars onward; enter at first bar where fractal fires")
    parser.add_argument("--entry-bar", type=int, default=None,
                        help="Fixed entry bar index (overrides default min-bars for entry)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-forward mode: only match against days before the test date")
    parser.add_argument("--replay", help="Replay a single day in detail (YYYY-MM-DD)")
    parser.add_argument("--fractal-report", action="store_true",
                        help="Print detailed fractal signal accuracy report")
    parser.add_argument("--walk-forward-report", action="store_true",
                        help="Run rolling walk-forward analysis with IS vs OOS comparison")
    parser.add_argument("--train-days", type=int, default=30,
                        help="Training window size in days (default: 30)")
    parser.add_argument("--test-days", type=int, default=10,
                        help="Test window size in days (default: 10)")
    args = parser.parse_args()

    if args.walk_forward_report:
        report = run_walk_forward(
            cache_path=Path(args.cache),
            start_date=args.start,
            end_date=args.end,
            train_days=args.train_days,
            test_days=args.test_days,
            min_entry_bars=args.min_bars,
        )
        print_walk_forward_report(report)
        sys.exit(0)

    if args.replay:
        result = run_backtest(
            cache_path=Path(args.cache),
            start_date=args.replay,
            end_date=args.replay,
            min_entry_bars=args.min_bars,
            verbose=True,
            walk_forward=args.walk_forward if hasattr(args, 'walk_forward') else False,
        )
        if result.trade_log:
            t = result.trade_log[0]
            print(f"\n--- REPLAY DETAIL: {args.replay} ---")
            print(f"  Verdict: {t['verdict']} @ {t['conf']}% confidence")
            print(f"  Entry:   {t['entry']:.2f}")
            print(f"  Target:  {t['target']:.2f}")
            print(f"  Stop:    {t['stop']:.2f}")
            print(f"  Outcome: {t['outcome']} | P&L: {t['pnl']:+.2f} pts")
            print(f"  Bars held: {t['bars_held']}")
            print(f"  Exit at: {t['exit_price']:.2f}")
        else:
            print(f"\n--- REPLAY: {args.replay} ---")
            print("  No trade generated for this day.")
        if result.signal_log:
            s = result.signal_log[0]
            print(f"\n  --- Signals ---")
            print(f"  Fractal: {s.get('fractal_direction', 'N/A')} ({s.get('fractal_confidence', 0)}%)")
            print(f"  Regime:  {s.get('regime', 'N/A')} (VIX: {s.get('vix', 0):.1f})")
            print(f"  Slippage: {s.get('slippage', 0):.2f} pts")
        print_results(result)
        if args.fractal_report:
            print_fractal_report(result.signal_log)
        sys.exit(0)

    result = run_backtest(
        cache_path=Path(args.cache),
        start_date=args.start,
        end_date=args.end,
        min_entry_bars=args.min_bars,
        verbose=args.verbose,
        any_bar=args.any_bar,
        entry_bar=args.entry_bar,
        walk_forward=args.walk_forward,
    )
    print_results(result)
    if args.fractal_report:
        print_fractal_report(result.signal_log)
