"""
Backtest Weight Comparison — Current vs Proposed Fractal Weights
================================================================
Runs the same backtest with two different weight configurations to compare
which produces better trading outcomes.

Usage:
    python backtest_weights.py [--walk-forward] [--verbose] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np

# Must import before monkey-patching
import fractal_engine
from backtest import run_backtest, BacktestResult


WEIGHT_SETS = {
    "CURRENT": {
        "price_corr": 0.18, "dtw": 0.15, "momentum": 0.12, "atr_norm": 0.10,
        "wick_pattern": 0.08, "body_sequence": 0.08, "bar_range_match": 0.06,
        "segment_match": 0.06, "vol_shape": 0.06, "range_pattern": 0.06, "vol_match": 0.05,
    },
    "PROPOSED": {
        "price_corr": 0.15, "dtw": 0.10, "momentum": 0.08, "atr_norm": 0.10,
        "wick_pattern": 0.10, "body_sequence": 0.08, "bar_range_match": 0.06,
        "segment_match": 0.12, "vol_shape": 0.10, "range_pattern": 0.06, "vol_match": 0.05,
    },
}


def patch_weights(new_weights: dict):
    """Monkey-patch score_similarity to use different weights."""
    original_fn = fractal_engine.score_similarity.__wrapped__ \
        if hasattr(fractal_engine.score_similarity, '__wrapped__') \
        else fractal_engine.score_similarity

    def patched_score_similarity(today, hist, hist_partial, outcome_bonus=0.0, context_bonus=0.0):
        scores = {}
        n_bars = min(len(today.price_shape), len(hist_partial.price_shape))
        if n_bars < 6:
            return {"composite": 0.0}

        recency = np.exp(np.linspace(0, 1.1, n_bars))
        recency /= recency.sum()

        t_p = today.price_shape[:n_bars]
        h_p = hist_partial.price_shape[:n_bars]
        scores["price_corr"] = max(0, fractal_engine._weighted_corr(t_p, h_p, recency))

        try:
            scores["dtw"] = max(0, 1.0 - fractal_engine.dtw_distance(t_p, h_p) * 2)
        except Exception:
            scores["dtw"] = 0.0

        n_m = min(len(today.momentum_curve), len(hist_partial.momentum_curve))
        if n_m >= 6:
            rec_m = np.exp(np.linspace(0, 1.1, n_m)); rec_m /= rec_m.sum()
            scores["momentum"] = max(0, fractal_engine._weighted_corr(
                today.momentum_curve[:n_m], hist_partial.momentum_curve[:n_m], rec_m))
        else:
            scores["momentum"] = 0.0

        n_v = min(len(today.volume_shape), len(hist_partial.volume_shape))
        if n_v >= 6:
            scores["vol_shape"] = max(0, fractal_engine._safe_corr(today.volume_shape[:n_v], hist_partial.volume_shape[:n_v]))
        else: scores["vol_shape"] = 0.0

        n_r = min(len(today.range_expansion), len(hist_partial.range_expansion))
        if n_r >= 6:
            scores["range_pattern"] = max(0, fractal_engine._safe_corr(today.range_expansion[:n_r], hist_partial.range_expansion[:n_r]))
        else: scores["range_pattern"] = 0.0

        if today.volatility > 0 and np.isfinite(today.volatility) and np.isfinite(hist_partial.volatility):
            scores["vol_match"] = max(0, 1.0 - abs(today.volatility - hist_partial.volatility) / today.volatility)
        else:
            scores["vol_match"] = 0.5

        n_w = min(len(today.wick_ratios), len(hist_partial.wick_ratios))
        if n_w >= 6:
            scores["wick_pattern"] = max(0, fractal_engine._safe_corr(today.wick_ratios[:n_w], hist_partial.wick_ratios[:n_w]))
        else: scores["wick_pattern"] = 0.0

        n_b = min(len(today.body_directions), len(hist_partial.body_directions))
        if n_b >= 6:
            scores["body_sequence"] = float(np.mean(today.body_directions[:n_b] == hist_partial.body_directions[:n_b]))
        else:
            scores["body_sequence"] = 0.0

        n_br = min(len(today.bar_range_curve), len(hist_partial.bar_range_curve))
        if n_br >= 6:
            scores["bar_range_match"] = max(0, fractal_engine._safe_corr(today.bar_range_curve[:n_br], hist_partial.bar_range_curve[:n_br]))
        else: scores["bar_range_match"] = 0.0

        n_a = min(len(today.atr_norm_curve), len(hist_partial.atr_norm_curve))
        if n_a >= 6:
            scores["atr_norm"] = max(0, fractal_engine._safe_corr(today.atr_norm_curve[:n_a], hist_partial.atr_norm_curve[:n_a]))
        else: scores["atr_norm"] = 0.0

        _seg_weights = {"open_drive": 1.0, "am_trend": 1.2, "lunch": 0.8, "pm_trend": 1.5, "close": 2.0}
        seg_scores = []
        seg_w = []
        for seg_name in fractal_engine.SEGMENTS:
            t_seg = today.segment_shapes.get(seg_name)
            h_seg = hist_partial.segment_shapes.get(seg_name)
            if t_seg is not None and h_seg is not None:
                n_s = min(len(t_seg), len(h_seg))
                if n_s >= 3:
                    seg_scores.append(max(0, fractal_engine._safe_corr(t_seg[:n_s], h_seg[:n_s])))
                    seg_w.append(_seg_weights.get(seg_name, 1.0))
        if seg_scores:
            scores["segment_match"] = float(np.average(seg_scores, weights=seg_w))
        else:
            scores["segment_match"] = 0.0

        # USE THE PATCHED WEIGHTS
        composite = sum(scores.get(k, 0) * w for k, w in new_weights.items())
        composite += context_bonus + outcome_bonus
        scores["composite"] = min(0.99, composite)
        scores["context_bonus"] = context_bonus
        scores["outcome_bonus"] = outcome_bonus
        return scores

    fractal_engine.score_similarity = patched_score_similarity


def print_comparison(results: dict[str, BacktestResult]):
    """Print side-by-side comparison of weight sets."""
    print()
    print("=" * 80)
    print("WEIGHT COMPARISON RESULTS")
    print("=" * 80)

    header = f"{'Metric':<30}"
    for name in results:
        header += f"  {name:>18}"
    print(header)
    print("-" * 80)

    def row(label, fn):
        line = f"{label:<30}"
        for name, r in results.items():
            val = fn(r)
            line += f"  {val:>18}"
        print(line)

    row("Total days", lambda r: str(r.total_days))
    row("Days with signal", lambda r: str(r.days_with_signal))
    row("Trades taken", lambda r: str(r.trades_taken))

    print()
    row("Wins", lambda r: str(r.wins))
    row("Losses", lambda r: str(r.losses))
    row("Win Rate",
        lambda r: f"{r.wins/r.trades_taken*100:.1f}%" if r.trades_taken else "N/A")

    print()
    row("Total P&L (pts)",
        lambda r: f"{r.total_pnl:+.2f}")
    row("Total P&L ($)",
        lambda r: f"${r.total_pnl * 50:+,.0f}")
    row("Avg Win",
        lambda r: f"+{r.gross_wins/r.wins:.2f}" if r.wins else "N/A")
    row("Avg Loss",
        lambda r: f"-{r.gross_losses/r.losses:.2f}" if r.losses else "N/A")
    row("Profit Factor",
        lambda r: f"{r.gross_wins/r.gross_losses:.2f}" if r.gross_losses > 0 else "inf")

    print()
    row("Max Drawdown (pts)",
        lambda r: f"{r.max_drawdown:.2f}")
    row("Stops", lambda r: str(r.stops))
    row("Time Exits", lambda r: str(r.time_exits))

    # Sharpe
    def calc_sharpe(r):
        if not r.trade_log or len(r.trade_log) < 2:
            return "N/A"
        pnls = [t["pnl"] for t in r.trade_log]
        std = np.std(pnls, ddof=1)
        if std == 0:
            return "N/A"
        return f"{(np.mean(pnls) / std) * np.sqrt(252):.2f}"

    print()
    row("Sharpe (annualized)", calc_sharpe)

    # Direction breakdown
    print()
    for direction in ["BULL", "BEAR"]:
        label = "BULLISH" if direction == "BULL" else "BEARISH"
        def dir_stats(r, d=direction):
            trades = [t for t in r.trade_log if d in t["verdict"]]
            if not trades:
                return "0 trades"
            wins = sum(1 for t in trades if t["pnl"] > 0)
            pnl = sum(t["pnl"] for t in trades)
            return f"{wins}/{len(trades)} ({wins/len(trades)*100:.0f}%) {pnl:+.1f}"
        row(f"  {label}", dir_stats)

    print()
    print("=" * 80)

    # Weight diff summary
    print("\nWEIGHT DIFFERENCES:")
    print(f"  {'Feature':<20} {'CURRENT':>10} {'PROPOSED':>10} {'Delta':>10}")
    print("  " + "-" * 52)
    for key in WEIGHT_SETS["CURRENT"]:
        cur = WEIGHT_SETS["CURRENT"][key]
        prop = WEIGHT_SETS["PROPOSED"][key]
        delta = prop - cur
        marker = " *" if abs(delta) > 0.001 else ""
        print(f"  {key:<20} {cur:>10.2f} {prop:>10.2f} {delta:>+10.2f}{marker}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fractal weight configurations")
    parser.add_argument("--cache", default="fractal_cache.db")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--entry-bar", type=int, default=None)
    parser.add_argument("--any-bar", action="store_true")
    args = parser.parse_args()

    all_results = {}

    for name, weights in WEIGHT_SETS.items():
        print(f"\n{'='*40}")
        print(f"  Running backtest: {name} weights")
        print(f"{'='*40}")

        # Patch the weights
        patch_weights(weights)

        result = run_backtest(
            cache_path=Path(args.cache),
            start_date=args.start,
            end_date=args.end,
            verbose=args.verbose,
            walk_forward=args.walk_forward,
            entry_bar=args.entry_bar,
            any_bar=args.any_bar,
        )
        all_results[name] = result

    print_comparison(all_results)
