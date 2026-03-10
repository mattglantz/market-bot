"""
Shadow Mode / Paper Twin — runs alternate parameters alongside live pipeline.

Logs every cycle comparison (live vs shadow) to the `shadow_signals` table
for post-analysis. Never executes real trades. Uses deep-copied data to
avoid mutating live state.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from bot_config import logger, CFG, now_et


class ShadowMode:
    """Compare live pipeline output against a shadow parameter set."""

    def __init__(self, db_path: Path = None, shadow_params: dict = None):
        self.db_path = db_path or CFG.DB_FILE
        self.shadow_params = shadow_params or {}
        self.enabled = bool(self.shadow_params)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id        TEXT,
                    timestamp       TEXT NOT NULL,
                    price           REAL,

                    -- Live pipeline
                    live_verdict    TEXT,
                    live_confidence INTEGER,
                    live_contracts  INTEGER,

                    -- Shadow pipeline
                    shadow_verdict    TEXT,
                    shadow_confidence INTEGER,
                    shadow_contracts  INTEGER,

                    -- Comparison
                    verdict_match     INTEGER,
                    confidence_delta  INTEGER,

                    -- Shadow params used
                    params_json       TEXT,

                    -- Phantom trade tracking
                    shadow_entry      REAL,
                    shadow_target     REAL,
                    shadow_stop       REAL,
                    phantom_pnl       REAL,
                    phantom_outcome   TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_shadow_ts
                ON shadow_signals(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_shadow_cycle
                ON shadow_signals(cycle_id)
            """)
            conn.commit()

    def evaluate(self, cycle_id: str, price: float, data: dict,
                 metrics: dict, accuracy_tracker, regime: dict,
                 md, live_verdict: str, live_confidence: int,
                 live_contracts: int,
                 news_info: tuple = None) -> dict | None:
        """
        Run the confidence pipeline with shadow params and log comparison.

        Returns comparison dict, or None if shadow is disabled.
        """
        if not self.enabled:
            return None

        try:
            # Deep-copy to avoid mutating live state
            shadow_data = copy.deepcopy(data)
            shadow_metrics = copy.deepcopy(metrics)
            shadow_regime = copy.deepcopy(regime)

            # Apply shadow parameter overrides to regime/data
            self._apply_shadow_params(shadow_regime, shadow_data)

            # Run confidence pipeline with shadow params
            from confidence_engine import apply_confidence_pipeline
            (s_data, s_verdict, s_conf, s_contracts,
             _pos_str, _confluence, _decomp) = apply_confidence_pipeline(
                shadow_data, shadow_metrics, accuracy_tracker,
                shadow_regime, md, news_info=news_info,
            )

            # Determine shadow flat threshold
            s_flat = shadow_regime.get("flat_threshold", 60)

            # Shadow trade decision
            shadow_would_trade = (
                s_verdict not in ("FLAT", "NEUTRAL") and s_conf >= s_flat
            )

            # Extract shadow entry/target/stop
            shadow_entry = float(s_data.get("current_price", price)) if shadow_would_trade else None
            shadow_target = float(s_data.get("target", 0)) if shadow_would_trade else None
            shadow_stop = float(s_data.get("invalidation", 0)) if shadow_would_trade else None

            verdict_match = (live_verdict == s_verdict)
            confidence_delta = s_conf - live_confidence

            result = {
                "shadow_verdict": s_verdict,
                "shadow_confidence": s_conf,
                "shadow_contracts": s_contracts,
                "verdict_match": verdict_match,
                "confidence_delta": confidence_delta,
                "shadow_would_trade": shadow_would_trade,
                "shadow_entry": shadow_entry,
                "shadow_target": shadow_target,
                "shadow_stop": shadow_stop,
            }

            # Log to DB
            self._log(cycle_id, price, live_verdict, live_confidence,
                      live_contracts, result)

            return result

        except Exception as e:
            logger.warning(f"[SHADOW] Evaluation error: {e}")
            return None

    def _apply_shadow_params(self, regime: dict, data: dict):
        """Override regime/data values with shadow params."""
        # Override flat thresholds
        if "flat_threshold" in self.shadow_params:
            regime["flat_threshold"] = self.shadow_params["flat_threshold"]

        # Override VIX thresholds
        if "vix_thresholds" in self.shadow_params:
            regime["vix_thresholds"] = self.shadow_params["vix_thresholds"]

        # Override position tiers
        if "position_tiers" in self.shadow_params:
            regime["position_tiers"] = self.shadow_params["position_tiers"]

        # Override RR minimum
        if "rr_minimum" in self.shadow_params:
            regime["rr_minimum"] = self.shadow_params["rr_minimum"]

        # Override confidence directly (simulate a manual bump)
        if "confidence_adj" in self.shadow_params:
            raw = int(data.get("confidence", 0))
            data["confidence"] = raw + self.shadow_params["confidence_adj"]

    def _log(self, cycle_id: str, price: float,
             live_verdict: str, live_confidence: int, live_contracts: int,
             result: dict):
        """Persist comparison to shadow_signals table."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO shadow_signals (
                    cycle_id, timestamp, price,
                    live_verdict, live_confidence, live_contracts,
                    shadow_verdict, shadow_confidence, shadow_contracts,
                    verdict_match, confidence_delta, params_json,
                    shadow_entry, shadow_target, shadow_stop
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_id,
                now_et().strftime("%Y-%m-%d %H:%M:%S"),
                price,
                live_verdict,
                live_confidence,
                live_contracts,
                result["shadow_verdict"],
                result["shadow_confidence"],
                result["shadow_contracts"],
                1 if result["verdict_match"] else 0,
                result["confidence_delta"],
                json.dumps(self.shadow_params),
                result.get("shadow_entry"),
                result.get("shadow_target"),
                result.get("shadow_stop"),
            ))
            conn.commit()

    def update_phantom_pnl(self, current_price: float):
        """
        Update phantom P&L for shadow trades that haven't been resolved.

        Checks if phantom entry/target/stop would have been hit and
        updates phantom_pnl and phantom_outcome.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, shadow_verdict, shadow_entry, shadow_target, shadow_stop
                FROM shadow_signals
                WHERE shadow_entry IS NOT NULL
                  AND phantom_outcome IS NULL
                  AND timestamp >= date('now', '-1 day')
            """).fetchall()

            for row in rows:
                entry = row["shadow_entry"]
                target = row["shadow_target"]
                stop = row["shadow_stop"]
                verdict = row["shadow_verdict"]

                if not entry or not target or not stop:
                    continue

                long = verdict.upper() in ("LONG", "BULLISH", "BUY")
                pnl = (current_price - entry) if long else (entry - current_price)

                outcome = None
                if long:
                    if current_price >= target:
                        pnl = target - entry
                        outcome = "WIN"
                    elif current_price <= stop:
                        pnl = stop - entry
                        outcome = "LOSS"
                else:
                    if current_price <= target:
                        pnl = entry - target
                        outcome = "WIN"
                    elif current_price >= stop:
                        pnl = entry - stop
                        outcome = "LOSS"

                conn.execute("""
                    UPDATE shadow_signals
                    SET phantom_pnl = ?, phantom_outcome = ?
                    WHERE id = ?
                """, (round(pnl, 2), outcome, row["id"]))

            conn.commit()

    def get_comparison_stats(self, days: int = 7) -> dict:
        """Get shadow vs live comparison statistics."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM shadow_signals
                WHERE timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp
            """, (f"-{days}",)).fetchall()

        if not rows:
            return {
                "total_cycles": 0,
                "agreement_rate": 0.0,
                "avg_confidence_delta": 0.0,
                "shadow_trades": 0,
                "shadow_wins": 0,
                "shadow_losses": 0,
                "shadow_pnl": 0.0,
            }

        total = len(rows)
        matches = sum(1 for r in rows if r["verdict_match"])
        avg_delta = sum(r["confidence_delta"] for r in rows) / total

        shadow_trades = [r for r in rows if r["shadow_entry"] is not None]
        shadow_wins = sum(1 for r in shadow_trades if r["phantom_outcome"] == "WIN")
        shadow_losses = sum(1 for r in shadow_trades if r["phantom_outcome"] == "LOSS")
        shadow_pnl = sum(r["phantom_pnl"] or 0 for r in shadow_trades)

        return {
            "total_cycles": total,
            "agreement_rate": (matches / total) * 100 if total > 0 else 0,
            "avg_confidence_delta": round(avg_delta, 1),
            "shadow_trades": len(shadow_trades),
            "shadow_wins": shadow_wins,
            "shadow_losses": shadow_losses,
            "shadow_pnl": round(shadow_pnl, 2),
        }

    def format_telegram_report(self) -> str:
        """Format shadow mode stats for Telegram."""
        stats = self.get_comparison_stats(days=7)

        if stats["total_cycles"] == 0:
            return (
                "*\U0001f47b Shadow Mode*\n\n"
                "No shadow data yet. Enable with SHADOW\\_ENABLED=1 "
                "and set SHADOW\\_PARAMS in .env"
            )

        shadow_wr = "--"
        st = stats["shadow_trades"]
        resolved = stats["shadow_wins"] + stats["shadow_losses"]
        if resolved > 0:
            shadow_wr = f"{stats['shadow_wins'] / resolved * 100:.0f}%"

        pnl_d = stats["shadow_pnl"] * CFG.POINT_VALUE
        params_str = json.dumps(self.shadow_params, indent=None)
        if len(params_str) > 60:
            params_str = params_str[:57] + "..."

        return (
            f"*\U0001f47b Shadow Mode (7d)*\n"
            f"\n"
            f"`Cycles    ` {stats['total_cycles']}\n"
            f"`Agreement ` {stats['agreement_rate']:.0f}%\n"
            f"`Avg Δconf ` {stats['avg_confidence_delta']:+.1f}\n"
            f"\n"
            f"\u2500\u2500\u2500 Phantom Trades \u2500\u2500\u2500\n"
            f"`Trades    ` {st}\n"
            f"`Win Rate  ` {shadow_wr} ({stats['shadow_wins']}W/{stats['shadow_losses']}L)\n"
            f"`Phantom$  ` {'+'if pnl_d >= 0 else ''}{pnl_d:,.0f}\n"
            f"\n"
            f"`Params    ` `{params_str}`"
        )
