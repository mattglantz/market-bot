"""
SQLite trading journal — thread-safe trade logging and statistics.

Extracted from market_bot_v26.py.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from bot_config import logger, CFG, now_et
import trade_status as ts

# Event type constants for trade_events table
EVT_OPENED = "OPENED"
EVT_STATUS_CHANGE = "STATUS_CHANGE"
EVT_STOP_MOVED = "STOP_MOVED"
EVT_PARTIAL_CLOSE = "PARTIAL_CLOSE"
EVT_PNL_UPDATE = "PNL_UPDATE"
EVT_CLOSED = "CLOSED"


class Journal:
    """Thread-safe SQLite trading journal — replaces the fragile CSV approach."""

    def __init__(self, db_path: Path = CFG.DB_FILE):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    price       REAL    NOT NULL,
                    verdict     TEXT    NOT NULL,
                    confidence  INTEGER NOT NULL DEFAULT 0,
                    target      REAL,
                    stop        REAL,
                    status      TEXT    NOT NULL DEFAULT '{ts.OPEN}',
                    pnl         REAL    NOT NULL DEFAULT 0.0,
                    contracts   INTEGER NOT NULL DEFAULT 1,
                    reasoning   TEXT,
                    session     TEXT,
                    fractal_recorded INTEGER DEFAULT NULL,
                    oca_group   TEXT    DEFAULT NULL
                )
            """)
            # Migration: add columns if DB predates these features
            for col_sql in [
                "ALTER TABLE trades ADD COLUMN fractal_recorded INTEGER DEFAULT NULL",
                "ALTER TABLE trades ADD COLUMN oca_group TEXT DEFAULT NULL",
                "ALTER TABLE trades ADD COLUMN contracts INTEGER NOT NULL DEFAULT 1",
                "ALTER TABLE trades ADD COLUMN signals TEXT DEFAULT NULL",
                "ALTER TABLE trades ADD COLUMN partial_closed INTEGER DEFAULT 0",
                "ALTER TABLE trades ADD COLUMN stop_updated_at TEXT DEFAULT NULL",
            ]:
                try:
                    conn.execute(col_sql)
                    conn.commit()
                except Exception:
                    pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date        TEXT PRIMARY KEY,
                    trades      INTEGER DEFAULT 0,
                    wins        INTEGER DEFAULT 0,
                    losses      INTEGER DEFAULT 0,
                    total_pnl   REAL    DEFAULT 0.0,
                    max_drawdown REAL   DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_reviews (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT NOT NULL,
                    trade_id    INTEGER,
                    lesson      TEXT,
                    summary     TEXT,
                    timestamp   TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_scores (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id        INTEGER NOT NULL,
                    signal_time     TEXT NOT NULL,
                    price_at_signal REAL NOT NULL,
                    verdict         TEXT NOT NULL,
                    price_30m       REAL DEFAULT NULL,
                    price_60m       REAL DEFAULT NULL,
                    price_120m      REAL DEFAULT NULL,
                    max_favorable   REAL DEFAULT NULL,
                    max_adverse     REAL DEFAULT NULL,
                    UNIQUE(trade_id)
                )
            """)
            # Event sourcing: append-only audit trail for trade lifecycle
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id    INTEGER NOT NULL,
                    event_type  TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    data        TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_events_trade "
                "ON trade_events(trade_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_events_ts "
                "ON trade_events(timestamp)"
            )
            conn.commit()
        logger.info(f"Journal DB ready at {self.db_path}")

    def _emit_event(self, conn, trade_id: int, event_type: str,
                    before: dict = None, after: dict = None,
                    context: dict = None):
        """Append an immutable event to trade_events. Called inside existing transactions."""
        data = json.dumps({
            "before": before or {},
            "after": after or {},
            "context": context or {},
        })
        conn.execute(
            "INSERT INTO trade_events (trade_id, event_type, timestamp, data) "
            "VALUES (?, ?, ?, ?)",
            (trade_id, event_type, now_et().strftime("%Y-%m-%d %H:%M:%S"), data),
        )

    def add_trade(self, price: float, verdict: str, confidence: int,
                  target: float, stop: float, contracts: int = 1,
                  reasoning: str = "",
                  session: str = "", oca_group: str = None,
                  signals: dict = None) -> int:
        trade_ts = now_et().strftime("%Y-%m-%d %H:%M")
        signals_json = json.dumps(signals) if signals else None
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO trades (timestamp, price, verdict, confidence, target, stop, contracts, status, pnl, reasoning, session, oca_group, signals) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, '{ts.OPEN}', 0.0, ?, ?, ?, ?)",
                (trade_ts, price, verdict, confidence, target, stop, max(contracts, 1), reasoning, session, oca_group, signals_json),
            )
            trade_id = cur.lastrowid
            self._emit_event(conn, trade_id, EVT_OPENED, after={
                "price": price, "verdict": verdict, "confidence": confidence,
                "target": target, "stop": stop, "contracts": max(contracts, 1),
                "status": ts.OPEN,
            })
            conn.commit()
        logger.info(f"Trade logged: {verdict} {contracts}x @ {price} | T:{target} S:{stop} | Conf:{confidence}%")
        return trade_id

    def add_skipped_trade(self, price: float, verdict: str, confidence: int,
                          target: float, stop: float, contracts: int = 1,
                          reason: str = "", signals: dict = None) -> int:
        """Log a trade that was filtered out (R:R gate, etc.) for phantom P&L tracking."""
        trade_ts = now_et().strftime("%Y-%m-%d %H:%M")
        signals_json = json.dumps(signals) if signals else None
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO trades (timestamp, price, verdict, confidence, target, stop, contracts, status, pnl, reasoning, session, signals) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, '{ts.SKIPPED}', 0.0, ?, 'SKIPPED', ?)",
                (trade_ts, price, verdict, confidence, target, stop, max(contracts, 1), reason, signals_json),
            )
            conn.commit()
            trade_id = cur.lastrowid
        logger.info(f"Skipped trade logged: {verdict} @ {price} | T:{target} S:{stop} | Reason: {reason}")
        return trade_id

    def get_skipped_trades_today(self) -> list:
        """Get today's skipped trades for phantom P&L reporting."""
        today = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE timestamp LIKE ? AND status = '{ts.SKIPPED}' ORDER BY id",
                (f"{today}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE status IN ('{ts.OPEN}', '{ts.FLOATING}') ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_trade(self, trade_id: int, status: str, pnl: float,
                     stop: float = None, contracts: int = None,
                     target: float = None, stop_updated_at: str = None):
        with self._conn() as conn:
            # Capture before-state for event sourcing
            before_row = conn.execute(
                "SELECT status, pnl, stop, contracts, target FROM trades WHERE id=?",
                (trade_id,),
            ).fetchone()
            before = dict(before_row) if before_row else {}

            fields = ["status=?", "pnl=?"]
            params = [status, round(pnl, 2)]
            if stop is not None:
                fields.append("stop=?")
                params.append(stop)
            if contracts is not None:
                fields.append("contracts=?")
                params.append(contracts)
            if target is not None:
                fields.append("target=?")
                params.append(target)
            if stop_updated_at is not None:
                fields.append("stop_updated_at=?")
                params.append(stop_updated_at)
            params.append(trade_id)
            conn.execute(
                f"UPDATE trades SET {', '.join(fields)} WHERE id=?",
                params,
            )

            # Build after-state and determine event type
            after = {"status": status, "pnl": round(pnl, 2)}
            if stop is not None:
                after["stop"] = stop
            if contracts is not None:
                after["contracts"] = contracts
            if target is not None:
                after["target"] = target

            prev_status = before.get("status", "")
            if status in ts.CLOSED_STATUSES and prev_status not in ts.CLOSED_STATUSES:
                evt = EVT_CLOSED
            elif stop is not None and stop != before.get("stop"):
                evt = EVT_STOP_MOVED
            elif contracts is not None and contracts != before.get("contracts"):
                evt = EVT_PARTIAL_CLOSE
            elif status != prev_status:
                evt = EVT_STATUS_CHANGE
            else:
                evt = EVT_PNL_UPDATE

            self._emit_event(conn, trade_id, evt, before=before, after=after)
            conn.commit()

    # --- Event Sourcing Queries ---

    def get_trade_timeline(self, trade_id: int) -> list:
        """Return full event history for a trade, oldest-first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_events WHERE trade_id=? ORDER BY id ASC",
                (trade_id,),
            ).fetchall()
        result = []
        for r in rows:
            try:
                data = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError):
                data = {}
            result.append({
                "id": r["id"],
                "event_type": r["event_type"],
                "timestamp": r["timestamp"],
                "data": data,
            })
        return result

    def replay_trade(self, trade_id: int) -> dict:
        """Reconstruct trade state from events. Returns final state dict."""
        events = self.get_trade_timeline(trade_id)
        state = {}
        for evt in events:
            after = evt["data"].get("after", {})
            state.update(after)
        return state

    def get_last_verdict(self) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT verdict FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["verdict"] if row else "NONE"

    def get_today_stats(self) -> dict:
        today = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE timestamp LIKE ?", (f"{today}%",)
            ).fetchall()
        trades = [dict(r) for r in rows if r["status"] != ts.SKIPPED]
        closed = [t for t in trades if t["status"] in ts.CLOSED_STATUSES]
        floating = [t for t in trades if t["status"] == ts.FLOATING]
        wins = sum(1 for t in closed if t["status"] == ts.WIN)
        losses = sum(1 for t in closed if t["status"] in ts.LOSS_STATUSES)
        # Weight P&L by contracts: pnl column is per-contract points
        realized = sum(t["pnl"] * int(t.get("contracts", 1) or 1) for t in closed)
        float_pnl = sum(t["pnl"] * int(t.get("contracts", 1) or 1) for t in floating)
        return {
            "total": len(trades),
            "wins": wins,
            "losses": losses,
            "realized": realized,
            "floating": float_pnl,
            "win_rate": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
            "net": realized + float_pnl,
        }

    def get_weekly_stats(self, detailed: bool = False) -> dict:
        _now = now_et()
        week_start = (_now - timedelta(days=_now.weekday())).strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE timestamp >= ? AND status IN {ts.CLOSED_SQL}",
                (week_start,),
            ).fetchall()
            trades = [dict(r) for r in rows]
            wins = sum(1 for t in trades if t["status"] == ts.WIN)
            losses = sum(1 for t in trades if t["status"] in ts.LOSS_STATUSES)
            pnls = [t["pnl"] * int(t.get("contracts", 1) or 1) for t in trades]
            total_pnl = sum(pnls) if pnls else 0
            result = {"wins": wins, "losses": losses, "pnl": total_pnl, "total": len(trades)}

            if detailed and trades:
                result["best_trade_pts"] = max(pnls) if pnls else 0
                result["worst_trade_pts"] = min(pnls) if pnls else 0
                result["win_rate"] = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0
                result["expectancy_pts"] = round(total_pnl / len(pnls), 2) if pnls else 0
                # Prior week comparison
                prev_week_start = (_now - timedelta(days=_now.weekday() + 7)).strftime("%Y-%m-%d")
                prev_rows = conn.execute(
                    f"SELECT pnl, contracts, status FROM trades "
                    f"WHERE timestamp >= ? AND timestamp < ? AND status IN {ts.CLOSED_SQL}",
                    (prev_week_start, week_start),
                ).fetchall()
                prev_trades = [dict(r) for r in prev_rows]
                result["prev_week_pnl"] = sum(
                    t["pnl"] * int(t.get("contracts", 1) or 1) for t in prev_trades
                )
                result["prev_week_trades"] = len(prev_trades)
        return result

    def get_monthly_stats(self) -> dict:
        _now = now_et()
        month_start = _now.replace(day=1).strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE timestamp >= ? AND status IN {ts.CLOSED_SQL}",
                (month_start,),
            ).fetchall()
        trades = [dict(r) for r in rows]
        wins = sum(1 for t in trades if t["status"] == ts.WIN)
        losses = sum(1 for t in trades if t["status"] in ts.LOSS_STATUSES)
        pnls = [t["pnl"] * int(t.get("contracts", 1) or 1) for t in trades]
        total_pnl = sum(pnls) if pnls else 0
        return {
            "total": len(trades),
            "wins": wins,
            "losses": losses,
            "pnl": total_pnl,
            "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            "best_trade_pts": max(pnls) if pnls else 0,
            "worst_trade_pts": min(pnls) if pnls else 0,
            "expectancy_pts": round(total_pnl / len(pnls), 2) if pnls else 0,
        }

    def get_signal_performance(self, min_trades: int = 5) -> list:
        """Analyze win rates by signal combination from recorded signals."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT signals, status FROM trades "
                    f"WHERE signals IS NOT NULL AND status IN {ts.DECIDED_SQL} "
                    "ORDER BY id DESC LIMIT 500"
                ).fetchall()

            if not rows:
                return []

            combo_stats = {}
            for r in rows:
                try:
                    sigs = json.loads(r["signals"])
                except (json.JSONDecodeError, TypeError):
                    continue

                # Create a signature from key signal directions
                key_parts = []
                if sigs.get("fractal_dir"):
                    key_parts.append(f"F:{sigs['fractal_dir']}")
                if sigs.get("mtf"):
                    key_parts.append(f"MTF:{sigs['mtf']}")
                if sigs.get("flow"):
                    key_parts.append(f"Flow:{sigs['flow']}")
                if sigs.get("gex"):
                    key_parts.append(f"GEX:{sigs['gex']}")

                if not key_parts:
                    continue

                combo = " | ".join(sorted(key_parts))
                if combo not in combo_stats:
                    combo_stats[combo] = {"wins": 0, "total": 0}
                combo_stats[combo]["total"] += 1
                if r["status"] == ts.WIN:
                    combo_stats[combo]["wins"] += 1

            # Filter to combos with enough trades and compute win rate
            results = []
            for combo, stats in combo_stats.items():
                if stats["total"] >= min_trades:
                    wr = stats["wins"] / stats["total"] * 100
                    results.append({
                        "combo": combo,
                        "wins": stats["wins"],
                        "total": stats["total"],
                        "win_rate": round(wr, 1),
                    })

            return sorted(results, key=lambda x: x["win_rate"], reverse=True)
        except Exception as e:
            logger.warning(f"Signal performance query failed: {e}")
            return []

    # --- Post-Trade Review (Feature 2) ---

    def get_closed_trades_today(self) -> list:
        """Get all closed trades from today for post-trade review."""
        today = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE timestamp LIKE ? AND status IN {ts.CLOSED_SQL} ORDER BY id",
                (f"{today}%",),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_trade_review(self, date: str, reviews: list, summary: str):
        """Save Claude's post-trade review. reviews: [{trade_id, lesson}, ...]"""
        ts_now = now_et().strftime("%Y-%m-%d %H:%M")
        with self._conn() as conn:
            for r in reviews:
                conn.execute(
                    "INSERT INTO trade_reviews (date, trade_id, lesson, summary, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (date, r.get("trade_id"), r.get("lesson", ""), None, ts_now),
                )
            conn.execute(
                "INSERT INTO trade_reviews (date, trade_id, lesson, summary, timestamp) "
                "VALUES (?, NULL, NULL, ?, ?)",
                (date, summary, ts_now),
            )
            conn.commit()

    def get_trade_review(self, date: str) -> dict:
        """Retrieve the trade review for a given date."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT trade_id, lesson, summary FROM trade_reviews WHERE date = ? ORDER BY id",
                (date,),
            ).fetchall()
        if not rows:
            return {}
        lessons = []
        summary = ""
        for r in rows:
            if r["trade_id"] is not None:
                lessons.append({"trade_id": r["trade_id"], "lesson": r["lesson"]})
            if r["summary"]:
                summary = r["summary"]
        return {"lessons": lessons, "summary": summary}

    # --- Daily Stats ---

    def save_daily_stats(self, date_str: str = None):
        """Compute and persist daily stats from closed trades.

        Called at end-of-day recap. Overwrites if already exists for the date.
        """
        if date_str is None:
            date_str = now_et().strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT pnl, contracts, status FROM trades "
                f"WHERE timestamp LIKE ? AND status IN {ts.CLOSED_SQL}",
                (f"{date_str}%",),
            ).fetchall()
        trades = [dict(r) for r in rows]
        if not trades:
            return
        wins = sum(1 for t in trades if t["status"] == ts.WIN)
        losses = sum(1 for t in trades if t["status"] in ts.LOSS_STATUSES)
        pnls = [t["pnl"] * int(t.get("contracts", 1) or 1) for t in trades]
        total_pnl = sum(pnls)
        # Max drawdown: largest cumulative trough from peak
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO daily_stats "
                "(date, trades, wins, losses, total_pnl, max_drawdown) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (date_str, len(trades), wins, losses, total_pnl, max_dd),
            )
            conn.commit()
        logger.info(
            f"Daily stats saved: {date_str} | "
            f"{len(trades)} trades, {wins}W/{losses}L, "
            f"PnL={total_pnl:+.1f} pts, MaxDD={max_dd:.1f} pts"
        )

    # --- Signal Scoring (Feature 5) ---

    def add_signal_score(self, trade_id: int, signal_time: str,
                         price_at_signal: float, verdict: str):
        """Record a new signal for forward-looking scoring."""
        with self._conn() as conn:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO signal_scores "
                    "(trade_id, signal_time, price_at_signal, verdict) "
                    "VALUES (?, ?, ?, ?)",
                    (trade_id, signal_time, price_at_signal, verdict),
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"Signal score insert failed: {e}")

    def get_pending_signal_scores(self) -> list:
        """Get signal_scores rows still missing 30/60/120m prices."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_scores "
                "WHERE price_30m IS NULL OR price_60m IS NULL OR price_120m IS NULL "
                "ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_signal_score(self, score_id: int, **kwargs):
        """Update specific fields on a signal_scores row."""
        allowed = ("price_30m", "price_60m", "price_120m", "max_favorable", "max_adverse")
        fields, params = [], []
        for k, v in kwargs.items():
            if k in allowed:
                fields.append(f"{k}=?")
                params.append(v)
        if not fields:
            return
        params.append(score_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE signal_scores SET {', '.join(fields)} WHERE id=?", params,
            )
            conn.commit()

    def get_signal_quality_stats(self, days: int = 7) -> dict:
        """Compute signal quality statistics for the last N days."""
        cutoff = (now_et() - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signal_scores WHERE signal_time >= ? "
                "AND price_120m IS NOT NULL ORDER BY id",
                (cutoff,),
            ).fetchall()
        if not rows:
            return {"total": 0}
        scores = [dict(r) for r in rows]
        correct_30 = correct_60 = correct_120 = 0
        for s in scores:
            p0 = s["price_at_signal"]
            is_long = "BULL" in s["verdict"].upper()
            if is_long:
                if s["price_30m"] and s["price_30m"] > p0: correct_30 += 1
                if s["price_60m"] and s["price_60m"] > p0: correct_60 += 1
                if s["price_120m"] and s["price_120m"] > p0: correct_120 += 1
            else:
                if s["price_30m"] and s["price_30m"] < p0: correct_30 += 1
                if s["price_60m"] and s["price_60m"] < p0: correct_60 += 1
                if s["price_120m"] and s["price_120m"] < p0: correct_120 += 1
        n = len(scores)
        return {
            "total": n,
            "correct_30m_pct": round(correct_30 / n * 100, 1),
            "correct_60m_pct": round(correct_60 / n * 100, 1),
            "correct_120m_pct": round(correct_120 / n * 100, 1),
        }
