"""
Health metrics collection for bot performance monitoring.

Records per-cycle metrics to a SQLite database for trend analysis.
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from bot_config import logger, CFG, now_et


class HealthMetrics:
    """Records per-cycle performance metrics to SQLite for trend analysis."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or CFG.HEALTH_METRICS_DB
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
                CREATE TABLE IF NOT EXISTS cycle_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id        TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    ibkr_latency_ms REAL DEFAULT 0,
                    claude_latency_ms REAL DEFAULT 0,
                    fractal_latency_ms REAL DEFAULT 0,
                    total_cycle_ms  REAL DEFAULT 0,
                    data_source     TEXT DEFAULT 'unknown',
                    verdict         TEXT DEFAULT 'N/A',
                    confidence      INTEGER DEFAULT 0,
                    errors          INTEGER DEFAULT 0,
                    error_details   TEXT DEFAULT NULL
                )
            """)
            conn.commit()
        logger.info(f"HealthMetrics DB ready at {self.db_path}")

    def record_cycle(self, cycle_id: str, ibkr_ms: float = 0, claude_ms: float = 0,
                     fractal_ms: float = 0, total_ms: float = 0, data_source: str = "unknown",
                     verdict: str = "N/A", confidence: int = 0, errors: int = 0,
                     error_details: str = None):
        """Record metrics for one analysis cycle."""
        ts = now_et().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO cycle_metrics "
                    "(cycle_id, timestamp, ibkr_latency_ms, claude_latency_ms, fractal_latency_ms, "
                    "total_cycle_ms, data_source, verdict, confidence, errors, error_details) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cycle_id, ts, ibkr_ms, claude_ms, fractal_ms, total_ms,
                     data_source, verdict, confidence, errors, error_details),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"HealthMetrics record failed: {e}")

    def get_recent_metrics(self, hours: int = 24) -> list:
        """Get metrics from the last N hours."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM cycle_metrics WHERE timestamp >= datetime('now', ?) "
                    "ORDER BY id DESC LIMIT 500",
                    (f"-{hours} hours",),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"HealthMetrics query failed: {e}")
            return []

    def get_summary(self, hours: int = 24) -> dict:
        """Get aggregate stats for the last N hours."""
        metrics = self.get_recent_metrics(hours)
        if not metrics:
            return {"cycles": 0, "avg_cycle_ms": 0, "avg_claude_ms": 0,
                    "avg_fractal_ms": 0, "total_errors": 0}

        cycles = len(metrics)
        avg_cycle = sum(m["total_cycle_ms"] for m in metrics) / cycles
        avg_claude = sum(m["claude_latency_ms"] for m in metrics) / cycles
        avg_fractal = sum(m["fractal_latency_ms"] for m in metrics) / cycles
        total_errors = sum(m["errors"] for m in metrics)

        return {
            "cycles": cycles,
            "avg_cycle_ms": round(avg_cycle, 0),
            "avg_claude_ms": round(avg_claude, 0),
            "avg_fractal_ms": round(avg_fractal, 0),
            "total_errors": total_errors,
        }

    def cleanup(self, days: int = 30):
        """Remove metrics older than N days."""
        try:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM cycle_metrics WHERE timestamp < datetime('now', ?)",
                    (f"-{days} days",),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"HealthMetrics cleanup failed: {e}")
