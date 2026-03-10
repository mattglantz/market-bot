"""
Bot configuration, logging, and shared utilities.

This is the foundation module — imported by all other modules.
Contains: ET timezone, now_et(), logger, Config/CFG, HTTP session, shared locks.
"""

import os
import sys
import logging
import logging.handlers
import threading
import uuid
import functools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Timezone ---
ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    """Return the current time in US/Eastern, timezone-aware."""
    return datetime.now(ET)


# --- Load .env BEFORE anything else reads os.getenv() ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # Using system env vars only


# --- Logging ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_DIR / "bot.log",
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
_file_handler.suffix = "%Y%m%d"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        _file_handler,
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("MarketBot")

# Suppress verbose ib_insync callbacks (updatePortfolio, position, commissionReport)
logging.getLogger("ib_insync").setLevel(logging.WARNING)
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)
logging.getLogger("ib_insync.client").setLevel(logging.WARNING)


# --- Configuration ---
def _safe_float(env_key: str, default: float) -> float:
    try:
        return float(os.getenv(env_key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid env var {env_key}={os.getenv(env_key)!r}, using default {default}")
        return default

def _safe_int(env_key: str, default: int) -> int:
    try:
        return int(os.getenv(env_key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid env var {env_key}={os.getenv(env_key)!r}, using default {default}")
        return default


@dataclass
class Config:
    """All configuration in one place. Reads from environment variables."""
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "PASTE_YOUR_TOKEN_HERE")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "PASTE_YOUR_CHAT_ID_HERE")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    WINDOW_10M: str = "10 min"
    WINDOW_1H: str = "1h"
    WINDOW_1D: str = "Daily"
    TARGET_TICKER: str = "ES"

    DAILY_NEWS: list = field(default_factory=lambda: ["08:30", "10:00", "14:00"])

    ACCOUNT_SIZE: float = field(default_factory=lambda: _safe_float("ACCOUNT_SIZE", 50000))
    MAX_RISK_PCT: float = field(default_factory=lambda: _safe_float("MAX_RISK_PCT", 2.0))
    MAX_DAILY_LOSS: float = field(default_factory=lambda: _safe_float("MAX_DAILY_LOSS", -500.0))  # dollars (e.g., -500 = $500 max loss)
    TRAILING_STOP_TRIGGER: float = 10.0
    # Progressive trailing stop levels: list of (float_pnl_threshold, stop_offset_from_entry)
    # When float P&L >= threshold, move stop to entry + offset.
    # Must be sorted ascending by threshold.
    TRAILING_STOP_LEVELS: list = field(default_factory=lambda: [
        (10.0, 0.0),    # +10 pts → stop to entry (breakeven)
        (15.0, 5.0),    # +15 pts → stop to entry + 5
        (20.0, 10.0),   # +20 pts → stop to entry + 10
        (25.0, 15.0),   # +25 pts → stop to entry + 15
    ])
    POINT_VALUE: float = 50.0   # ES = $50/point, NQ = $20/point

    ACTIVE_INTERVAL: int = 600
    OVERNIGHT_INTERVAL: int = 1800
    HEARTBEAT_INTERVAL: int = 3600

    DB_FILE: Path = Path("trading_journal.db")
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    IBKR_HOST: str = os.getenv("IBKR_HOST", "127.0.0.1")
    IBKR_PORT: int = field(default_factory=lambda: _safe_int("IBKR_PORT", 7497))
    IBKR_CLIENT_ID: int = field(default_factory=lambda: _safe_int("IBKR_CLIENT_ID", 10))

    ES_CONTRACT_RISK: float = field(default_factory=lambda: _safe_float("ES_CONTRACT_RISK", 250))

    BREADTH_TICKERS: str = "NVDA AAPL MSFT AMZN META GOOGL TSLA AVGO"

    # --- Instrument Profiles ---
    # Each instrument has its own point value and contract risk.
    # This is groundwork for multi-instrument support.
    INSTRUMENTS: dict = field(default_factory=lambda: {
        "ES": {"point_value": 50.0, "contract_risk": 250, "exchange": "CME", "tick_size": 0.25},
        "NQ": {"point_value": 20.0, "contract_risk": 400, "exchange": "CME", "tick_size": 0.25},
        "YM": {"point_value": 5.0, "contract_risk": 150, "exchange": "CBOT", "tick_size": 1.0},
        "RTY": {"point_value": 50.0, "contract_risk": 200, "exchange": "CME", "tick_size": 0.10},
    })
    ACTIVE_INSTRUMENT: str = "ES"  # Currently traded instrument

    # --- Tunable parameters ---
    VIX_THRESHOLDS: list = field(default_factory=lambda: [27, 20, 14])
    FLAT_THRESHOLDS: dict = field(default_factory=lambda: {
        "HIGH": 65, "ELEVATED": 60, "NORMAL": 60, "LOW": 55
    })
    RR_MINIMUM: float = 1.2
    POSITION_TIERS: list = field(default_factory=lambda: [
        (90, 4), (80, 3), (70, 2), (60, 1)
    ])
    MAX_OPEN_TRADES: int = 2
    TIME_EXIT_SECONDS: int = 7200
    FRACTAL_TIME_BUDGET: float = 15.0
    OPTIONS_CACHE_TTL: int = 180
    CHART_CLEANUP_DAYS: int = 30
    BACKTEST_SLIPPAGE_NORMAL: float = 0.25
    BACKTEST_SLIPPAGE_HIGH_VOL: float = 0.75
    HEALTH_METRICS_DB: Path = Path("health_metrics.db")
    RECONNECT_BACKOFF_MAX: int = 300
    RECONNECT_BACKOFF_BASE: int = 30
    ACCURACY_CACHE_TTL: int = 60
    STOP_CLUSTER_MAX_DIST: float = 30.0  # Max distance (pts) for stop cluster detection
    STOP_CLUSTER_MIN_DIST: float = 2.0   # Min distance (pts) to exclude current price noise

    # --- Shadow Mode ---
    # Set SHADOW_ENABLED=1 and SHADOW_PARAMS='{"flat_threshold": 55}' in .env
    SHADOW_ENABLED: bool = (os.getenv("SHADOW_ENABLED", "0") == "1")
    SHADOW_PARAMS: dict = field(default_factory=lambda: _parse_shadow_params())


def _parse_shadow_params() -> dict:
    """Parse SHADOW_PARAMS from env var (JSON string)."""
    raw = os.getenv("SHADOW_PARAMS", "")
    if not raw:
        return {}
    try:
        import json
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


CFG = Config()

# --- Hot-Reloadable Config ---
CONFIG_OVERRIDES_FILE = Path("config_overrides.json")

# Fields that can be hot-reloaded at runtime (safe to change without restart)
_RELOADABLE_FIELDS = {
    "FLAT_THRESHOLDS": dict,
    "RR_MINIMUM": float,
    "MAX_DAILY_LOSS": float,
    "POSITION_TIERS": list,
    "MAX_OPEN_TRADES": int,
    "TIME_EXIT_SECONDS": int,
    "TRAILING_STOP_LEVELS": list,
}


def reload_config() -> dict:
    """Re-read config_overrides.json and apply changes to CFG.

    Returns dict of {field: new_value} for fields that changed.
    """
    import json
    if not CONFIG_OVERRIDES_FILE.exists():
        return {}
    try:
        with open(CONFIG_OVERRIDES_FILE, "r") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {CONFIG_OVERRIDES_FILE}: {e}")
        return {}

    changes = {}
    for key, val in overrides.items():
        if key not in _RELOADABLE_FIELDS:
            continue
        expected_type = _RELOADABLE_FIELDS[key]
        try:
            if expected_type == float:
                val = float(val)
            elif expected_type == int:
                val = int(val)
            # dict and list are already the right type from JSON
        except (ValueError, TypeError):
            logger.warning(f"Config override {key}={val!r} — invalid type, skipping")
            continue

        old_val = getattr(CFG, key, None)
        if old_val != val:
            setattr(CFG, key, val)
            changes[key] = val
            logger.info(f"Config reload: {key} = {old_val!r} → {val!r}")

    return changes


def save_config_override(key: str, value) -> bool:
    """Persist a single config override to config_overrides.json."""
    import json
    if key not in _RELOADABLE_FIELDS:
        return False
    overrides = {}
    if CONFIG_OVERRIDES_FILE.exists():
        try:
            with open(CONFIG_OVERRIDES_FILE, "r") as f:
                overrides = json.load(f)
        except (json.JSONDecodeError, OSError):
            overrides = {}
    overrides[key] = value
    try:
        with open(CONFIG_OVERRIDES_FILE, "w") as f:
            json.dump(overrides, f, indent=2)
        return True
    except OSError as e:
        logger.warning(f"Failed to write {CONFIG_OVERRIDES_FILE}: {e}")
        return False


# --- Robust HTTP Session ---
def create_robust_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


HTTP = create_robust_session()


# --- Shared threading lock (used by market_data, indicators, session_utils) ---
_cache_lock = threading.Lock()


# --- safe_fetch decorator for exception logging (#1) ---
def safe_fetch(func):
    """Decorator that catches and logs exceptions, returning None on failure."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception(f"safe_fetch: exception in {func.__qualname__}")
            return None
    return wrapper


# --- Correlation ID support (#7) ---
def generate_cycle_id() -> str:
    """Generate a short unique correlation ID for a bot cycle."""
    return uuid.uuid4().hex[:12]
