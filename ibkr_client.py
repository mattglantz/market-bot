"""
IBKR Client — Live Market Data via Interactive Brokers TWS/Gateway
===================================================================
Primary data source for the ES trading bot. Falls back to yfinance
if IBKR is not connected.

Requirements:
    pip install ib_insync

Setup:
    1. Open TWS or IB Gateway
    2. Enable API connections: File → Global Configuration → API → Settings
       - Enable ActiveX and Socket Clients
       - Socket port: 7497 (TWS paper) or 7496 (TWS live) or 4001/4002 (Gateway)
       - Allow connections from localhost
    3. Set IBKR_PORT in .env (default 7497 for paper trading)
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np

from bot_config import CFG

logger = logging.getLogger("MarketBot")

# Python 3.10+ removed auto-creation of event loops in non-async contexts.
# ib_insync / eventkit tries to get_event_loop() at import time.
# We must create one before importing.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock, Future, ContFuture, Index, Option, Forex, util
    from ib_insync.contract import Contract
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    logger.warning("ib_insync not installed. Run: pip install ib_insync")

# Fallback
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


class IBKRClient:
    """
    Manages connection to IBKR TWS/Gateway and provides market data.
    Auto-reconnects on failure. Falls back to yfinance if IBKR unavailable.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 10):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib: Optional[IB] = None
        self.connected = False
        self._contracts: Dict[str, Contract] = {}
        self._last_connect_attempt = None
        self._backoff_seconds = 30  # exponential backoff: 30 → 60 → 120 → 300 (cap)

        # Options chain cache
        self._options_cache: Optional[dict] = None
        self._options_cache_time: float = 0.0
        self._options_cache_price: float = 0.0

        if IBKR_AVAILABLE:
            self.connect()
        else:
            logger.warning("IBKR: ib_insync not available. Using yfinance fallback.")

    def connect(self) -> bool:
        """Connect to TWS/Gateway with exponential backoff. Returns True if successful."""
        if not IBKR_AVAILABLE:
            return False

        # Don't retry too frequently — use exponential backoff
        if self._last_connect_attempt:
            elapsed = (datetime.now() - self._last_connect_attempt).total_seconds()
            if elapsed < self._backoff_seconds:
                return self.connected

        self._last_connect_attempt = datetime.now()

        try:
            if self.ib and self.ib.isConnected():
                self.connected = True
                return True

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
            self.connected = True
            self._backoff_seconds = 30  # reset backoff on successful connection
            import time as _time
            self._last_health_check = _time.time()  # skip health check for first 5 min
            logger.info(f"IBKR connected on port {self.port}")

            # Pre-qualify key contracts
            self._setup_contracts()
            return True

        except Exception as e:
            logger.warning(f"IBKR connection failed (port {self.port}), "
                           f"next retry in {self._backoff_seconds}s: {e}")
            self.connected = False
            # Exponential backoff: 30 → 60 → 120 → 300 (cap)
            self._backoff_seconds = min(self._backoff_seconds * 2, 300)
            return False

    def _ensure_connected(self) -> bool:
        """Reconnect if needed. Also re-qualify contracts if near expiry.

        Runs a periodic health check (every 5 min) via reqCurrentTime()
        to detect stale TCP connections that report isConnected() = True.
        """
        if self.connected and self.ib and self.ib.isConnected():
            # Periodic health check — only every 5 minutes to avoid overhead
            import time as _time
            _now = _time.time()
            if _now - getattr(self, '_last_health_check', 0) > 300:
                try:
                    self.ib.reqCurrentTime()
                    self._last_health_check = _now
                except Exception as e:
                    logger.warning(f"IBKR health check failed (stale connection?): {e}")
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                    self.connected = False
                    return self.connect()
            # Check if front-month futures are near expiry and need re-qualification
            self._check_contract_expiry()
            return True
        return self.connect()

    def _check_contract_expiry(self):
        """Re-qualify futures contracts if within 2 days of expiry."""
        from datetime import datetime, timedelta
        now = datetime.now()
        for sym in ("ES", "NQ", "DXY"):
            contract = self._contracts.get(sym)
            if contract and hasattr(contract, "lastTradeDateOrContractMonth"):
                try:
                    exp_str = contract.lastTradeDateOrContractMonth
                    exp_date = datetime.strptime(exp_str, "%Y%m%d")
                    if (exp_date - now).days < 2:
                        logger.info(f"IBKR: {sym} contract expiring {exp_str}, re-qualifying...")
                        self._setup_contracts()
                        return  # Re-setup handles all contracts at once
                except (ValueError, TypeError):
                    pass

    def _setup_contracts(self):
        """Pre-qualify frequently used contracts."""
        try:
            # ES futures — find front month dynamically
            es = Future("ES", "CME")
            es_all = self.ib.reqContractDetails(es)
            if es_all:
                # Filter to CME only (not QBALGO), sort by expiry, pick nearest
                cme_only = [cd for cd in es_all if cd.contract.exchange == "CME"]
                cme_only.sort(key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
                if cme_only:
                    front = cme_only[0].contract
                    self.ib.qualifyContracts(front)
                    self._contracts["ES"] = front
                    logger.info(f"IBKR: ES front month → {front.localSymbol} (exp {front.lastTradeDateOrContractMonth})")

            # ES continuous futures — for deep historical backfill (2+ years)
            try:
                es_cont = ContFuture("ES", "CME")
                self.ib.qualifyContracts(es_cont)
                self._contracts["ES_CONT"] = es_cont
                logger.info(f"IBKR: ES continuous contract qualified")
            except Exception as e:
                logger.warning(f"IBKR: ES continuous contract qualification failed: {e}")

            # NQ futures — same approach
            nq = Future("NQ", "CME")
            nq_all = self.ib.reqContractDetails(nq)
            if nq_all:
                cme_only = [cd for cd in nq_all if cd.contract.exchange == "CME"]
                cme_only.sort(key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
                if cme_only:
                    front = cme_only[0].contract
                    self.ib.qualifyContracts(front)
                    self._contracts["NQ"] = front
                    logger.info(f"IBKR: NQ front month → {front.localSymbol}")

            # SPX index (for options)
            spx = Index("SPX", "CBOE")
            qualified = self.ib.qualifyContracts(spx)
            if qualified:
                self._contracts["SPX"] = qualified[0]

            # VIX index
            vix = Index("VIX", "CBOE")
            qualified = self.ib.qualifyContracts(vix)
            if qualified:
                self._contracts["VIX"] = qualified[0]

            # VIX9D (9-day VIX) for term structure
            vix9d = Index("VIX9D", "CBOE")
            qualified = self.ib.qualifyContracts(vix9d)
            if qualified:
                self._contracts["VIX9D"] = qualified[0]

            # Breadth stocks
            for sym in ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AVGO"]:
                stk = Stock(sym, "SMART", "USD")
                qualified = self.ib.qualifyContracts(stk)
                if qualified:
                    self._contracts[sym] = qualified[0]

            # TNX (10Y yield)
            tnx = Index("TNX", "CBOE")
            qualified = self.ib.qualifyContracts(tnx)
            if qualified:
                self._contracts["TNX"] = qualified[0]

            # DXY — find front month DX future
            try:
                dx = Future("DX", "NYBOT")
                dx_all = self.ib.reqContractDetails(dx)
                if dx_all:
                    nybot_only = [cd for cd in dx_all if cd.contract.exchange == "NYBOT"]
                    nybot_only.sort(key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
                    if nybot_only:
                        front = nybot_only[0].contract
                        self.ib.qualifyContracts(front)
                        self._contracts["DXY"] = front
                        logger.info(f"IBKR: DX front month -> {front.localSymbol}")
            except Exception as e:
                logger.warning(f"IBKR: DXY contract qualification failed: {e}")

            # NYSE TICK index (real institutional flow — replaces 8-stock proxy)
            try:
                tick_nyse = Index("TICK-NYSE", "NYSE")
                qualified = self.ib.qualifyContracts(tick_nyse)
                if qualified:
                    self._contracts["TICK-NYSE"] = qualified[0]
            except Exception as e:
                logger.warning(f"IBKR: TICK-NYSE qualification failed: {e}")

            # RSP (equal-weight S&P 500 ETF — for real breadth vs cap-weight SPY)
            for sym in ["RSP", "SPY"]:
                try:
                    stk = Stock(sym, "SMART", "USD")
                    qualified = self.ib.qualifyContracts(stk)
                    if qualified:
                        self._contracts[sym] = qualified[0]
                except Exception as e:
                    logger.warning(f"IBKR: {sym} qualification failed: {e}")

            logger.info(f"IBKR: {len(self._contracts)} contracts qualified")

        except Exception as e:
            logger.warning(f"IBKR contract setup failed: {e}")

    def disconnect(self):
        """Clean disconnect."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False

    # =================================================================
    # --- HISTORICAL BARS ---
    # =================================================================

    def get_historical_bars(self, symbol: str, duration: str, bar_size: str,
                           what_to_show: str = "TRADES") -> pd.DataFrame:
        """
        Fetch historical bars from IBKR.

        Args:
            symbol: Contract key ("ES", "NQ", "SPX", "VIX", etc.)
            duration: e.g. "2 D", "5 D", "60 D", "1 Y"
            bar_size: e.g. "1 min", "5 mins", "15 mins", "1 hour", "1 day"
            what_to_show: "TRADES", "MIDPOINT", "BID", "ASK"

        Returns: DataFrame with Open, High, Low, Close, Volume columns
        """
        if not self._ensure_connected():
            return pd.DataFrame()

        contract = self._contracts.get(symbol)
        if not contract:
            logger.warning(f"IBKR: No contract for {symbol}")
            return pd.DataFrame()

        try:
            # For indices, use TRADES or MIDPOINT
            if isinstance(contract, (Index,)):
                what_to_show = "TRADES"

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,  # Include overnight/globex
                formatDate=1,
            )

            if not bars:
                return pd.DataFrame()

            df = util.df(bars)
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"
            }, inplace=True)

            # Validate expected columns exist after rename
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(df.columns):
                logger.warning(f"IBKR bars missing columns: {required - set(df.columns)}. Got: {list(df.columns)}")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.warning(f"IBKR historical bars failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_live_price(self, symbol: str) -> float:
        """Get the latest price for a contract."""
        if not self._ensure_connected():
            return 0.0

        contract = self._contracts.get(symbol)
        if not contract:
            return 0.0

        ticker = None
        try:
            ticker = self.ib.reqMktData(contract, "", False, False)
            # Wait up to 3s for data (1s often insufficient during volatile markets)
            for _ in range(6):
                self.ib.sleep(0.5)
                price = ticker.marketPrice()
                if price is not None and not np.isnan(price):
                    break
            else:
                price = ticker.marketPrice()

            if price is not None and not np.isnan(price):
                return float(price)

            # Fallback to last close
            if ticker.close is not None and not np.isnan(ticker.close):
                return float(ticker.close)

            return 0.0

        except Exception as e:
            logger.warning(f"IBKR live price failed for {symbol}: {e}")
            return 0.0
        finally:
            if ticker is not None:
                try:
                    self.ib.cancelMktData(contract)
                except Exception as e:
                    logger.debug(f"cancelMktData cleanup failed: {e}")

    # =================================================================
    # --- SPX OPTIONS (for Gamma) ---
    # =================================================================

    def get_spx_options_chain(self, expiry: str = None, strike_range: float = 50) -> dict:
        """
        Fetch SPX option chain for a specific expiry (or nearest 0DTE).
        Returns dict with 'calls' and 'puts' DataFrames.

        Results are cached for OPTIONS_CACHE_TTL seconds. Cache is also
        invalidated when SPX price moves more than 5 points from the
        price at cache time.

        Args:
            expiry: "YYYYMMDD" or None for nearest
            strike_range: +/- points from ATM to fetch
        """
        # --- Check cache validity ---
        now = time.time()
        cache_age = now - self._options_cache_time
        cache_fresh = cache_age < CFG.OPTIONS_CACHE_TTL

        if cache_fresh and self._options_cache is not None:
            # Quick price check — invalidate if price moved > 5 pts
            current_price = self.get_live_price("SPX") or self.get_live_price("ES")
            if current_price > 0 and abs(current_price - self._options_cache_price) <= 5:
                logger.info(f"IBKR: returning cached options chain "
                            f"(age={cache_age:.0f}s, price_delta="
                            f"{abs(current_price - self._options_cache_price):.1f})")
                return self._options_cache

        if not self._ensure_connected():
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": "N/A"}

        spx = self._contracts.get("SPX")
        if not spx:
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": "N/A"}

        try:
            # Get available expirations
            chains = self.ib.reqSecDefOptParams(spx.symbol, "", spx.secType, spx.conId)
            if not chains:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": "N/A"}

            # Find 0DTE or nearest expiry
            today_str = datetime.now().strftime("%Y%m%d")
            all_exps = set()
            exchange = None
            for c in chains:
                if c.exchange in ("SMART", "CBOE"):
                    all_exps.update(c.expirations)
                    exchange = c.exchange

            sorted_exps = sorted(all_exps)
            if expiry and expiry in sorted_exps:
                target_exp = expiry
            else:
                # Find nearest (today or next)
                target_exp = None
                for exp in sorted_exps:
                    if exp >= today_str:
                        target_exp = exp
                        break
                if not target_exp and sorted_exps:
                    target_exp = sorted_exps[-1]

            if not target_exp:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": "N/A"}

            # Get SPX price for strike filtering
            spx_price = self.get_live_price("SPX")
            if spx_price <= 0:
                spx_price = self.get_live_price("ES")  # ES ~ SPX

            min_strike = spx_price - strike_range
            max_strike = spx_price + strike_range

            # Request option contracts
            call_data = []
            put_data = []

            # Generate strikes (SPX has $5 strikes near ATM, $25 further out)
            strikes = list(range(int(min_strike // 5 * 5), int(max_strike // 5 * 5) + 5, 5))

            for strike in strikes:
                for right in ["C", "P"]:
                    opt = Option("SPX", target_exp, strike, right, "SMART")
                    try:
                        qualified = self.ib.qualifyContracts(opt)
                        if not qualified:
                            continue
                        opt = qualified[0]

                        # Request market data for OI and greeks
                        ticker = self.ib.reqMktData(opt, "100,101,104,106", False, False)
                        self.ib.sleep(0.1)

                        row = {
                            "strike": strike,
                            "openInterest": getattr(ticker, "callOpenInterest", 0) if right == "C"
                                            else getattr(ticker, "putOpenInterest", 0),
                            "volume": getattr(ticker, "volume", 0) or 0,
                            "lastPrice": ticker.marketPrice() if ticker.marketPrice() and not np.isnan(ticker.marketPrice()) else 0,
                            "bid": ticker.bid if ticker.bid and not np.isnan(ticker.bid) else 0,
                            "ask": ticker.ask if ticker.ask and not np.isnan(ticker.ask) else 0,
                            "impliedVol": getattr(ticker, "impliedVolatility", 0) or 0,
                        }
                        # Extract delta + gamma from modelGreeks
                        greeks = getattr(ticker, "modelGreeks", None)
                        row["delta"] = greeks.delta if greeks and hasattr(greeks, "delta") and greeks.delta else 0
                        row["gamma"] = greeks.gamma if greeks and hasattr(greeks, "gamma") and greeks.gamma else 0

                        if right == "C":
                            call_data.append(row)
                        else:
                            put_data.append(row)

                        self.ib.cancelMktData(opt)

                    except Exception as e:
                        logger.warning(f"IBKR: failed to fetch option SPX {strike}{right} "
                                       f"{target_exp}: {e}")
                        continue

            calls_df = pd.DataFrame(call_data) if call_data else pd.DataFrame()
            puts_df = pd.DataFrame(put_data) if put_data else pd.DataFrame()

            result = {"calls": calls_df, "puts": puts_df, "expiry": target_exp}

            # --- Store in cache ---
            self._options_cache = result
            self._options_cache_time = time.time()
            self._options_cache_price = spx_price
            logger.info(f"IBKR: options chain cached (price={spx_price:.1f}, "
                        f"calls={len(call_data)}, puts={len(put_data)})")

            return result

        except Exception as e:
            logger.warning(f"IBKR SPX options chain failed: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expiry": "N/A"}

    # =================================================================
    # --- SPX IV SNAPSHOT (for Skew + Term Structure) ---
    # =================================================================

    def get_spx_iv_snapshot(self, num_expiries: int = 3, strikes_per_side: int = 6) -> dict:
        """
        Fetch IV at key strikes across multiple expirations for skew + term structure.
        Returns: {
            "spx_price": float,
            "expirations": [
                {"expiry": "20260302", "dte": 0, "atm_iv": 0.18, "calls": [...], "puts": [...]},
                ...
            ]
        }
        Each call/put entry: {"strike": 6900, "iv": 0.20, "delta": 0.45, "bid": 5.2, "ask": 5.5}
        """
        if not self._ensure_connected():
            return {"spx_price": 0, "expirations": []}

        spx = self._contracts.get("SPX")
        if not spx:
            return {"spx_price": 0, "expirations": []}

        try:
            # Get SPX price
            spx_price = self.get_live_price("SPX")
            if spx_price <= 0:
                spx_price = self.get_live_price("ES")

            if spx_price <= 0:
                return {"spx_price": 0, "expirations": []}

            # Get available expirations
            chains = self.ib.reqSecDefOptParams(spx.symbol, "", spx.secType, spx.conId)
            if not chains:
                return {"spx_price": spx_price, "expirations": []}

            today_str = datetime.now().strftime("%Y%m%d")
            all_exps = set()
            for c in chains:
                if c.exchange in ("SMART", "CBOE"):
                    all_exps.update(c.expirations)

            sorted_exps = sorted(e for e in all_exps if e >= today_str)
            target_exps = sorted_exps[:num_expiries]

            if not target_exps:
                return {"spx_price": spx_price, "expirations": []}

            # ATM strike (nearest $5)
            atm = round(spx_price / 5) * 5
            # Key strikes: ATM ± 5, 10, 15, 20, 25, 30 (every $5)
            strike_offsets = list(range(-strikes_per_side * 5, (strikes_per_side + 1) * 5, 5))
            strikes = [atm + off for off in strike_offsets if off != 0] + [atm]
            strikes = sorted(set(strikes))

            exp_results = []
            for exp in target_exps:
                exp_date = datetime.strptime(exp, "%Y%m%d")
                dte = max((exp_date - datetime.now()).days, 0)

                calls = []
                puts = []
                atm_call_iv = 0
                atm_put_iv = 0

                # --- Batch qualify all option contracts for this expiry ---
                all_opts = []
                opt_keys = []  # (strike, right) for each contract
                for strike in strikes:
                    for right in ["C", "P"]:
                        opt = Option("SPX", exp, strike, right, "SMART")
                        all_opts.append(opt)
                        opt_keys.append((strike, right))

                try:
                    qualified = self.ib.qualifyContracts(*all_opts)
                except Exception as e:
                    logger.debug(f"Batch qualify for {exp} failed: {e}")
                    qualified = []

                # Filter to successfully qualified contracts
                valid = [(opt_keys[i], all_opts[i]) for i in range(len(all_opts))
                         if i < len(qualified) and qualified[i] is not None
                         and hasattr(all_opts[i], 'conId') and all_opts[i].conId > 0]

                # --- Request market data for all contracts at once ---
                tickers = []
                for (strike, right), opt in valid:
                    try:
                        t = self.ib.reqMktData(opt, "106", False, False)
                        tickers.append(((strike, right), opt, t))
                    except Exception as e:
                        logger.debug(f"reqMktData {strike}{right} failed: {e}")

                # Single sleep to let all data arrive (vs 0.15s × N serial sleeps)
                if tickers:
                    self.ib.sleep(1.5)

                # --- Harvest results ---
                for (strike, right), opt, ticker in tickers:
                    try:
                        greeks = getattr(ticker, "modelGreeks", None)
                        iv = greeks.impliedVol if greeks and greeks.impliedVol else 0
                        delta = greeks.delta if greeks and greeks.delta else 0
                        bid = ticker.bid if ticker.bid and not np.isnan(ticker.bid) else 0
                        ask = ticker.ask if ticker.ask and not np.isnan(ticker.ask) else 0

                        entry = {
                            "strike": strike,
                            "iv": round(iv, 4) if iv else 0,
                            "delta": round(delta, 3) if delta else 0,
                            "bid": round(bid, 2),
                            "ask": round(ask, 2),
                            "mid": round((bid + ask) / 2, 2) if bid and ask else 0,
                        }

                        if right == "C":
                            calls.append(entry)
                            if strike == atm:
                                atm_call_iv = iv or 0
                        else:
                            puts.append(entry)
                            if strike == atm:
                                atm_put_iv = iv or 0
                    except Exception as e:
                        logger.debug(f"IV snapshot strike {strike}{right} failed: {e}")

                # Cancel all market data subscriptions
                for _, opt, _ in tickers:
                    try:
                        self.ib.cancelMktData(opt)
                    except Exception as e:
                        logger.debug(f"cancelMktData cleanup failed: {e}")

                atm_iv = (atm_call_iv + atm_put_iv) / 2 if atm_call_iv and atm_put_iv else atm_call_iv or atm_put_iv

                exp_results.append({
                    "expiry": exp,
                    "dte": dte,
                    "atm_iv": round(atm_iv, 4),
                    "atm_strike": atm,
                    "calls": sorted(calls, key=lambda x: x["strike"]),
                    "puts": sorted(puts, key=lambda x: x["strike"]),
                })

            return {"spx_price": spx_price, "expirations": exp_results}

        except Exception as e:
            logger.warning(f"IBKR IV snapshot failed: {e}")
            return {"spx_price": spx_price if spx_price > 0 else 0, "expirations": []}

    # =================================================================
    # --- BREADTH DATA ---
    # =================================================================

    def get_breadth_data(self, symbols: list, bar_size: str = "1 min",
                         duration: str = "1 D") -> pd.DataFrame:
        """Fetch close prices for multiple symbols — used for Mag7 breadth."""
        if not self._ensure_connected():
            return pd.DataFrame()

        frames = {}
        for sym in symbols:
            contract = self._contracts.get(sym)
            if not contract:
                continue
            try:
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=duration,
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
                if bars:
                    df = util.df(bars)
                    df.set_index("date", inplace=True)
                    df.index = pd.to_datetime(df.index)
                    frames[sym] = df["close"]
            except Exception as e:
                logger.warning(f"IBKR: breadth data fetch failed for {sym}: {e}")
                continue

        if frames:
            return pd.DataFrame(frames)
        return pd.DataFrame()

    # =================================================================
    # --- TICK-BY-TICK DATA (True Cumulative Delta) ---
    # =================================================================

    def get_tick_data(self, symbol: str = "ES", count: int = 1000) -> pd.DataFrame:
        """
        Fetch recent tick-by-tick trades for true cumulative delta.

        IBKR reqHistoricalTicks returns up to 1000 ticks per request.
        Each tick has a price, size, and exchange. We classify each trade
        as buyer- or seller-initiated by comparing to the previous tick
        (uptick/zero-plus = BUY, downtick/zero-minus = SELL).

        Returns DataFrame with columns: time, price, size, side
        """
        if not self._ensure_connected():
            return pd.DataFrame()

        contract = self._contracts.get(symbol)
        if not contract:
            logger.warning(f"IBKR: no contract for {symbol} in tick data")
            return pd.DataFrame()

        try:
            # Fetch last N ticks ending now
            end_dt = datetime.now()
            ticks = self.ib.reqHistoricalTicks(
                contract,
                startDateTime="",
                endDateTime=end_dt,
                numberOfTicks=count,
                whatToShow="TRADES",
                useRth=False,
            )

            if not ticks:
                return pd.DataFrame()

            records = []
            prev_price = 0.0
            for t in ticks:
                price = float(t.price)
                size = int(t.size)
                # Tick rule: uptick = buyer, downtick = seller
                if price > prev_price:
                    side = "BUY"
                elif price < prev_price:
                    side = "SELL"
                else:
                    side = "BUY"  # zero-tick defaults to last direction
                prev_price = price
                records.append({
                    "time": t.time,
                    "price": price,
                    "size": size,
                    "side": side,
                })

            df = pd.DataFrame(records)
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            logger.debug(f"IBKR: fetched {len(df)} ticks for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"IBKR tick data fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    # =================================================================
    # --- STATUS ---
    # =================================================================

    def get_status(self) -> str:
        """Human-readable status string."""
        if not IBKR_AVAILABLE:
            return "❌ ib_insync not installed"
        if not self.connected:
            return "❌ Not connected (IBKR required — will retry)"
        contracts = len(self._contracts)
        return f"✅ Connected (port {self.port}) | {contracts} contracts"


# Legacy MarketDataIBKR class removed (v26.1) — superseded by market_data.py MarketData
