"""
Technical indicators for ES futures analysis.

All calc_* and classify_* functions extracted from market_bot_v26.py.
Each function takes a MarketData instance and returns computed values.
"""

from __future__ import annotations

import calendar
from datetime import datetime, timedelta, time as dtime
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from bot_config import logger, now_et, ET, CFG, _cache_lock

if TYPE_CHECKING:
    from market_data import MarketData

# Module-level RVOL state -- holds last valid reading to survive zero-volume bars
_rvol_last_valid: dict = {"rvol": 1.0, "status": "NORMAL"}


# =================================================================
# --- VWAP ---
# =================================================================

def calc_vwap(md: MarketData) -> Tuple[str, str, dict]:
    """Corrected VWAP with proper volume-weighted standard deviation bands."""
    try:
        df = md.es_today_1m.copy()
        if df.empty or len(df) < 5:
            return "N/A", "Normal", {}

        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_vol = df["Volume"].cumsum()
        cum_tp_vol = (typical * df["Volume"]).cumsum()

        # Avoid division by zero
        cum_vol = cum_vol.replace(0, np.nan)
        vwap = cum_tp_vol / cum_vol

        # Proper VWAP standard deviation: sqrt of volume-weighted variance
        vwap_var = ((typical - vwap) ** 2 * df["Volume"]).cumsum() / cum_vol
        vwap_std = np.sqrt(vwap_var)

        curr_price = df["Close"].iloc[-1]
        curr_vwap = vwap.iloc[-1] if not vwap.empty else np.nan
        curr_std = vwap_std.iloc[-1] if not vwap_std.empty else np.nan

        if pd.isna(curr_vwap) or pd.isna(curr_std) or curr_std == 0:
            return "N/A", "Normal", {}

        upper_1 = curr_vwap + curr_std
        lower_1 = curr_vwap - curr_std
        upper_2 = curr_vwap + (2 * curr_std)
        lower_2 = curr_vwap - (2 * curr_std)

        status = "Normal"
        if curr_price > upper_2:
            status = "OVERBOUGHT (>+2 SD)"
        elif curr_price < lower_2:
            status = "OVERSOLD (<-2 SD)"
        elif curr_price > upper_1:
            status = "Stretched High (+1 to +2 SD)"
        elif curr_price < lower_1:
            status = "Stretched Low (-1 to -2 SD)"

        levels = {
            "vwap": round(curr_vwap, 2),
            "upper_1": round(upper_1, 2),
            "lower_1": round(lower_1, 2),
            "upper_2": round(upper_2, 2),
            "lower_2": round(lower_2, 2),
        }
        return f"{curr_vwap:.2f}", status, levels

    except Exception as e:
        logger.warning(f"VWAP calculation failed: {e}", exc_info=True)
        return "N/A", "Normal", {}


# =================================================================
# --- VOLUME PROFILE ---
# =================================================================

def calc_volume_profile(md: MarketData, bins: int = 50) -> dict:
    """Session volume profile with POC, VAH, VAL."""
    try:
        df = md.es_today_1m.copy()
        if df.empty or len(df) < 10:
            return {"poc": "N/A", "vah": "N/A", "val": "N/A"}

        price_range = np.linspace(df["Low"].min(), df["High"].max(), bins + 1)
        vol_profile = np.zeros(bins)

        for i in range(bins):
            mask = (df["Close"] >= price_range[i]) & (df["Close"] < price_range[i + 1])
            vol_profile[i] = df.loc[mask, "Volume"].sum()

        # POC = highest volume price
        poc_idx = np.argmax(vol_profile)
        poc = (price_range[poc_idx] + price_range[poc_idx + 1]) / 2

        # Value Area = 70% of total volume centered on POC
        total_vol = vol_profile.sum()
        if total_vol == 0:
            return {"poc": "N/A", "vah": "N/A", "val": "N/A"}

        sorted_idx = np.argsort(vol_profile)[::-1]
        cum_vol = np.cumsum(vol_profile[sorted_idx])
        va_count = np.searchsorted(cum_vol, total_vol * 0.70) + 1
        va_indices = sorted_idx[:va_count]
        val = (price_range[va_indices.min()] + price_range[va_indices.min() + 1]) / 2
        vah = (price_range[va_indices.max()] + price_range[va_indices.max() + 1]) / 2

        return {
            "poc": round(poc, 2),
            "vah": round(vah, 2),
            "val": round(val, 2),
        }
    except Exception as e:
        logger.warning(f"Volume profile failed: {e}", exc_info=True)
        return {"poc": "N/A", "vah": "N/A", "val": "N/A"}


# =================================================================
# --- CUMULATIVE DELTA ---
# =================================================================

def _cum_delta_from_bars(md: MarketData) -> Tuple[float, str]:
    """Fallback: approximate delta from bar direction (Close >= Open = buy volume)."""
    df = md.es_today_1m.copy()
    if df.empty or len(df) < 5:
        return 0.0, "N/A"

    delta = np.where(df["Close"] >= df["Open"], df["Volume"], -df["Volume"])
    cum_delta = np.cumsum(delta)
    if len(cum_delta) == 0:
        return 0.0, "N/A"
    current = float(cum_delta[-1])

    recent = cum_delta[-30:] if len(cum_delta) >= 30 else cum_delta
    trend = "RISING" if len(recent) >= 2 and recent[-1] > recent[0] else "FALLING"

    if current > 0:
        bias = f"NET BUYERS ({trend}) (proxy)"
    elif current < 0:
        bias = f"NET SELLERS ({trend}) (proxy)"
    else:
        bias = "NEUTRAL (proxy)"
    return current, bias


def calc_cumulative_delta(md: MarketData) -> Tuple[float, str]:
    """
    Cumulative delta: buy volume minus sell volume.
    Primary: IBKR tick-by-tick data (trade-by-trade side classification).
    Fallback: bar-direction proxy (Close >= Open → buy volume).
    """
    try:
        # --- Try real tick data first ---
        if md.ibkr and md.ibkr.connected:
            try:
                tick_df = md.ibkr.get_tick_data("ES", count=1000)
                if not tick_df.empty and len(tick_df) >= 50:
                    buy_vol = int(tick_df.loc[tick_df["side"] == "BUY", "size"].sum())
                    sell_vol = int(tick_df.loc[tick_df["side"] == "SELL", "size"].sum())
                    current = float(buy_vol - sell_vol)

                    # Trend from last ~200 ticks
                    recent = tick_df.tail(200)
                    recent_buy = int(recent.loc[recent["side"] == "BUY", "size"].sum())
                    recent_sell = int(recent.loc[recent["side"] == "SELL", "size"].sum())
                    trend = "RISING" if recent_buy > recent_sell else "FALLING"

                    if current > 0:
                        bias = f"NET BUYERS ({trend}) (tick)"
                    elif current < 0:
                        bias = f"NET SELLERS ({trend}) (tick)"
                    else:
                        bias = "NEUTRAL (tick)"
                    return current, bias
            except Exception as e:
                logger.debug(f"Tick delta failed, using bar proxy: {e}")

        # --- Fallback: bar direction proxy ---
        return _cum_delta_from_bars(md)

    except Exception as e:
        logger.warning(f"Cumulative delta failed: {e}", exc_info=True)
        return 0.0, "N/A"


# =================================================================
# --- PRIOR DAY LEVELS ---
# =================================================================

def calc_prior_day_levels(md: MarketData) -> dict:
    """Previous session high, low, close, range -- key reference levels."""
    try:
        df = md.es_daily
        if df.empty or len(df) < 2:
            return {"prev_high": "N/A", "prev_low": "N/A", "prev_close": "N/A", "prev_range": "N/A"}

        prev = df.iloc[-2]
        return {
            "prev_high": round(float(prev["High"]), 2),
            "prev_low": round(float(prev["Low"]), 2),
            "prev_close": round(float(prev["Close"]), 2),
            "prev_range": round(float(prev["High"] - prev["Low"]), 2),
        }
    except Exception as e:
        logger.warning(f"Prior day levels failed: {e}", exc_info=True)
        return {"prev_high": "N/A", "prev_low": "N/A", "prev_close": "N/A", "prev_range": "N/A"}


# =================================================================
# --- INITIAL BALANCE ---
# =================================================================

def calc_initial_balance(md: MarketData) -> dict:
    """Initial Balance = first 60 minutes of RTH (9:30-10:30 ET)."""
    try:
        df = md.es_today_1m.copy()
        if df.empty:
            return {"ib_high": "N/A", "ib_low": "N/A", "ib_range": "N/A", "ib_status": "N/A"}

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York")

        rth_start = df.index[0].replace(hour=9, minute=30, second=0)
        rth_end = df.index[0].replace(hour=10, minute=30, second=0)
        ib_df = df[(df.index >= rth_start) & (df.index <= rth_end)]

        if ib_df.empty:
            return {"ib_high": "N/A", "ib_low": "N/A", "ib_range": "N/A", "ib_status": "Waiting for IB"}

        ib_high = float(ib_df["High"].max())
        ib_low = float(ib_df["Low"].min())
        ib_range = ib_high - ib_low
        curr = md.current_price

        now = now_et()
        if now.hour < 10 or (now.hour == 10 and now.minute < 30):
            status = "FORMING"
        elif curr > ib_high:
            status = f"ABOVE IB (+{curr - ib_high:.2f})"
        elif curr < ib_low:
            status = f"BELOW IB (-{ib_low - curr:.2f})"
        else:
            status = "INSIDE IB (Balanced)"

        return {
            "ib_high": round(ib_high, 2),
            "ib_low": round(ib_low, 2),
            "ib_range": round(ib_range, 2),
            "ib_status": status,
        }
    except Exception as e:
        logger.warning(f"Initial balance failed: {e}", exc_info=True)
        return {"ib_high": "N/A", "ib_low": "N/A", "ib_range": "N/A", "ib_status": "N/A"}


# =================================================================
# --- SYNTHETIC BREADTH ---
# =================================================================

def calc_synthetic_breadth(md: MarketData) -> str:
    """
    Market breadth: Mag7 stock count + RSP/SPY divergence.
    RSP (equal-weight S&P500) vs SPY (cap-weight) reveals whether the
    broad market is participating or if mega-caps are masking weakness.
    """
    try:
        # --- Mag7 component (unchanged) ---
        data = md.breadth_data
        mag7_str = ""
        green = red = 0
        if not data.empty:
            opens = data.iloc[0]
            closes = data.iloc[-1]
            pct_moves = ((closes - opens) / opens) * 100
            green = int((pct_moves > 0.05).sum())
            red = int((pct_moves < -0.05).sum())
            flat = 8 - green - red
            mag7_str = f"{green}/8 Green"

        # --- RSP/SPY divergence component ---
        rsp = getattr(md, "rsp", pd.DataFrame())
        spy = getattr(md, "spy", pd.DataFrame())

        div_str = ""
        if not rsp.empty and not spy.empty and len(rsp) >= 5 and len(spy) >= 5:
            rsp_ret = (float(rsp["Close"].iloc[-1]) - float(rsp["Close"].iloc[0])) / float(rsp["Close"].iloc[0]) * 100
            spy_ret = (float(spy["Close"].iloc[-1]) - float(spy["Close"].iloc[0])) / float(spy["Close"].iloc[0]) * 100
            spread = rsp_ret - spy_ret  # positive = broad market outperforming mega-caps

            if spread > 0.15:
                div_str = "Broad mkt leading"
            elif spread < -0.15:
                div_str = "Mega-cap only"
            else:
                div_str = "Broad participation"

        # --- Combine ---
        if not data.empty and div_str:
            # Full signal with both components
            if green >= 6 and "leading" in div_str:
                return f"STRONG BUYING ({mag7_str}, {div_str})"
            elif red >= 6 and "only" in div_str:
                return f"STRONG SELLING ({8 - green}/8 Red, {div_str})"
            elif green >= 6:
                return f"STRONG BUYING ({mag7_str}, {div_str})"
            elif red >= 6:
                return f"STRONG SELLING ({8 - green}/8 Red, {div_str})"
            elif green > red:
                return f"Positive Bias ({mag7_str}, {div_str})"
            elif red > green:
                return f"Negative Bias ({8 - green}/8 Red, {div_str})"
            else:
                return f"Neutral/Mixed ({green}G {red}R, {div_str})"
        elif not data.empty:
            # Mag7 only (no RSP/SPY)
            if green >= 6:
                return f"STRONG BUYING ({mag7_str})"
            elif red >= 6:
                return f"STRONG SELLING ({8 - green}/8 Red)"
            elif green > red:
                return f"Positive Bias ({mag7_str})"
            elif red > green:
                return f"Negative Bias ({8 - green}/8 Red)"
            else:
                return f"Neutral/Mixed ({green}G {red}R)"
        else:
            return "N/A"

    except Exception as e:
        logger.warning(f"Breadth calc failed: {e}", exc_info=True)
        return "N/A"


# =================================================================
# --- GAMMA LEVELS ---
# =================================================================

def calc_gamma_levels(md: MarketData) -> Tuple[str, str]:
    """
    Calculate intraday gamma walls from SPX options (0DTE / nearest expiry).
    Uses IBKR live options chain when connected, falls back to yfinance.
    """
    try:
        es_price = md.current_price
        if es_price <= 0:
            return "N/A", "N/A"

        # --- TRY IBKR FIRST (real-time SPX 0DTE chain) ---
        if md.ibkr and md.ibkr.connected:
            try:
                chain = md.ibkr.get_spx_options_chain(strike_range=50)
                calls_df = chain.get("calls", pd.DataFrame())
                puts_df = chain.get("puts", pd.DataFrame())
                expiry_used = chain.get("expiry", "N/A")

                if not calls_df.empty and not puts_df.empty:
                    oi_col = "openInterest"
                    if calls_df[oi_col].sum() == 0:
                        oi_col = "volume"

                    calls_f = calls_df[calls_df[oi_col] > 0].copy()
                    puts_f = puts_df[puts_df[oi_col] > 0].copy()

                    if not calls_f.empty and not puts_f.empty:
                        calls_above = calls_f[calls_f["strike"] >= es_price]
                        puts_below = puts_f[puts_f["strike"] <= es_price]

                        if calls_above.empty:
                            calls_above = calls_f
                        if puts_below.empty:
                            puts_below = puts_f

                        call_wall = float(calls_above.loc[calls_above[oi_col].idxmax(), "strike"])
                        put_wall = float(puts_below.loc[puts_below[oi_col].idxmax(), "strike"])

                        calls_sorted = calls_above.sort_values(oi_col, ascending=False)
                        puts_sorted = puts_below.sort_values(oi_col, ascending=False)
                        call_wall_2 = float(calls_sorted.iloc[1]["strike"]) if len(calls_sorted) > 1 else call_wall
                        put_wall_2 = float(puts_sorted.iloc[1]["strike"]) if len(puts_sorted) > 1 else put_wall

                        calls_above_oi = calls_f[calls_f["strike"] > es_price][oi_col].sum()
                        puts_below_oi = puts_f[puts_f["strike"] < es_price][oi_col].sum()
                        net_gamma = "CALL-HEAVY (capped upside)" if calls_above_oi > puts_below_oi * 1.3 else \
                                    "PUT-HEAVY (supported below)" if puts_below_oi > calls_above_oi * 1.3 else \
                                    "BALANCED"

                        md._gamma_detail = {
                            "call_wall": call_wall, "put_wall": put_wall,
                            "call_wall_2": call_wall_2, "put_wall_2": put_wall_2,
                            "call_oi": int(calls_f[oi_col].sum()),
                            "put_oi": int(puts_f[oi_col].sum()),
                            "net_gamma": net_gamma,
                            "expirations_used": [expiry_used],
                            "source": f"IBKR SPX 0DTE ({expiry_used})",
                            "distance_to_call": round(call_wall - es_price, 1),
                            "distance_to_put": round(es_price - put_wall, 1),
                        }
                        logger.info(
                            f"Gamma (IBKR): Call {call_wall:.0f} ({call_wall - es_price:+.0f}) | "
                            f"Put {put_wall:.0f} ({put_wall - es_price:+.0f}) | {net_gamma}"
                        )
                        return f"{call_wall:.0f}", f"{put_wall:.0f}"

                logger.info("IBKR SPX chain empty/incomplete, falling back to yfinance")
            except Exception as e:
                logger.warning(f"IBKR gamma failed, falling back to yfinance: {e}")

        # --- YFINANCE FALLBACK ---
        spx_approx = es_price

        try:
            spx = yf.Ticker("^SPX")
            exps = spx.options
            use_spx = True
            ticker_obj = spx
            strike_multiplier = 1.0
        except Exception:
            spy = yf.Ticker("SPY")
            exps = spy.options
            use_spx = False
            ticker_obj = spy
            strike_multiplier = 10.0
            spx_approx = es_price / 10.0

        if not exps:
            return "N/A", "N/A"

        near_exps = []
        for exp in exps[:4]:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                if (exp_date - now_et().replace(tzinfo=None)).days <= 2:
                    near_exps.append(exp)
            except ValueError:
                continue
        if not near_exps:
            near_exps = exps[:1]

        if use_spx:
            min_strike, max_strike = spx_approx - 50, spx_approx + 50
        else:
            min_strike, max_strike = spx_approx - 5, spx_approx + 5

        total_calls, total_puts = {}, {}
        for date in near_exps:
            try:
                chain = ticker_obj.option_chain(date)
                for _, row in chain.calls.iterrows():
                    s = float(row["strike"])
                    raw_oi = row.get("openInterest", 0)
                    oi = int(raw_oi) if pd.notna(raw_oi) else 0
                    if min_strike <= s <= max_strike and oi > 0:
                        total_calls[s] = total_calls.get(s, 0) + oi
                for _, row in chain.puts.iterrows():
                    s = float(row["strike"])
                    raw_oi = row.get("openInterest", 0)
                    oi = int(raw_oi) if pd.notna(raw_oi) else 0
                    if min_strike <= s <= max_strike and oi > 0:
                        total_puts[s] = total_puts.get(s, 0) + oi
            except Exception as e:
                logger.debug(f"Options chain expiry {date} failed: {e}")
                continue

        if not total_calls or not total_puts:
            return "N/A", "N/A"

        call_wall = max(total_calls, key=total_calls.get)
        put_wall = max(total_puts, key=total_puts.get)

        sorted_calls = sorted(total_calls.items(), key=lambda x: x[1], reverse=True)
        sorted_puts = sorted(total_puts.items(), key=lambda x: x[1], reverse=True)
        call_wall_2 = sorted_calls[1][0] if len(sorted_calls) > 1 else call_wall
        put_wall_2 = sorted_puts[1][0] if len(sorted_puts) > 1 else put_wall

        es_call = round(call_wall * strike_multiplier, 0)
        es_put = round(put_wall * strike_multiplier, 0)
        es_call_2 = round(call_wall_2 * strike_multiplier, 0)
        es_put_2 = round(put_wall_2 * strike_multiplier, 0)

        calls_above = sum(oi for s, oi in total_calls.items() if s * strike_multiplier > es_price)
        puts_below = sum(oi for s, oi in total_puts.items() if s * strike_multiplier < es_price)
        net_gamma = "CALL-HEAVY (capped upside)" if calls_above > puts_below * 1.3 else \
                    "PUT-HEAVY (supported below)" if puts_below > calls_above * 1.3 else \
                    "BALANCED"

        md._gamma_detail = {
            "call_wall": es_call, "put_wall": es_put,
            "call_wall_2": es_call_2, "put_wall_2": es_put_2,
            "call_oi": sum(total_calls.values()), "put_oi": sum(total_puts.values()),
            "net_gamma": net_gamma,
            "expirations_used": near_exps,
            "source": "SPX 0DTE (yf)" if use_spx else "SPY near-term (yf)",
            "distance_to_call": round(es_call - es_price, 1),
            "distance_to_put": round(es_price - es_put, 1),
        }
        logger.info(
            f"Gamma (yf): Call {es_call:.0f} ({es_call - es_price:+.0f}) | "
            f"Put {es_put:.0f} ({es_put - es_price:+.0f}) | {net_gamma}"
        )
        return f"{es_call:.0f}", f"{es_put:.0f}"

    except Exception as e:
        logger.warning(f"Gamma levels failed: {e}", exc_info=True)
        return "N/A", "N/A"


# =================================================================
# --- VSA AND STRUCTURE ---
# =================================================================

def calc_vsa_and_structure(md: MarketData) -> dict:
    """Volume Spread Analysis + market structure from 15m data."""
    try:
        df = md.es_15m
        if df.empty or len(df) < 20:
            return {"vsa": "N/A", "wick": "N/A", "structure": "N/A", "fractal": "N/A", "atr": 0}

        high, low, close, opn, vol = df["High"], df["Low"], df["Close"], df["Open"], df["Volume"]

        # ATR(14)
        tr = pd.concat(
            [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        atr_raw = tr.rolling(14).mean().iloc[-1]
        atr = float(atr_raw) if pd.notna(atr_raw) and np.isfinite(atr_raw) else 1.0

        # VSA
        vol_curr = float(vol.iloc[-1])
        vol_avg = float(vol.rolling(20).mean().iloc[-1])
        change = float(close.iloc[-1] - opn.iloc[-1])
        vsa = "NORMAL"
        if vol_curr > vol_avg * 1.5 and abs(change) < atr * 0.3:
            vsa = "CHURNING (High Vol, Small Move)"
        elif vol_curr < vol_avg * 0.6:
            vsa = "WEAK MOVE (Low Volume)"
        elif vol_curr > vol_avg * 1.5:
            vsa = "STRONG MOVE (High Volume)"

        # Wick analysis
        rng = float(high.iloc[-1] - low.iloc[-1])
        wick = "NEUTRAL"
        if rng > 0:
            upper_wick = float(high.iloc[-1] - max(close.iloc[-1], opn.iloc[-1]))
            lower_wick = float(min(close.iloc[-1], opn.iloc[-1]) - low.iloc[-1])
            if upper_wick / rng > 0.5:
                wick = "REJECTION (Sellers)"
            elif lower_wick / rng > 0.5:
                wick = "REJECTION (Buyers)"

        # Recent structure
        recent_high = float(high.iloc[-5:].max())
        recent_low = float(low.iloc[-5:].min())
        structure = f"Low: {recent_low:.2f} | High: {recent_high:.2f}"

        # Fractal bias (1H 50MA)
        df_1h = md.es_1h
        fractal = "N/A"
        if not df_1h.empty and len(df_1h) >= 50:
            ma_50 = float(df_1h["Close"].rolling(50).mean().iloc[-1])
            fractal = "BULLISH" if float(df_1h["Close"].iloc[-1]) > ma_50 else "BEARISH"

        return {"vsa": vsa, "wick": wick, "structure": structure, "fractal": fractal, "atr": round(atr, 2)}

    except Exception as e:
        logger.warning(f"VSA/structure failed: {e}", exc_info=True)
        return {"vsa": "N/A", "wick": "N/A", "structure": "N/A", "fractal": "N/A", "atr": 0}


# =================================================================
# --- GAP ANALYSIS ---
# =================================================================

def calc_gap_analysis(md: MarketData) -> dict:
    """Overnight gap analysis. ES gaps fill ~70% of the time."""
    try:
        daily = md.es_daily
        if daily.empty or len(daily) < 2:
            return {"gap_size": 0, "gap_dir": "N/A", "gap_pct": 0, "fill_status": "N/A", "summary": "N/A"}

        prev_close = float(daily.iloc[-2]["Close"])
        today_open = float(daily.iloc[-1]["Open"])
        current = md.current_price

        gap_pts = today_open - prev_close
        gap_pct = (gap_pts / prev_close) * 100
        gap_dir = "UP" if gap_pts > 0 else "DOWN" if gap_pts < 0 else "FLAT"

        if abs(gap_pts) < 1.0:
            fill_status = "NO GAP"
            summary = "No significant gap today."
        elif gap_dir == "UP":
            if current <= prev_close:
                fill_status = "FILLED"
                summary = f"Gap UP +{gap_pts:.1f} pts -- FULLY FILLED"
            elif current < today_open:
                fill_pct = (today_open - current) / gap_pts * 100
                fill_status = f"FILLING ({fill_pct:.0f}%)"
                summary = f"Gap UP +{gap_pts:.1f} pts -- {fill_pct:.0f}% filled, {prev_close:.2f} is fill target"
            else:
                fill_status = "EXTENDING"
                summary = f"Gap UP +{gap_pts:.1f} pts -- EXTENDING higher, gap fill at {prev_close:.2f}"
        else:  # gap down
            if current >= prev_close:
                fill_status = "FILLED"
                summary = f"Gap DOWN {gap_pts:.1f} pts -- FULLY FILLED"
            elif current > today_open:
                fill_pct = (current - today_open) / abs(gap_pts) * 100
                fill_status = f"FILLING ({fill_pct:.0f}%)"
                summary = f"Gap DOWN {gap_pts:.1f} pts -- {fill_pct:.0f}% filled, {prev_close:.2f} is fill target"
            else:
                fill_status = "EXTENDING"
                summary = f"Gap DOWN {gap_pts:.1f} pts -- EXTENDING lower, gap fill at {prev_close:.2f}"

        return {
            "gap_size": round(gap_pts, 2),
            "gap_dir": gap_dir,
            "gap_pct": round(gap_pct, 3),
            "fill_status": fill_status,
            "prev_close": prev_close,
            "today_open": today_open,
            "summary": summary,
        }
    except Exception as e:
        logger.warning(f"Gap analysis failed: {e}", exc_info=True)
        return {"gap_size": 0, "gap_dir": "N/A", "gap_pct": 0, "fill_status": "N/A", "summary": "N/A"}


# =================================================================
# --- RELATIVE VOLUME BY TIME OF DAY (RVOL) ---
# =================================================================

def calc_rvol(md: MarketData) -> dict:
    """
    Compare current 30-minute block volume against the SAME 30-min block
    average over recent sessions.

    Using 30-min blocks instead of 5-min bars smooths out the noise from
    individual bar randomness while still capturing time-of-day patterns
    (e.g., open vs midday vs close).

    RVOL > 1.5 = heavy institutional activity right now.
    RVOL < 0.5 = low-conviction move, likely to reverse.
    """
    global _rvol_last_valid
    try:
        today = now_et().date()
        if not hasattr(calc_rvol, '_last_date') or calc_rvol._last_date != today:
            with _cache_lock:
                _rvol_last_valid = {"rvol": 1.0, "status": "NORMAL"}
            calc_rvol._last_date = today
        df = md.es_5m.copy()
        if df.empty:
            return _rvol_last_valid

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York")

        now = now_et()
        current_hour = now.hour
        current_minute = now.minute
        is_rth = dtime(9, 30) <= dtime(current_hour, current_minute) <= dtime(16, 15)

        # Map each 5-min bar to a 30-min block (e.g., 10:00-10:29 → block 1000,
        # 10:30-10:59 → block 1030). We sum the 5-min volumes within each block.
        df["block_30"] = df.index.hour * 100 + (df.index.minute // 30) * 30
        df["Date"] = df.index.date
        today = now.date()

        # Current 30-min block — use the COMPLETED block, not the one in progress.
        # If we're in the first half of a 30-min block, the block is still filling,
        # so compare the PREVIOUS completed block instead.
        total_min = current_hour * 60 + current_minute
        current_block_start_min = (total_min // 30) * 30
        minutes_into_block = total_min - current_block_start_min

        if minutes_into_block < 5:
            # Just started a new block — use the prior completed block
            prev_block_min = current_block_start_min - 30
        else:
            # Enough bars accumulated in this block to compare
            prev_block_min = current_block_start_min

        prev_block_hour = prev_block_min // 60
        prev_block_minute = prev_block_min % 60
        time_block = prev_block_hour * 100 + prev_block_minute

        # Sum volume across all 5-min bars in each 30-min block per day
        vol_by_block = df.groupby(["Date", "block_30"])["Volume"].sum().reset_index()

        today_block = vol_by_block[
            (vol_by_block["Date"] == today) & (vol_by_block["block_30"] == time_block)
        ]
        if today_block.empty:
            return _rvol_last_valid if is_rth else {"rvol": 1.0, "status": "N/A"}

        current_vol = float(today_block["Volume"].iloc[-1])

        if current_vol == 0 and is_rth:
            logger.debug("RVOL: zero-volume 30-min block during RTH -- holding last valid reading")
            return _rvol_last_valid

        hist_block = vol_by_block[
            (vol_by_block["Date"] != today) & (vol_by_block["block_30"] == time_block)
        ]
        if hist_block.empty or current_vol == 0:
            return _rvol_last_valid if is_rth else {"rvol": 1.0, "status": "N/A"}

        avg_vol = float(hist_block["Volume"].mean())
        if avg_vol == 0:
            return _rvol_last_valid if is_rth else {"rvol": 1.0, "status": "N/A"}

        rvol = current_vol / avg_vol

        if rvol < 0.10 and is_rth and current_hour >= 10:
            logger.warning(f"RVOL sanity: computed {rvol:.2f}x during RTH -- likely stale data, holding {_rvol_last_valid['rvol']}x")
            return _rvol_last_valid

        if rvol >= 2.0:
            status = "VERY HIGH (Institutions active)"
        elif rvol >= 1.5:
            status = "HIGH (Above average)"
        elif rvol >= 0.8:
            status = "NORMAL"
        elif rvol >= 0.5:
            status = "LOW (Weak conviction)"
        else:
            status = "VERY LOW (Fade moves)"

        result = {"rvol": round(rvol, 2), "status": status}
        with _cache_lock:
            _rvol_last_valid = result
        return result

    except Exception as e:
        logger.warning(f"RVOL calculation failed: {e}", exc_info=True)
        return _rvol_last_valid


# =================================================================
# --- VIX TERM STRUCTURE ---
# =================================================================

def calc_vix_term_structure(md: MarketData) -> dict:
    """VIX vs VIX9D term structure analysis."""
    try:
        vix_spot = 0.0
        vix9d = 0.0
        source = "yfinance"

        if not md.vix.empty:
            vix_spot = float(md.vix["Close"].iloc[-1])

        if md.ibkr and md.ibkr.connected:
            try:
                vix9d_price = md.ibkr.get_live_price("VIX9D")
                if vix9d_price > 0:
                    vix9d = vix9d_price
                    source = "IBKR"
                else:
                    vix9d_bars = md.ibkr.get_historical_bars("VIX9D", "1 D", "15 mins")
                    if not vix9d_bars.empty:
                        vix9d = float(vix9d_bars["Close"].iloc[-1])
                        source = "IBKR (hist)"
            except Exception:
                pass

        if vix9d <= 0:
            try:
                vix9d_data = yf.Ticker("^VIX9D").history(period="2d", interval="15m")
                vix9d = float(vix9d_data["Close"].iloc[-1]) if not vix9d_data.empty else 0
                source = "yfinance"
            except Exception:
                vix9d = 0
        if vix9d <= 0 and vix_spot > 0:
            vix9d = vix_spot * 1.05  # Assume mild contango when VIX9D unavailable
            source = "estimated"

        if vix_spot <= 0 and md.ibkr and md.ibkr.connected:
            try:
                vix_spot = md.ibkr.get_live_price("VIX")
            except Exception:
                pass

        if vix_spot <= 0 or vix9d <= 0:
            return {"vix": vix_spot, "vix9d": 0, "ratio": 1.0, "structure": "N/A",
                    "signal": "N/A", "source": source}

        ratio = vix9d / vix_spot

        if ratio > 1.15:
            structure = "STEEP BACKWARDATION"
            signal = "EXTREME FEAR (contrarian bullish if extended)"
        elif ratio > 1.05:
            structure = "BACKWARDATION"
            signal = "ELEVATED FEAR (bearish, hedging heavy)"
        elif ratio > 0.95:
            structure = "FLAT"
            signal = "NEUTRAL (no strong signal)"
        elif ratio > 0.85:
            structure = "CONTANGO"
            signal = "CALM (bullish lean)"
        else:
            structure = "STEEP CONTANGO"
            signal = "COMPLACENT (watch for vol spike)"

        return {
            "vix": round(vix_spot, 2),
            "vix9d": round(vix9d, 2),
            "ratio": round(ratio, 3),
            "structure": structure,
            "signal": signal,
            "source": source,
        }
    except Exception as e:
        logger.warning(f"VIX term structure failed: {e}", exc_info=True)
        return {"vix": 0, "vix9d": 0, "ratio": 1.0, "structure": "N/A",
                "signal": "N/A", "source": "error"}


# =================================================================
# --- OVERNIGHT SESSION ---
# =================================================================

def calc_overnight_session(md: MarketData) -> dict:
    """Analyze the Globex/overnight session relative to prior RTH close."""
    try:
        df = md.es_today_1m.copy()
        daily = md.es_daily
        if df.empty or daily.empty or len(daily) < 2:
            return {"status": "N/A", "gap": 0, "gap_pct": 0, "gap_type": "N/A",
                    "overnight_high": 0, "overnight_low": 0, "overnight_range": 0,
                    "position_in_overnight": "N/A"}

        prev_close = float(daily.iloc[-2]["Close"])
        prev_high = float(daily.iloc[-2]["High"])
        prev_low = float(daily.iloc[-2]["Low"])
        prev_range = prev_high - prev_low

        if len(daily) >= 15:
            tr = pd.concat([
                daily["High"] - daily["Low"],
                (daily["High"] - daily["Close"].shift(1)).abs(),
                (daily["Low"] - daily["Close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_raw = tr.rolling(14).mean().iloc[-1]
            atr = float(atr_raw) if pd.notna(atr_raw) and np.isfinite(atr_raw) else 20.0
        else:
            atr = prev_range

        if df.index.tz is not None:
            df_tz = df.copy()
            df_tz.index = df_tz.index.tz_convert("America/New_York")
        else:
            df_tz = df

        rth_open_time = df_tz.index[0].replace(hour=9, minute=30, second=0)
        overnight = df_tz[df_tz.index < rth_open_time]

        # Use the actual RTH opening price (first bar at/after 9:30),
        # not the last pre-market bar's close.  On volatile opens these
        # can differ by several points, mis-classifying gap type.
        rth_bars = df_tz[df_tz.index >= rth_open_time]
        if not rth_bars.empty:
            rth_open = float(rth_bars["Open"].iloc[0])
        elif not overnight.empty:
            rth_open = float(overnight["Close"].iloc[-1])  # RTH not started yet
        else:
            rth_open = float(df["Open"].iloc[0])
        gap = rth_open - prev_close

        gap_pct = (gap / prev_close) * 100 if prev_close > 0 else 0
        gap_atr = abs(gap) / atr if atr > 0 else 0

        if abs(gap) < atr * 0.1:
            gap_type = "FLAT OPEN (No Gap)"
        elif gap > 0 and rth_open > prev_high:
            gap_type = "GAP UP ABOVE RANGE (True Gap)"
        elif gap < 0 and rth_open < prev_low:
            gap_type = "GAP DOWN BELOW RANGE (True Gap)"
        elif gap > 0:
            gap_type = "GAP UP (Inside Prior Range)"
        elif gap < 0:
            gap_type = "GAP DOWN (Inside Prior Range)"
        else:
            gap_type = "UNCHANGED"

        if gap_atr < 0.25:
            fill_prob = "HIGH (85%+ small gaps fill)"
        elif gap_atr < 0.5:
            fill_prob = "MODERATE (65% fill)"
        elif gap_atr < 1.0:
            fill_prob = "LOW (40% fill -- likely trend day)"
        else:
            fill_prob = "VERY LOW (Large gap -- trend continuation likely)"

        on_high = float(overnight["High"].max()) if not overnight.empty else rth_open
        on_low = float(overnight["Low"].min()) if not overnight.empty else rth_open
        on_range = on_high - on_low

        curr = md.current_price
        if on_range > 0:
            on_position = (curr - on_low) / on_range * 100
            if on_position > 80:
                pos_str = f"NEAR OVERNIGHT HIGH ({on_position:.0f}%)"
            elif on_position < 20:
                pos_str = f"NEAR OVERNIGHT LOW ({on_position:.0f}%)"
            else:
                pos_str = f"MID-RANGE ({on_position:.0f}%)"
        else:
            pos_str = "N/A"

        return {
            "gap": round(gap, 2),
            "gap_pct": round(gap_pct, 3),
            "gap_type": gap_type,
            "gap_atr_ratio": round(gap_atr, 2),
            "fill_probability": fill_prob,
            "overnight_high": round(on_high, 2),
            "overnight_low": round(on_low, 2),
            "overnight_range": round(on_range, 2),
            "position_in_overnight": pos_str,
            "prev_close": round(prev_close, 2),
        }

    except Exception as e:
        logger.warning(f"Overnight analysis failed: {e}", exc_info=True)
        return {"status": "Error", "gap": 0, "gap_pct": 0, "gap_type": "N/A",
                "overnight_high": 0, "overnight_low": 0, "overnight_range": 0,
                "position_in_overnight": "N/A", "fill_probability": "N/A",
                "gap_atr_ratio": 0, "prev_close": 0}


# =================================================================
# --- OPENING TYPE CLASSIFICATION ---
# =================================================================

def classify_opening_type(md: MarketData, ib: dict) -> dict:
    """Classify the opening type using Jim Dalton's Market Profile framework."""
    try:
        df = md.es_today_1m.copy()
        if df.empty or len(df) < 15:
            return {"type": "FORMING", "description": "Need more data", "bias": "NEUTRAL"}

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York")

        rth_start = df.index[0].replace(hour=9, minute=30, second=0)
        first_30 = df[(df.index >= rth_start) & (df.index < rth_start + timedelta(minutes=30))]
        first_15 = df[(df.index >= rth_start) & (df.index < rth_start + timedelta(minutes=15))]

        if first_15.empty:
            return {"type": "PRE-RTH", "description": "RTH not started", "bias": "NEUTRAL"}

        rth_open = float(first_15["Open"].iloc[0])
        first_15_high = float(first_15["High"].max())
        first_15_low = float(first_15["Low"].min())
        first_15_close = float(first_15["Close"].iloc[-1])

        first_15_move = first_15_close - rth_open
        first_15_range = first_15_high - first_15_low

        if first_15_range == 0:
            return {"type": "FLAT OPEN", "description": "No range in first 15m", "bias": "NEUTRAL"}

        if len(first_30) >= 15:
            first_30_low = float(first_30["Low"].min())
            first_30_high = float(first_30["High"].max())

            if first_15_move > 0:
                retrace = rth_open - first_30_low
                if retrace < first_15_range * 0.25 and first_15_move > first_15_range * 0.6:
                    return {"type": "OPEN DRIVE UP", "description": "Strong buy from open, never looked back", "bias": "STRONGLY BULLISH"}
            elif first_15_move < 0:
                retrace = first_30_high - rth_open
                if retrace < first_15_range * 0.25 and abs(first_15_move) > first_15_range * 0.6:
                    return {"type": "OPEN DRIVE DOWN", "description": "Strong sell from open, never looked back", "bias": "STRONGLY BEARISH"}

            five_min_mark = rth_start + timedelta(minutes=5)
            first_5 = first_15[first_15.index < five_min_mark] if not first_15[first_15.index < five_min_mark].empty else first_15
            first_5_move = float(first_5["Close"].iloc[-1]) - rth_open
            if first_5_move > 0 and first_15_move < 0:
                return {"type": "OPEN TEST DRIVE DOWN", "description": "Tested up first, then reversed down", "bias": "BEARISH"}
            elif first_5_move < 0 and first_15_move > 0:
                return {"type": "OPEN TEST DRIVE UP", "description": "Tested down first, then reversed up", "bias": "BULLISH"}

            curr = md.current_price
            if first_15_move > first_15_range * 0.5 and curr < rth_open:
                return {"type": "OPEN REJECTION REVERSE DOWN", "description": "Opened strong, sellers took over", "bias": "BEARISH"}
            elif first_15_move < -first_15_range * 0.5 and curr > rth_open:
                return {"type": "OPEN REJECTION REVERSE UP", "description": "Opened weak, buyers took over", "bias": "BULLISH"}

        bias = "LEAN BULLISH" if first_15_move > 0 else ("LEAN BEARISH" if first_15_move < 0 else "NEUTRAL")
        return {"type": "OPEN AUCTION", "description": "Balanced rotation, no clear conviction", "bias": bias}

    except Exception as e:
        logger.warning(f"Opening type classification failed: {e}", exc_info=True)
        return {"type": "ERROR", "description": str(e), "bias": "NEUTRAL"}


# =================================================================
# --- DAY TYPE CLASSIFICATION ---
# =================================================================

def classify_day_type(md: MarketData, ib: dict, structure: dict) -> dict:
    """Classify the likely day type based on IB width, delta, and volume."""
    try:
        ib_range = ib.get("ib_range", 0)
        ib_status = ib.get("ib_status", "N/A")

        # Use DAILY ATR, not 15m ATR.  The IB-to-ATR ratio must compare
        # the first-hour range against a full day's average range.
        # Using the 15m ATR (~5-10 pts) made ib_to_atr always > 1.0,
        # so every day was classified as "VERY WIDE IB".
        daily_atr = 0
        daily = md.es_daily
        if not daily.empty and len(daily) >= 15:
            _tr = pd.concat([
                daily["High"] - daily["Low"],
                (daily["High"] - daily["Close"].shift(1)).abs(),
                (daily["Low"] - daily["Close"].shift(1)).abs()
            ], axis=1).max(axis=1)
            _atr_raw = _tr.rolling(14).mean().iloc[-1]
            daily_atr = float(_atr_raw) if pd.notna(_atr_raw) and np.isfinite(_atr_raw) else 0

        if daily_atr == 0:
            daily_atr = structure.get("atr", 0)  # last resort fallback

        if ib_range == 0 or ib_range == "N/A" or daily_atr == 0:
            return {"type": "DEVELOPING", "description": "Waiting for IB completion",
                    "trend_probability": 0, "range_probability": 0}

        ib_to_atr = float(ib_range) / float(daily_atr) if float(daily_atr) > 0 else 1.0

        if ib_to_atr < 0.4:
            trend_prob = 70
            range_prob = 15
            day_type = "NARROW IB -- TREND DAY LIKELY"
            desc = (f"IB is only {ib_to_atr:.0%} of ATR. Coiled for expansion. "
                    f"Watch for IB breakout -- likely runs 1.5-2x IB range.")
        elif ib_to_atr < 0.6:
            trend_prob = 45
            range_prob = 30
            day_type = "MODERATE IB -- COULD GO EITHER WAY"
            desc = "Average IB width. Could develop into trend or range day."
        elif ib_to_atr < 0.8:
            trend_prob = 25
            range_prob = 50
            day_type = "WIDE IB -- RANGE DAY LIKELY"
            desc = "IB captured most of expected range. Likely rotational from here."
        else:
            trend_prob = 15
            range_prob = 60
            day_type = "VERY WIDE IB -- RANGE EXHAUSTION"
            desc = (f"IB is {ib_to_atr:.0%} of ATR. Most of the day's range is likely "
                    f"already in. Expect mean reversion to POC.")

        if "ABOVE" in str(ib_status):
            trend_prob = min(85, trend_prob + 20)
            desc += " IB broken to upside -- bullish continuation likely."
        elif "BELOW" in str(ib_status):
            trend_prob = min(85, trend_prob + 20)
            desc += " IB broken to downside -- bearish continuation likely."

        return {
            "type": day_type,
            "description": desc,
            "trend_probability": trend_prob,
            "range_probability": range_prob,
            "ib_to_atr": round(ib_to_atr, 2),
        }

    except Exception as e:
        logger.warning(f"Day type classification failed: {e}", exc_info=True)
        return {"type": "ERROR", "description": str(e),
                "trend_probability": 0, "range_probability": 0}


# =================================================================
# --- WEEKLY CONTEXT ---
# =================================================================

def calc_weekly_context(md: MarketData) -> dict:
    """Where are we in the weekly range? Monday vs Friday behavior differs."""
    try:
        now = now_et()
        dow = now.strftime("%A")
        daily = md.es_daily
        if daily.empty or len(daily) < 5:
            return {"day_of_week": dow, "weekly_position": "N/A", "notes": ""}

        monday = (now - timedelta(days=now.weekday())).date()
        _daily_idx = daily.index.tz_convert("America/New_York") if daily.index.tz is not None else daily.index
        week_data = daily[_daily_idx.date >= monday]
        if week_data.empty:
            week_data = daily.tail(1)
        week_high = float(week_data["High"].max())
        week_low = float(week_data["Low"].min())
        week_range = week_high - week_low
        curr = md.current_price

        if week_range > 0:
            position = (curr - week_low) / week_range * 100
            if position > 80:
                pos_str = f"NEAR WEEKLY HIGH ({position:.0f}%)"
            elif position < 20:
                pos_str = f"NEAR WEEKLY LOW ({position:.0f}%)"
            else:
                pos_str = f"MID-WEEK RANGE ({position:.0f}%)"
        else:
            pos_str = "N/A"

        notes = []
        third_friday = 0
        for day in range(15, 22):
            try:
                d = now.replace(day=day)
                if d.weekday() == 4:
                    third_friday = day
                    break
            except ValueError:
                continue
        if abs(now.day - third_friday) <= 4:
            notes.append("OPEX WEEK: Gamma effects amplified. Pin risk near large strikes.")

        last_day = calendar.monthrange(now.year, now.month)[1]
        if now.day >= last_day - 2:
            notes.append("MONTH-END: Institutional rebalancing flows. Possible large MOC imbalances.")

        return {
            "day_of_week": dow,
            "weekly_position": pos_str,
            "week_high": round(week_high, 2),
            "week_low": round(week_low, 2),
            "notes": " | ".join(notes) if notes else "No special conditions.",
        }

    except Exception as e:
        logger.warning(f"Weekly context failed: {e}", exc_info=True)
        return {"day_of_week": now_et().strftime("%A"), "weekly_position": "N/A", "notes": ""}


# =================================================================
# --- MULTI-TIMEFRAME MOMENTUM ALIGNMENT ---
# =================================================================

def calc_mtf_momentum(md: MarketData) -> dict:
    """
    Multi-timeframe momentum using ATR-normalized Rate of Change (ROC)
    with weighted scoring across 3 independent timeframes.

    Upgrades from old EMA-cross approach:
    - ROC is more responsive than EMA cross (less lag)
    - ATR normalization makes strength comparable across timeframes
    - Weighted scoring: daily (3x) > 1h (2x) > 5m (1x)
    - Dropped resampled 15m (was just correlated noise from 5m data)
    - ADX-based trend strength filters out choppy, directionless markets
    """
    try:
        results = {}

        # --- Timeframe weights: higher TF = more weight ---
        TF_WEIGHTS = {"5m": 1.0, "1h": 2.0, "daily": 3.0}

        def tf_momentum(df: pd.DataFrame, label: str) -> dict:
            """
            Compute momentum for a single timeframe.
            Returns {direction, strength, roc, adx} or N/A sentinel.
            """
            min_bars = 26  # Need 14 for ATR/ADX + lookback
            if df.empty or len(df) < min_bars:
                return {"direction": "N/A", "strength": 0.0, "roc": 0.0, "adx": 0.0}

            closes = df["Close"].astype(float)
            highs = df["High"].astype(float)
            lows = df["Low"].astype(float)

            # --- ROC: 10-bar rate of change ---
            roc_period = 10
            if len(closes) < roc_period + 1:
                return {"direction": "N/A", "strength": 0.0, "roc": 0.0, "adx": 0.0}

            roc = (float(closes.iloc[-1]) - float(closes.iloc[-roc_period - 1])) / max(float(closes.iloc[-roc_period - 1]), 0.01) * 100

            # --- ATR (14-period) for normalization ---
            tr_high_low = highs - lows
            tr_high_close = (highs - closes.shift(1)).abs()
            tr_low_close = (lows - closes.shift(1)).abs()
            tr = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            if atr <= 0:
                return {"direction": "N/A", "strength": 0.0, "roc": 0.0, "adx": 0.0}

            # Normalized ROC: how many ATRs has price moved in 10 bars?
            price_change = float(closes.iloc[-1]) - float(closes.iloc[-roc_period - 1])
            norm_roc = price_change / atr

            # --- ADX (14-period) for trend strength ---
            adx = _calc_adx(highs, lows, closes, period=14)

            # --- Direction classification ---
            # Strong: |norm_roc| >= 1.0 ATR AND adx >= 20 (trending)
            # Moderate: |norm_roc| >= 0.5 ATR (some momentum)
            # Neutral: weak move or no trend (adx < 15 and small roc)
            if norm_roc >= 0.5 and adx >= 20:
                direction = "BULLISH"
                strength = min(norm_roc, 3.0)  # cap at 3 ATR
            elif norm_roc <= -0.5 and adx >= 20:
                direction = "BEARISH"
                strength = min(abs(norm_roc), 3.0)
            elif norm_roc >= 1.0:
                # Strong ROC even without ADX confirmation
                direction = "BULLISH"
                strength = min(norm_roc, 3.0) * 0.7  # discount without trend
            elif norm_roc <= -1.0:
                direction = "BEARISH"
                strength = min(abs(norm_roc), 3.0) * 0.7
            else:
                direction = "NEUTRAL"
                strength = 0.0

            return {
                "direction": direction,
                "strength": round(strength, 2),
                "roc": round(roc, 2),
                "norm_roc": round(norm_roc, 2),
                "adx": round(adx, 1),
            }

        # --- Compute per-timeframe ---
        tf_5m = tf_momentum(md.es_5m, "5m")
        tf_1h = tf_momentum(md.es_1h, "1h")
        tf_daily = tf_momentum(md.es_daily, "daily")

        all_tf = {"5m": tf_5m, "1h": tf_1h, "daily": tf_daily}

        # Backward-compat: simple direction strings for telegram display
        results = {k: v["direction"] for k, v in all_tf.items()}
        # Keep 15m key for telegram display compatibility (shows "--" instead of crashing)
        results["15m"] = "--"

        # --- Weighted scoring ---
        # Each TF contributes: weight × direction(+1/-1/0) × strength
        weighted_bull = 0.0
        weighted_bear = 0.0
        total_weight = 0.0

        for tf_key, tf_data in all_tf.items():
            if tf_data["direction"] == "N/A":
                continue
            w = TF_WEIGHTS[tf_key]
            total_weight += w
            if tf_data["direction"] == "BULLISH":
                weighted_bull += w * tf_data["strength"]
            elif tf_data["direction"] == "BEARISH":
                weighted_bear += w * tf_data["strength"]

        if total_weight == 0:
            return {"alignment": "N/A", "score": 0, "timeframes": results,
                    "signal": "No data", "bullish_count": 0, "bearish_count": 0}

        # Net score: positive = bullish, negative = bearish
        net_score = weighted_bull - weighted_bear
        max_possible = total_weight * 3.0  # max strength=3.0 per TF

        # Normalize to -100..+100 range
        if max_possible > 0:
            normalized_score = int(round((net_score / max_possible) * 100))
            normalized_score = max(-100, min(100, normalized_score))
        else:
            normalized_score = 0

        # --- Classification (preserves same alignment labels for downstream compat) ---
        directions = [v for v in results.values() if v not in ("N/A", "--")]
        bullish_count = sum(1 for d in directions if d == "BULLISH")
        bearish_count = sum(1 for d in directions if d == "BEARISH")
        total = len(directions)

        if bullish_count == total and total >= 2:
            alignment = "FULL BULLISH ALIGNMENT"
            score = max(normalized_score, 80)  # floor at 80 for full alignment
            signal = "All timeframes bullish -- high conviction long"
        elif bearish_count == total and total >= 2:
            alignment = "FULL BEARISH ALIGNMENT"
            score = min(normalized_score, -80)  # floor at -80
            signal = "All timeframes bearish -- high conviction short"
        elif bullish_count >= total - 1 and bullish_count > bearish_count and total >= 2:
            alignment = "MOSTLY BULLISH"
            score = max(normalized_score, 40)  # at least 40
            signal = f"{bullish_count}/{total} timeframes bullish -- lean long"
        elif bearish_count >= total - 1 and bearish_count > bullish_count and total >= 2:
            alignment = "MOSTLY BEARISH"
            score = min(normalized_score, -40)
            signal = f"{bearish_count}/{total} timeframes bearish -- lean short"
        else:
            alignment = "MIXED / CONFLICTED"
            score = normalized_score  # keep the weighted score for granularity
            signal = "Timeframes disagree -- be cautious, range conditions likely"

        return {
            "alignment": alignment,
            "score": score,
            "timeframes": results,
            "signal": signal,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "details": {k: v for k, v in all_tf.items()},
        }

    except Exception as e:
        logger.warning(f"MTF momentum failed: {e}", exc_info=True)
        return {"alignment": "N/A", "score": 0, "timeframes": {}, "signal": "Error"}


def _calc_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series,
              period: int = 14) -> float:
    """
    Average Directional Index (ADX) — measures trend strength regardless
    of direction. ADX > 25 = strong trend, < 20 = weak/ranging.
    """
    try:
        plus_dm = highs.diff()
        minus_dm = -lows.diff()

        # Only keep positive DMs where one dominates the other
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # True Range
        tr_hl = highs - lows
        tr_hc = (highs - closes.shift(1)).abs()
        tr_lc = (lows - closes.shift(1)).abs()
        tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)

        # DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

        return float(adx.iloc[-1])
    except Exception:
        return 0.0


# =================================================================
# --- VPOC MIGRATION & NAKED POC TRACKING ---
# =================================================================

def calc_vpoc_migration(md: MarketData) -> dict:
    """Track where the Volume POC has been over the last 5 sessions."""
    try:
        daily = md.es_daily
        df_5m = md.es_5m
        if daily.empty or len(daily) < 5 or df_5m.empty:
            return {"migration": "N/A", "naked_pocs": [], "poc_list": [],
                    "trend": "N/A", "signal": "N/A"}

        if df_5m.index.tz is not None:
            df_local = df_5m.copy()
            df_local.index = df_local.index.tz_convert("America/New_York")
        else:
            df_local = df_5m.copy()

        df_local["Date"] = df_local.index.date
        dates = sorted(df_local["Date"].unique())

        poc_list = []
        for date in dates[-6:]:
            day_data = df_local[df_local["Date"] == date]
            if day_data.empty:
                continue

            closes = day_data["Close"].astype(float).values
            volumes = day_data["Volume"].astype(float).values

            if len(closes) < 5 or volumes.sum() == 0:
                continue

            price_min, price_max = closes.min(), closes.max()
            if price_max == price_min:
                poc_list.append({"date": str(date), "poc": round(float(closes.mean()), 2)})
                continue

            bins = np.linspace(price_min, price_max, min(30, len(closes)))
            vol_hist, bin_edges = np.histogram(closes, bins=bins, weights=volumes)
            poc_idx = vol_hist.argmax()
            poc_price = round(float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2), 2)
            poc_list.append({"date": str(date), "poc": poc_price})

        if len(poc_list) < 2:
            return {"migration": "N/A", "naked_pocs": [], "poc_list": poc_list,
                    "trend": "N/A", "signal": "N/A"}

        poc_values = [p["poc"] for p in poc_list]
        recent_3 = poc_values[-3:] if len(poc_values) >= 3 else poc_values
        if all(recent_3[i] <= recent_3[i + 1] for i in range(len(recent_3) - 1)):
            migration = "RISING"
            trend = "Value migrating HIGHER -- bullish"
        elif all(recent_3[i] >= recent_3[i + 1] for i in range(len(recent_3) - 1)):
            migration = "FALLING"
            trend = "Value migrating LOWER -- bearish"
        else:
            migration = "MIXED"
            trend = "Value area oscillating -- range-bound"

        current_price = md.current_price
        today = now_et().date()
        today_data = df_local[df_local["Date"] == today]

        naked_pocs = []
        for p in poc_list[:-1]:
            poc_price = p["poc"]
            if not today_data.empty:
                today_high = float(today_data["High"].max())
                today_low = float(today_data["Low"].min())
                if today_low <= poc_price <= today_high:
                    continue

            distance_pct = abs(poc_price - current_price) / current_price * 100
            if distance_pct < 1.0:
                naked_pocs.append({
                    "date": p["date"],
                    "poc": poc_price,
                    "distance": round(poc_price - current_price, 2),
                })

        signal_parts = [trend]
        if naked_pocs:
            closest = min(naked_pocs, key=lambda x: abs(x["distance"]))
            signal_parts.append(
                f"Nearest naked POC: {closest['poc']} ({closest['distance']:+.1f} pts from {closest['date']})"
            )

        return {
            "migration": migration,
            "naked_pocs": naked_pocs,
            "poc_list": poc_list[-5:],
            "trend": trend,
            "signal": " | ".join(signal_parts),
        }

    except Exception as e:
        logger.warning(f"VPOC migration failed: {e}", exc_info=True)
        return {"migration": "N/A", "naked_pocs": [], "poc_list": [],
                "trend": "N/A", "signal": "Error"}


# =================================================================
# --- NYSE TICK (real data with 8-stock proxy fallback) ---
# =================================================================

def _tick_proxy_fallback(md: MarketData) -> dict:
    """Fallback: approximate TICK from Mag7 breadth when NYSE TICK unavailable."""
    data = md.breadth_data
    if data.empty:
        return {"tick_proxy": 50, "extreme": "N/A", "cumulative": 0, "signal": "N/A",
                "source": "none"}
    try:
        data_5m = data.resample("5min").last().dropna(how="all")
    except Exception:
        data_5m = data
    if data_5m.empty or len(data_5m) < 5:
        return {"tick_proxy": 50, "extreme": "N/A", "cumulative": 0, "signal": "N/A",
                "source": "none"}

    ticks = data_5m.diff().apply(lambda col: (col > 0).astype(int)).iloc[1:]
    if ticks.empty:
        return {"tick_proxy": 50, "extreme": "N/A", "cumulative": 0, "signal": "N/A",
                "source": "none"}

    uptick_ratio = ticks.mean(axis=1)
    current_ratio = float(uptick_ratio.iloc[-1])
    recent = uptick_ratio.tail(12)
    bullish_extremes = int((recent > 0.80).sum())
    bearish_extremes = int((recent < 0.20).sum())
    cum_tick = float((uptick_ratio - 0.5).cumsum().iloc[-1])

    if current_ratio >= 0.90:  # Tightened from 0.85 to match real TICK extremity (±1000)
        extreme, signal = "BULLISH EXTREME", "Breadth proxy: strong uptick ratio"
    elif current_ratio <= 0.10:  # Tightened from 0.15 to match real TICK extremity
        extreme, signal = "BEARISH EXTREME", "Breadth proxy: strong downtick ratio"
    elif bullish_extremes >= 3:
        extreme = "BULLISH BIAS"
        signal = f"Breadth proxy: {bullish_extremes} bullish extremes in last hour"
    elif bearish_extremes >= 3:
        extreme = "BEARISH BIAS"
        signal = f"Breadth proxy: {bearish_extremes} bearish extremes in last hour"
    elif cum_tick > 1.5:
        extreme, signal = "POSITIVE DRIFT", "Breadth proxy: cumulative uptick positive"
    elif cum_tick < -1.5:
        extreme, signal = "NEGATIVE DRIFT", "Breadth proxy: cumulative uptick negative"
    else:
        extreme, signal = "NEUTRAL", "Breadth proxy: no extreme readings"

    return {
        "tick_proxy": round(current_ratio * 100, 1),
        "extreme": extreme, "cumulative": round(cum_tick, 2),
        "bullish_extremes": bullish_extremes, "bearish_extremes": bearish_extremes,
        "signal": signal, "source": "proxy",
    }


def calc_tick_proxy(md: MarketData) -> dict:
    """NYSE TICK index from IBKR.  Falls back to Mag7 breadth proxy."""
    try:
        df = getattr(md, "tick_nyse", pd.DataFrame())
        if df.empty or len(df) < 10:
            return _tick_proxy_fallback(md)

        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert("America/New_York")

        # NYSE TICK values are in the Close column of 1-min bars
        tick_vals = df["Close"].astype(float)
        current_tick = float(tick_vals.iloc[-1])

        # Cumulative TICK = running sum (proxy for institutional bias over session)
        cum_tick = float(tick_vals.sum())
        # Normalize cumulative by bar count for comparability
        cum_norm = cum_tick / len(tick_vals)

        # Recent extremes (last 12 bars = 12 minutes)
        recent = tick_vals.tail(12)
        bullish_extremes = int((recent > 800).sum())
        bearish_extremes = int((recent < -800).sum())

        # Classify
        if current_tick > 1000:
            extreme = "BULLISH EXTREME"
            signal = f"TICK {current_tick:+.0f}: strong institutional buying -- program trades"
        elif current_tick < -1000:
            extreme = "BEARISH EXTREME"
            signal = f"TICK {current_tick:+.0f}: strong institutional selling -- program trades"
        elif current_tick > 500:
            extreme = "BULLISH BIAS"
            signal = f"TICK {current_tick:+.0f}: moderate buying pressure"
        elif current_tick < -500:
            extreme = "BEARISH BIAS"
            signal = f"TICK {current_tick:+.0f}: moderate selling pressure"
        elif bullish_extremes >= 3:
            extreme = "BULLISH BIAS"
            signal = f"TICK {current_tick:+.0f}: {bullish_extremes} extremes >+800 in last 12m"
        elif bearish_extremes >= 3:
            extreme = "BEARISH BIAS"
            signal = f"TICK {current_tick:+.0f}: {bearish_extremes} extremes <-800 in last 12m"
        elif cum_norm > 200:
            extreme = "POSITIVE DRIFT"
            signal = f"TICK {current_tick:+.0f}: cumulative TICK positive -- steady bid"
        elif cum_norm < -200:
            extreme = "NEGATIVE DRIFT"
            signal = f"TICK {current_tick:+.0f}: cumulative TICK negative -- steady offers"
        else:
            extreme = "NEUTRAL"
            signal = f"TICK {current_tick:+.0f}: balanced flow"

        return {
            "tick_proxy": current_tick,
            "extreme": extreme,
            "cumulative": round(cum_norm, 1),
            "bullish_extremes": bullish_extremes,
            "bearish_extremes": bearish_extremes,
            "signal": signal,
            "source": "NYSE TICK",
        }

    except Exception as e:
        logger.warning(f"NYSE TICK failed, falling back to proxy: {e}", exc_info=True)
        return _tick_proxy_fallback(md)


# =================================================================
# --- ANCHORED VWAPS ---
# =================================================================

def calc_anchored_vwaps(md) -> dict:
    """Calculate VWAP anchored to key timeframes."""
    try:
        df = md.es_15m.copy()
        daily = md.es_daily.copy()
        if df.empty or len(df) < 20:
            return {"weekly_vwap": "N/A", "monthly_vwap": "N/A",
                    "swing_vwap": "N/A", "convergence": "N/A", "signal": "N/A"}

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        curr = md.current_price

        def calc_vwap_from(data: pd.DataFrame, start_date) -> float:
            mask = data.index >= pd.Timestamp(start_date)
            subset = data[mask]
            if subset.empty:
                return 0

            typical = (subset["High"].astype(float) + subset["Low"].astype(float) +
                       subset["Close"].astype(float)) / 3
            vol = subset["Volume"].astype(float)
            cum_tp_vol = (typical * vol).cumsum()
            cum_vol = vol.cumsum()

            vwap_series = cum_tp_vol / cum_vol
            last_vwap = float(vwap_series.iloc[-1])
            return round(last_vwap, 2) if not np.isnan(last_vwap) else 0

        now = now_et()
        monday = now - timedelta(days=now.weekday())
        monday_start = monday.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        weekly_vwap = calc_vwap_from(df, monday_start)

        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        monthly_vwap = calc_vwap_from(df, month_start)

        swing_vwap = 0
        swing_label = "N/A"
        if not daily.empty and len(daily) >= 10:
            if daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)

            recent = daily.tail(20)
            high_idx = recent["High"].idxmax()
            low_idx = recent["Low"].idxmin()

            if high_idx > low_idx:
                swing_date = high_idx
                swing_label = f"Swing High ({swing_date.strftime('%m/%d')})"
            else:
                swing_date = low_idx
                swing_label = f"Swing Low ({swing_date.strftime('%m/%d')})"

            swing_vwap = calc_vwap_from(df, swing_date)

        vwaps = [v for v in [weekly_vwap, monthly_vwap, swing_vwap] if v > 0]
        convergence = "N/A"
        convergence_level = 0
        if len(vwaps) >= 2:
            spread = max(vwaps) - min(vwaps)
            avg_vwap = np.mean(vwaps)
            spread_pct = spread / avg_vwap * 100 if avg_vwap > 0 else 99

            if spread_pct < 0.1:
                convergence = f"TIGHT CONVERGENCE at {avg_vwap:.2f} (all within {spread:.1f} pts)"
                convergence_level = avg_vwap
            elif spread_pct < 0.3:
                convergence = f"MODERATE CONVERGENCE near {avg_vwap:.2f} ({spread:.1f} pts spread)"
                convergence_level = avg_vwap
            else:
                convergence = f"Spread: {spread:.1f} pts between anchored VWAPs"

        signals = []
        if weekly_vwap > 0:
            dist_w = curr - weekly_vwap
            if abs(dist_w) < 3:
                signals.append(f"AT weekly VWAP ({weekly_vwap:.2f})")
            elif dist_w > 0:
                signals.append(f"Above weekly VWAP by {dist_w:.1f}")
            else:
                signals.append(f"Below weekly VWAP by {abs(dist_w):.1f}")

        if monthly_vwap > 0:
            dist_m = curr - monthly_vwap
            if abs(dist_m) < 5:
                signals.append(f"AT monthly VWAP ({monthly_vwap:.2f})")

        if convergence_level > 0 and abs(curr - convergence_level) < 5:
            signals.append("VWAP CONVERGENCE ZONE -- strong S/R magnet")

        return {
            "weekly_vwap": weekly_vwap if weekly_vwap > 0 else "N/A",
            "monthly_vwap": monthly_vwap if monthly_vwap > 0 else "N/A",
            "swing_vwap": swing_vwap if swing_vwap > 0 else "N/A",
            "swing_label": swing_label,
            "convergence": convergence,
            "convergence_level": convergence_level,
            "signal": " | ".join(signals) if signals else "No anchored VWAP signals.",
        }

    except Exception as e:
        logger.warning(f"Anchored VWAPs failed: {e}", exc_info=True)
        return {"weekly_vwap": "N/A", "monthly_vwap": "N/A",
                "swing_vwap": "N/A", "convergence": "N/A", "signal": "Error"}


# =================================================================
# --- LIQUIDITY SWEEP / STOP HUNT DETECTION ---
# =================================================================

def calc_liquidity_sweeps(md) -> dict:
    """Detect liquidity sweeps / stop hunts."""
    try:
        df = md.es_5m.copy()
        if df.empty or len(df) < 50:
            return {"sweeps": [], "stop_clusters": [], "active_sweep": "NONE", "signal": "N/A"}

        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)

        curr = md.current_price
        today = now_et().date()

        recent = df.tail(200)

        swing_highs = []
        swing_lows = []
        highs = recent["High"].astype(float).values
        lows = recent["Low"].astype(float).values
        times = recent.index

        for i in range(5, len(recent) - 5):
            if highs[i] == max(highs[i-5:i+6]):
                swing_highs.append({"price": float(highs[i]), "time": times[i]})
            if lows[i] == min(lows[i-5:i+6]):
                swing_lows.append({"price": float(lows[i]), "time": times[i]})

        def dedup_levels(levels, threshold=2.0):
            if not levels:
                return []
            sorted_lvls = sorted(levels, key=lambda x: x["price"])
            result = [sorted_lvls[0]]
            for lvl in sorted_lvls[1:]:
                if abs(lvl["price"] - result[-1]["price"]) > threshold:
                    result.append(lvl)
            return result

        swing_highs = dedup_levels(swing_highs)
        swing_lows = dedup_levels(swing_lows)

        sweeps = []
        today_data = df[df.index.date == today] if not df.empty else pd.DataFrame()
        SWEEP_EXPIRY_BARS = 18  # 18 × 5min = 90 minutes

        if not today_data.empty:
            today_highs = today_data["High"].astype(float)
            today_lows = today_data["Low"].astype(float)
            now_ts = now_et().replace(tzinfo=None)

            for sh in swing_highs[-5:]:
                # Find the bar that swept above this swing high
                sweep_bars = today_data[today_highs > sh["price"]]
                if sweep_bars.empty:
                    continue
                sweep_time = sweep_bars.index[0]
                # Check: swept above AND reversed below AND within expiry window
                bars_since = len(today_data[today_data.index >= sweep_time])
                if curr < sh["price"] and bars_since <= SWEEP_EXPIRY_BARS:
                    age_min = int((now_ts - sweep_time).total_seconds() / 60)
                    sweeps.append({
                        "type": "BULL TRAP",
                        "level": sh["price"],
                        "swept_by": round(float(today_highs.max()) - sh["price"], 2),
                        "age_min": age_min,
                        "description": f"Swept above {sh['price']:.2f} then reversed -- stops hit above ({age_min}m ago)",
                    })

            for sl in swing_lows[-5:]:
                # Find the bar that swept below this swing low
                sweep_bars = today_data[today_lows < sl["price"]]
                if sweep_bars.empty:
                    continue
                sweep_time = sweep_bars.index[0]
                # Check: swept below AND reversed above AND within expiry window
                bars_since = len(today_data[today_data.index >= sweep_time])
                if curr > sl["price"] and bars_since <= SWEEP_EXPIRY_BARS:
                    age_min = int((now_ts - sweep_time).total_seconds() / 60)
                    sweeps.append({
                        "type": "BEAR TRAP",
                        "level": sl["price"],
                        "swept_by": round(sl["price"] - float(today_lows.min()), 2),
                        "age_min": age_min,
                        "description": f"Swept below {sl['price']:.2f} then reversed -- stops hit below ({age_min}m ago)",
                    })

        base = int(curr // 25) * 25
        stop_clusters = []
        _max_dist = CFG.STOP_CLUSTER_MAX_DIST
        _min_dist = CFG.STOP_CLUSTER_MIN_DIST
        for level in range(base - 50, base + 75, 25):
            level = float(level)
            distance = round(level - curr, 2)
            if abs(distance) < _max_dist and abs(distance) > _min_dist:
                cluster_type = "ROUND" if level % 100 == 0 else "QUARTER"
                stop_clusters.append({
                    "level": level,
                    "distance": distance,
                    "type": cluster_type,
                    "description": f"{'Major' if cluster_type == 'ROUND' else 'Minor'} stop cluster at {level:.0f} ({distance:+.0f} pts)",
                })

        active_sweep = "NONE"
        sweep_signal = "No active sweeps."
        if sweeps:
            latest = sweeps[-1]
            active_sweep = latest["type"]
            age_str = f" ({latest['age_min']}m ago)" if latest.get("age_min", 0) > 0 else ""
            if latest["type"] == "BULL TRAP":
                sweep_signal = f"BULL TRAP at {latest['level']:.2f} -- stops swept above, now reversing down{age_str}."
            else:
                sweep_signal = f"BEAR TRAP at {latest['level']:.2f} -- stops swept below, now reversing up{age_str}."

        nearest_above = [c for c in stop_clusters if c["distance"] > 0]
        nearest_below = [c for c in stop_clusters if c["distance"] < 0]
        cluster_signal = ""
        if nearest_above:
            cluster_signal += f"Stops above: {nearest_above[0]['level']:.0f} ({nearest_above[0]['distance']:+.0f})"
        if nearest_below:
            if cluster_signal:
                cluster_signal += " | "
            cluster_signal += f"Stops below: {nearest_below[-1]['level']:.0f} ({nearest_below[-1]['distance']:+.0f})"

        return {
            "sweeps": sweeps,
            "stop_clusters": stop_clusters,
            "active_sweep": active_sweep,
            "signal": sweep_signal,
            "cluster_signal": cluster_signal,
            "swing_highs": [{"price": s["price"]} for s in swing_highs[-3:]],
            "swing_lows": [{"price": s["price"]} for s in swing_lows[-3:]],
        }

    except Exception as e:
        logger.warning(f"Liquidity sweep detection failed: {e}", exc_info=True)
        return {"sweeps": [], "stop_clusters": [], "active_sweep": "NONE", "signal": "Error"}


# =================================================================
# --- CROSS-ASSET CORRELATION ---
# =================================================================

_CROSS_CORR_DEFAULT = {
    "es_dxy_corr": 0, "es_vix_corr": 0, "es_tnx_corr": 0,
    "regime": "N/A", "divergences": [], "signal": "N/A",
}


def calc_cross_asset_correlation(md: MarketData) -> dict:
    """
    Rolling correlation of ES vs DXY, VIX, TNX.
    Standard risk-on: ES↑ DXY↓ VIX↓.  Divergences are early warnings.
    """
    try:
        es = md.es_15m
        dxy = getattr(md, "dxy", pd.Series(dtype=float))
        vix = md.vix
        tnx = md.tnx

        if es.empty or len(es) < 25:
            return _CROSS_CORR_DEFAULT.copy()

        es_ret = es["Close"].pct_change().dropna()
        if len(es_ret) < 20:
            return _CROSS_CORR_DEFAULT.copy()

        correlations = {}
        divergences = []

        # --- ES vs DXY ---
        if not dxy.empty and len(dxy) >= 20:
            dxy_ret = dxy.pct_change().dropna()
            # Align on overlapping indices
            aligned = pd.concat([es_ret, dxy_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 20:
                aligned.columns = ["es", "dxy"]
                corr_raw = aligned["es"].rolling(20).corr(aligned["dxy"]).iloc[-1]
                if pd.notna(corr_raw) and np.isfinite(corr_raw):
                    corr = float(corr_raw)
                    correlations["es_dxy_corr"] = round(corr, 3)
                    # Normally inverse: ES↑ = DXY↓.  Positive corr = divergence
                    if corr > 0.3:
                        divergences.append("ES+DXY rising together (unusual)")

        # --- ES vs VIX ---
        if not vix.empty and len(vix) >= 20:
            vix_ret = vix["Close"].pct_change().dropna()
            aligned = pd.concat([es_ret, vix_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 20:
                aligned.columns = ["es", "vix"]
                corr_raw = aligned["es"].rolling(20).corr(aligned["vix"]).iloc[-1]
                if pd.notna(corr_raw) and np.isfinite(corr_raw):
                    corr = float(corr_raw)
                    correlations["es_vix_corr"] = round(corr, 3)
                    # Normally inverse: ES↑ = VIX↓.  Positive corr = divergence
                    if corr > 0.2:
                        divergences.append("ES+VIX rising together (complacency warning)")

        # --- ES vs TNX (10Y yield) ---
        if isinstance(tnx, pd.Series) and not tnx.empty and len(tnx) >= 20:
            tnx_ret = tnx.pct_change().dropna()
            aligned = pd.concat([es_ret, tnx_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 20:
                aligned.columns = ["es", "tnx"]
                corr_raw = aligned["es"].rolling(20).corr(aligned["tnx"]).iloc[-1]
                if pd.notna(corr_raw) and np.isfinite(corr_raw):
                    correlations["es_tnx_corr"] = round(corr_raw, 3)
                    # No fixed normal — depends on regime

        # --- Regime classification ---
        es_dxy = correlations.get("es_dxy_corr", 0)
        es_vix = correlations.get("es_vix_corr", 0)

        if es_dxy < -0.3 and es_vix < -0.3:
            regime = "RISK-ON (classic)"
        elif es_dxy > 0.3 and es_vix > 0.3:
            regime = "RISK-OFF (correlated sell)"
        elif es_dxy > 0.3 and es_vix < -0.3:
            regime = "USD-DRIVEN (dollar strength)"
        elif divergences:
            regime = "DIVERGENCE WARNING"
        else:
            regime = "MIXED"

        signal_parts = [regime]
        if divergences:
            signal_parts.extend(divergences)

        result = _CROSS_CORR_DEFAULT.copy()
        result.update(correlations)
        result["regime"] = regime
        result["divergences"] = divergences
        result["signal"] = " | ".join(signal_parts)
        return result

    except Exception as e:
        logger.warning(f"Cross-asset correlation failed: {e}", exc_info=True)
        return _CROSS_CORR_DEFAULT.copy()


# =================================================================
# --- IV SKEW ANALYSIS ---
# =================================================================

_iv_skew_cache: dict = {}
_iv_skew_cache_time: float = 0
_IV_SKEW_TTL = 300  # 5-minute cache


def calc_iv_skew(md: MarketData) -> dict:
    """
    SPX implied volatility skew and term structure from IBKR IV snapshot.
    Analyzes put/call skew and near/far term structure slope.
    Cached for 5 minutes since IV changes slowly intraday.
    """
    global _iv_skew_cache, _iv_skew_cache_time
    import time as _time

    default = {
        "put_call_skew": 0, "skew_signal": "N/A",
        "term_slope": 0, "term_signal": "N/A",
        "atm_iv": 0, "signal": "N/A", "source": "none",
    }

    try:
        # Check cache
        now = _time.time()
        if _iv_skew_cache and (now - _iv_skew_cache_time) < _IV_SKEW_TTL:
            return _iv_skew_cache

        if not (md.ibkr and md.ibkr.connected):
            return default

        snapshot = md.ibkr.get_spx_iv_snapshot(num_expiries=2, strikes_per_side=4)
        if not snapshot or not snapshot.get("expirations"):
            return default

        exps = snapshot["expirations"]
        spx_price = snapshot.get("spx_price", 0)
        if spx_price <= 0:
            return default

        # --- Analyze first (nearest) expiry for skew ---
        exp0 = exps[0]
        atm_iv = exp0.get("atm_iv", 0)
        puts = exp0.get("puts", [])
        calls = exp0.get("calls", [])

        # Find ~25-delta strikes (roughly 1-2 strikes OTM)
        otm_put_iv = 0
        otm_call_iv = 0
        for p in puts:
            if 0 < abs(p.get("delta", 0)) < 0.35 and p.get("iv", 0) > 0:
                otm_put_iv = p["iv"]
                break
        for c in calls:
            if 0 < abs(c.get("delta", 0)) < 0.35 and c.get("iv", 0) > 0:
                otm_call_iv = c["iv"]
                break

        # Put/call skew: >1.0 = puts more expensive (fear), <1.0 = calls expensive
        put_call_skew = 0
        skew_signal = "N/A"
        if otm_put_iv > 0 and otm_call_iv > 0:
            put_call_skew = round(otm_put_iv / otm_call_iv, 3)
            if put_call_skew > 1.20:
                skew_signal = "HEAVY PUT SKEW (fear premium -- downside protection bid)"
            elif put_call_skew > 1.05:
                skew_signal = "MODERATE PUT SKEW (normal hedging)"
            elif put_call_skew < 0.95:
                skew_signal = "CALL SKEW (upside demand -- bullish positioning)"
            else:
                skew_signal = "FLAT SKEW (balanced)"

        # --- Term structure slope (near vs far ATM IV) ---
        term_slope = 0
        term_signal = "N/A"
        if len(exps) >= 2:
            atm_iv_near = exps[0].get("atm_iv", 0)
            atm_iv_far = exps[1].get("atm_iv", 0)
            if atm_iv_near > 0 and atm_iv_far > 0:
                term_slope = round(atm_iv_far - atm_iv_near, 4)
                if term_slope > 0.02:
                    term_signal = "CONTANGO (normal -- calm near-term)"
                elif term_slope < -0.02:
                    term_signal = "BACKWARDATION (near-term fear elevated)"
                else:
                    term_signal = "FLAT TERM STRUCTURE"

        # Composite signal
        signals = []
        if skew_signal != "N/A":
            signals.append(f"Skew {put_call_skew:.2f}: {skew_signal}")
        if term_signal != "N/A":
            signals.append(f"Term: {term_signal}")

        result = {
            "put_call_skew": put_call_skew,
            "skew_signal": skew_signal,
            "term_slope": term_slope,
            "term_signal": term_signal,
            "atm_iv": round(atm_iv, 4) if atm_iv else 0,
            "signal": " | ".join(signals) if signals else "N/A",
            "source": "IBKR IV snapshot",
        }

        # Update cache
        _iv_skew_cache = result
        _iv_skew_cache_time = now
        return result

    except Exception as e:
        logger.warning(f"IV skew analysis failed: {e}", exc_info=True)
        return default


# =================================================================
# --- DELTA AT KEY LEVELS (Level-Aware Order Flow) ---
# =================================================================

def _delta_at_levels_empty() -> dict:
    """Return empty / fallback result for delta-at-levels."""
    return {
        "levels": [],
        "net_bias": "NEUTRAL",
        "bias_score": 0,
        "strongest_signal": "N/A",
        "signal": "N/A",
        "source": "N/A",
    }


def calc_delta_at_levels(md: "MarketData", metrics: dict) -> dict:
    """
    Compute buy/sell order-flow imbalance at key price levels.

    For each key level (VWAP, prior day H/L/C, VPOC, VAH/VAL, gamma walls)
    we measure net delta (buy vol − sell vol) inside a ±zone around the
    level, using tick data (preferred) or 1-minute bar proxies (fallback).

    Patterns detected per level:
      ABSORPTION  — price tested the level, heavy opposing delta rejected it
      BREAKOUT    — price broke through WITH confirming delta
      EXHAUSTION  — price near the level but delta is dying (low magnitude)
      REJECTION   — brief breach then immediate reversal in delta
      NEUTRAL     — insufficient activity or level not tested

    Returns:
        dict with:
            levels: list[dict]  — per-level breakdown
            net_bias: str       — aggregate BULLISH / BEARISH / NEUTRAL
            bias_score: int     — weighted score (-100..+100)
            strongest_signal: str — most actionable level pattern
            signal: str         — human-readable summary
            source: str         — tick / proxy
    """
    try:
        curr_price = md.current_price
        if curr_price <= 0:
            return _delta_at_levels_empty()

        # ----------------------------------------------------------
        # 1. Gather key levels from metrics already computed
        # ----------------------------------------------------------
        key_levels: list[dict] = []  # {"name", "price", "role"}

        # VWAP
        vwap_lvls = metrics.get("vwap_levels", {})
        if vwap_lvls.get("vwap"):
            key_levels.append({"name": "VWAP", "price": float(vwap_lvls["vwap"]), "role": "pivot"})

        # Volume profile
        vpoc = metrics.get("vpoc", {})
        if vpoc.get("poc") and vpoc["poc"] != "N/A":
            key_levels.append({"name": "VPOC", "price": float(vpoc["poc"]), "role": "pivot"})
        if vpoc.get("vah") and vpoc["vah"] != "N/A":
            key_levels.append({"name": "VAH", "price": float(vpoc["vah"]), "role": "resistance"})
        if vpoc.get("val") and vpoc["val"] != "N/A":
            key_levels.append({"name": "VAL", "price": float(vpoc["val"]), "role": "support"})

        # Prior day
        prior = metrics.get("prior", {})
        if prior.get("prev_high") and prior["prev_high"] != "N/A":
            key_levels.append({"name": "PrevHigh", "price": float(prior["prev_high"]), "role": "resistance"})
        if prior.get("prev_low") and prior["prev_low"] != "N/A":
            key_levels.append({"name": "PrevLow", "price": float(prior["prev_low"]), "role": "support"})
        if prior.get("prev_close") and prior["prev_close"] != "N/A":
            key_levels.append({"name": "PrevClose", "price": float(prior["prev_close"]), "role": "pivot"})

        # Gamma walls
        gamma = metrics.get("gamma_detail", {})
        if gamma.get("call_wall"):
            key_levels.append({"name": "GammaCall", "price": float(gamma["call_wall"]), "role": "resistance"})
        if gamma.get("put_wall"):
            key_levels.append({"name": "GammaPut", "price": float(gamma["put_wall"]), "role": "support"})

        # IB levels
        ib = metrics.get("ib", {})
        if ib.get("ib_high") and ib["ib_high"] != "N/A":
            key_levels.append({"name": "IB_High", "price": float(ib["ib_high"]), "role": "resistance"})
        if ib.get("ib_low") and ib["ib_low"] != "N/A":
            key_levels.append({"name": "IB_Low", "price": float(ib["ib_low"]), "role": "support"})

        if not key_levels:
            return _delta_at_levels_empty()

        # ----------------------------------------------------------
        # 2. Build price→delta map from tick or bar data
        # ----------------------------------------------------------
        # Zone: ±2 ES points (8 ticks) around each level
        ZONE_PTS = 2.0

        # Filter to levels within ±30 pts of current price (relevant zone)
        key_levels = [lv for lv in key_levels if abs(lv["price"] - curr_price) <= 30]
        if not key_levels:
            return _delta_at_levels_empty()

        # Try tick data first (high fidelity)
        source = "proxy"
        price_delta_map: list[dict] = []  # list of {"price": float, "delta": float}

        if md.ibkr and md.ibkr.connected:
            try:
                tick_df = md.ibkr.get_tick_data("ES", count=1000)
                if not tick_df.empty and len(tick_df) >= 50:
                    source = "tick"
                    for _, row in tick_df.iterrows():
                        d = float(row["size"]) if row["side"] == "BUY" else -float(row["size"])
                        price_delta_map.append({"price": float(row["price"]), "delta": d})
            except Exception as e:
                logger.debug(f"Delta-at-levels tick fetch failed, using bars: {e}")

        # Fallback: 1-minute bars as proxy
        if not price_delta_map:
            df = md.es_today_1m.copy() if not md.es_today_1m.empty else md.es_1m.copy()
            if not df.empty:
                # Use last 60 bars (~1 hour of 1-min data)
                recent = df.tail(60)
                for _, bar in recent.iterrows():
                    bar_mid = (float(bar["High"]) + float(bar["Low"])) / 2
                    vol = float(bar["Volume"]) if bar["Volume"] > 0 else 1
                    # Bar direction: close >= open → buy, else sell
                    d = vol if bar["Close"] >= bar["Open"] else -vol
                    price_delta_map.append({"price": bar_mid, "delta": d})

        if not price_delta_map:
            return _delta_at_levels_empty()

        # ----------------------------------------------------------
        # 3. Compute delta at each key level
        # ----------------------------------------------------------
        level_results: list[dict] = []
        bull_score = 0.0
        bear_score = 0.0

        for lv in key_levels:
            lvl_price = lv["price"]
            zone_lo = lvl_price - ZONE_PTS
            zone_hi = lvl_price + ZONE_PTS

            # Filter activity in the zone
            in_zone = [p for p in price_delta_map if zone_lo <= p["price"] <= zone_hi]
            buy_vol = sum(p["delta"] for p in in_zone if p["delta"] > 0)
            sell_vol = sum(abs(p["delta"]) for p in in_zone if p["delta"] < 0)
            net_delta = buy_vol - sell_vol
            total_vol = buy_vol + sell_vol
            distance = curr_price - lvl_price  # positive = price above level

            # Skip if negligible activity
            if total_vol < 5:
                level_results.append({
                    "name": lv["name"], "price": lvl_price, "role": lv["role"],
                    "net_delta": 0, "buy_vol": 0, "sell_vol": 0,
                    "pattern": "NEUTRAL", "strength": 0, "distance": round(distance, 2),
                })
                continue

            # Imbalance ratio
            imbalance = net_delta / total_vol if total_vol > 0 else 0  # -1..+1

            # Classify pattern
            price_near = abs(distance) <= ZONE_PTS * 2  # within 4 pts
            price_above = distance > ZONE_PTS
            price_below = distance < -ZONE_PTS

            pattern = "NEUTRAL"
            strength = 0  # 0-100

            if price_near:
                # Price is AT the level right now
                if abs(imbalance) < 0.15:
                    pattern = "EXHAUSTION"
                    strength = 30
                elif lv["role"] == "support":
                    if imbalance > 0.25:
                        pattern = "ABSORPTION"
                        strength = int(min(100, imbalance * 120))
                    elif imbalance < -0.25:
                        pattern = "BREAKOUT"
                        strength = int(min(100, abs(imbalance) * 120))
                elif lv["role"] == "resistance":
                    if imbalance < -0.25:
                        pattern = "ABSORPTION"
                        strength = int(min(100, abs(imbalance) * 120))
                    elif imbalance > 0.25:
                        pattern = "BREAKOUT"
                        strength = int(min(100, imbalance * 120))
                else:
                    # Pivot — either side is meaningful
                    if abs(imbalance) > 0.3:
                        pattern = "ABSORPTION" if total_vol > 50 else "REJECTION"
                        strength = int(min(100, abs(imbalance) * 100))
            elif price_above and lv["role"] == "support":
                # Price bounced above support — was it defended?
                if imbalance > 0.2 and total_vol > 20:
                    pattern = "ABSORPTION"
                    strength = int(min(80, imbalance * 100))
            elif price_below and lv["role"] == "resistance":
                # Price rejected below resistance — was it defended?
                if imbalance < -0.2 and total_vol > 20:
                    pattern = "ABSORPTION"
                    strength = int(min(80, abs(imbalance) * 100))
            elif price_above and lv["role"] == "resistance":
                # Price broke above resistance
                if imbalance > 0.3:
                    pattern = "BREAKOUT"
                    strength = int(min(90, imbalance * 110))
            elif price_below and lv["role"] == "support":
                # Price broke below support
                if imbalance < -0.3:
                    pattern = "BREAKOUT"
                    strength = int(min(90, abs(imbalance) * 110))

            # REJECTION: brief breach, delta flipped
            if price_near and total_vol > 30:
                recent_zone = in_zone[-min(len(in_zone), 10):]
                if len(recent_zone) >= 5:
                    early_delta = sum(p["delta"] for p in recent_zone[:len(recent_zone) // 2])
                    late_delta = sum(p["delta"] for p in recent_zone[len(recent_zone) // 2:])
                    if early_delta * late_delta < 0 and abs(late_delta) > abs(early_delta) * 0.5:
                        pattern = "REJECTION"
                        strength = max(strength, int(min(85, abs(late_delta / (abs(early_delta) + 1)) * 50)))

            # Accumulate directional score
            wt = 1.0
            if lv["name"] in ("VWAP", "VPOC"):
                wt = 1.5  # key pivot levels
            elif lv["name"].startswith("Gamma"):
                wt = 1.3  # gamma walls are significant

            if pattern == "ABSORPTION":
                if lv["role"] == "support":
                    bull_score += strength * wt / 100
                elif lv["role"] == "resistance":
                    bear_score += strength * wt / 100
                else:
                    # Pivot: absorption direction from imbalance
                    if imbalance > 0:
                        bull_score += strength * wt / 100
                    else:
                        bear_score += strength * wt / 100
            elif pattern == "BREAKOUT":
                if price_above:
                    bull_score += strength * wt / 100
                else:
                    bear_score += strength * wt / 100
            elif pattern == "REJECTION":
                # Rejection at support → bullish, at resistance → bearish
                if lv["role"] == "support":
                    bull_score += strength * wt / 100 * 0.7
                elif lv["role"] == "resistance":
                    bear_score += strength * wt / 100 * 0.7
            elif pattern == "EXHAUSTION":
                # Mild contrarian hint
                if lv["role"] == "resistance" and price_near:
                    bear_score += 0.2
                elif lv["role"] == "support" and price_near:
                    bull_score += 0.2

            level_results.append({
                "name": lv["name"],
                "price": lvl_price,
                "role": lv["role"],
                "net_delta": round(net_delta, 1),
                "buy_vol": round(buy_vol, 1),
                "sell_vol": round(sell_vol, 1),
                "pattern": pattern,
                "strength": strength,
                "distance": round(distance, 2),
            })

        # ----------------------------------------------------------
        # 4. Aggregate bias
        # ----------------------------------------------------------
        net = bull_score - bear_score
        total_score = bull_score + bear_score
        if total_score == 0:
            bias = "NEUTRAL"
            bias_pct = 0
        elif net > 0.5:
            bias = "BULLISH"
            bias_pct = int(min(100, net / max(total_score, 1) * 100))
        elif net < -0.5:
            bias = "BEARISH"
            bias_pct = int(max(-100, net / max(total_score, 1) * 100))
        else:
            bias = "NEUTRAL"
            bias_pct = int(net / max(total_score, 1) * 100)

        # Find strongest actionable signal
        actionable = [lv for lv in level_results if lv["pattern"] not in ("NEUTRAL", "EXHAUSTION")]
        actionable.sort(key=lambda x: x["strength"], reverse=True)
        if actionable:
            top = actionable[0]
            strongest = f"{top['pattern']} at {top['name']} ({top['price']:.0f}, str={top['strength']})"
        else:
            strongest = "No strong level signals"

        # Build signal summary
        active_patterns = [lv for lv in level_results if lv["pattern"] != "NEUTRAL"]
        if active_patterns:
            parts = [f"{lv['name']}: {lv['pattern']}({lv['strength']})" for lv in active_patterns[:4]]
            signal = " | ".join(parts)
        else:
            signal = "No active level patterns"

        return {
            "levels": level_results,
            "net_bias": bias,
            "bias_score": bias_pct,
            "strongest_signal": strongest,
            "signal": signal,
            "source": source,
        }

    except Exception as e:
        logger.warning(f"Delta at levels failed: {e}", exc_info=True)
        return _delta_at_levels_empty()
