# File: Rank_MACD_PreCross_1to2Day.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import time as dtime

# ===================== CONFIG =====================
DATA_DIR      = "ALPACA_SC_DAILY_DATA"
TICKERS_CSV   = "SmallCap.csv"        # must contain at least: Ticker; optional: FloatShares
BENCH_TICKER  = "SPY"
OUTPUT_CSV    = "SC_macd_pre_cross_ranked_PM.csv"

LOOKBACK_CAL_DAYS = 75                # ~2.5 months (calendar) is fine for MACD(12,26,9)
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ALLOW_PENDING_TODAY = True            # allow “today” as the event if still negative & rising

# Optional relaxed checks for “pending today” (start strict; set small tolerances if needed)
NEAR_ZERO_EPS   = 0.0                 # treat h ≥ -eps as negative (e.g., 0.0005)
RUN_MAX_ABS_TOL = 0.0                 # absolute wiggle for “highest of run”
RUN_MAX_REL_TOL = 0.0                 # relative wiggle as fraction of |run_max| (e.g., 0.05 = 5%)

# Quality thresholds (used only in score; NOT for filtering)
VOL_SURGE_MIN    = 1.5
CLOSE_UPPER_QTL  = 0.70
TR_EXPAND_MIN    = 1.20
ATR_PCT_RANGE    = (0.02, 0.07)
ROOM_TO_RUN_ATR  = 1.0
RR_MIN           = 1.5

# Build a provisional "today" daily candle from 5-min **RTH only** (no premarket)
INCLUDE_TODAY_FROM_INTRADAY = False
INTRADAY_DIR = "ALPACA_SC_DATA"
TZ_EU = "Europe/Amsterdam"
TZ_NY = "America/New_York"

# ===================== CONFIG (add) =====================
PREMARKET_DIR = "ALPACA_PM_SC_DATA"   # your folder with premarket 5m bars (CET time)
PM_USE = True

# Pre-market ranking thresholds (tune as you like)
PM_GAP_MIN        = 0.005   # +0.5% gap vs. yesterday close
PM_REL_VOL_MIN    = 0.30    # PM volume at least 30% of 20d avg daily volume
PM_TR_ATR_MIN     = 0.25    # PM range at least 0.25x ATR20
PM_CLOSE_POS_MIN  = 0.60    # PM close in upper 40% of its PM range
PM_FLOAT_TURN_MIN = 0.0002  # 0.02% of float traded in PM (if float available)

# weight mix for a concise premarket power score (0..1)
PM_W_ENERGY  = 0.45  # gap + PM range/ATR
PM_W_FLOW    = 0.25  # PM rel vol + float turnover
PM_W_LEVELS  = 0.20  # relation to yesterday high / EMAs
PM_W_POSITION= 0.10  # closing near PM high
# ========================================================

# ---------- SIMULATION CLOCK ----------
# If True, the script pretends the current NY time is SIM_NOW_NY.
SIMULATE_PREOPEN = True
# Choose the session you want to simulate and a pre-open time (before 09:30).
# Example: simulate Aug 22, 2025 at 09:15 New York time.
SIM_NOW_NY = "2025-08-22 09:15"   # format: YYYY-MM-DD HH:MM (NY time)
# -------------------------------------

# Weights for the score (kept for convenience; NO filtering anywhere)
W = {
    "price_above_20": 1.0,
    "price_above_50": 0.7,
    "ema20_slope_pos": 1.0,
    "vol_surge": 1.2,
    "close_high_pct": 0.8,
    "tr_expand": 0.9,
    "atr_sweet": 0.6,
    "rs_slope_pos": 0.8,
    "room_run": 1.0,
    "rr_ok": 1.2,
    "hist_closeness": 0.5,
    "pending_penalty": -0.2
}

# Debug controls
DEBUG = True
DEBUG_TICKERS = []          # e.g. ["PHM","COIN"]
DEBUG_SAMPLE  = 12
# ===================================================

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii","replace").decode("ascii"), flush=True)

def _now_ny():
    if SIMULATE_PREOPEN and SIM_NOW_NY:
        return pd.Timestamp(SIM_NOW_NY, tz=TZ_NY)
    return pd.Timestamp.now(tz=TZ_NY)

def _today_ny_date():
    return _now_ny().date()



# ---------- Load tickers + float shares from LargeCap.csv ----------
def load_tickers_and_floats(csv_path: str):
    df = pd.read_csv(csv_path)
    # ticker column
    col_tk = None
    for c in ["Ticker","ticker","Symbol","symbol","SYM"]:
        if c in df.columns:
            col_tk = c
            break
    if not col_tk:
        raise ValueError(f"No ticker column in {csv_path}. Columns: {list(df.columns)}")
    df[col_tk] = df[col_tk].astype(str).str.strip()
    tickers = df[col_tk][(df[col_tk]!="") & df[col_tk].notna()].unique().tolist()

    # float shares column (optional)
    col_fs = None
    for c in ["FloatShares","Float_Shares","Float","SharesFloat","free_float","FreeFloat"]:
        if c in df.columns:
            col_fs = c
            break
    float_map = None
    if col_fs:
        try:
            f = df[[col_tk, col_fs]].dropna()
            # keep zeros (0 means “unknown or zero” → we won’t divide by it)
            f[col_fs] = pd.to_numeric(f[col_fs], errors="coerce")
            float_map = dict(zip(f[col_tk], f[col_fs]))
            log(f"Loaded float shares from {csv_path} ({len(float_map)} tickers with numeric values).")
        except Exception as e:
            log(f"Warning: failed parsing FloatShares from {csv_path}: {e}")
            float_map = None

    return tickers, float_map

# ---------- diagnostics ----------
TODAY_ATTEMPTED = 0
TODAY_APPENDED  = 0
TODAY_EMPTY     = 0
TODAY_BEFORE_RTH= 0
EVENT_TODAY     = 0
EVENT_PENDING   = 0
FAIL_SAMPLES    = []

def _load_intraday_today_ohlcv(ticker: str):
    """Aggregate today's 5-min RTH bars (09:30–now ET) into a provisional daily OHLCV."""
    fp = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
    if not os.path.exists(fp):
        return None

    dfm = pd.read_csv(fp, parse_dates=["Date"])
    dfm.rename(columns={"Date":"dt","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    try:
        dfm["dt"] = dfm["dt"].dt.tz_localize(TZ_EU, ambiguous="infer").dt.tz_convert(TZ_NY)
    except Exception:
        dfm["dt"] = dfm["dt"].dt.tz_convert(TZ_NY)
    dfm.set_index("dt", inplace=True)

    # now_ny   = pd.Timestamp.now(tz=TZ_NY)
    # today_ny = now_ny.date()

    now_ny   = _now_ny()
    today_ny = _today_ny_date()

    if now_ny.time() < dtime(9,30):
        return {"before_rth": True}

    day = dfm[dfm.index.date == today_ny].between_time(dtime(9,30), now_ny.time())
    if day.empty:
        return None

    return {
        "date":   pd.Timestamp(today_ny).tz_localize(None),
        "open":   float(day["open"].iloc[0]),
        "high":   float(day["high"].max()),
        "low":    float(day["low"].min()),
        "close":  float(day["close"].iloc[-1]),
        "volume": float(day["volume"].sum()),
    }

def _augment_daily_with_today(df_daily: pd.DataFrame, ticker: str) -> pd.DataFrame:
    global TODAY_ATTEMPTED, TODAY_APPENDED, TODAY_EMPTY, TODAY_BEFORE_RTH
    if not INCLUDE_TODAY_FROM_INTRADAY:
        return df_daily

    TODAY_ATTEMPTED += 1
    today_bar = _load_intraday_today_ohlcv(ticker)

    if today_bar is None:
        TODAY_EMPTY += 1
        return df_daily
    if isinstance(today_bar, dict) and today_bar.get("before_rth"):
        TODAY_BEFORE_RTH += 1
        return df_daily

    df = df_daily.copy()
    idx = today_bar["date"]
    df.loc[idx, ["open","high","low","close","volume"]] = [
        today_bar["open"], today_bar["high"], today_bar["low"], today_bar["close"], today_bar["volume"]
    ]
    df.sort_index(inplace=True)
    TODAY_APPENDED += 1
    return df

def load_daily(ticker: str) -> pd.DataFrame:
    fp = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()

    df = pd.read_csv(fp, parse_dates=["Date"])
    df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    if INCLUDE_TODAY_FROM_INTRADAY:
        df = _augment_daily_with_today(df, ticker)

    if LOOKBACK_CAL_DAYS:
        #cutoff = (pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None) - pd.Timedelta(days=LOOKBACK_CAL_DAYS))
        cutoff = (_now_ny().normalize().tz_localize(None) - pd.Timedelta(days=LOOKBACK_CAL_DAYS))
        df = df[df.index >= cutoff]

    return df

def ema(s, span): 
    return s.ewm(span=span, adjust=False).mean()

def macd_hist(close):
    fast = ema(close, MACD_FAST); slow = ema(close, MACD_SLOW)
    macd = fast - slow; signal = ema(macd, MACD_SIG)
    return macd, signal, macd - signal

def true_range(df: pd.DataFrame):
    prev_close = df["close"].shift(1)
    x1 = (df["high"] - df["low"]).abs()
    x2 = (df["high"] - prev_close).abs()
    x3 = (df["low"] - prev_close).abs()
    return pd.concat([x1, x2, x3], axis=1).max(axis=1)

def atr(df, n=20): return true_range(df).rolling(n).mean()

def close_in_range_pct(row):
    rng = row["high"] - row["low"]
    if rng <= 0: return 1.0
    return (row["close"] - row["low"]) / rng

def rs_slope_pos(price: pd.Series, bench: pd.Series, lookback=15):
    aligned = price.align(bench, join="inner")
    if aligned[0].empty or len(aligned[0]) < lookback + 1: return np.nan
    rs = aligned[0] / aligned[1]
    try:
        v = (rs.iloc[-1] / rs.iloc[-1 - lookback]) - 1.0
        return float(v > 0)
    except Exception:
        return np.nan

# ---------- Event detection (prefer pending-today if valid) ----------
def _is_highest_of_run(hist: pd.Series, cand):
    h_c = float(hist.loc[cand])
    nonneg_before = (hist[:cand] >= 0)
    if nonneg_before.any():
        last_nonneg = nonneg_before[::-1].idxmax()
        if nonneg_before.loc[last_nonneg]:
            start_loc = min(hist.index.get_loc(last_nonneg) + 1, hist.index.get_loc(cand))
            neg_run_start = hist.index[start_loc]
        else:
            neg_run_start = hist.index[0]
    else:
        neg_run_start = hist.index[0]
    neg_run = hist.loc[neg_run_start:cand]
    neg_run = neg_run[neg_run < 0]
    if neg_run.empty: return (False, h_c, np.nan)
    run_max = float(neg_run.max())
    tol = max(RUN_MAX_ABS_TOL, RUN_MAX_REL_TOL * abs(run_max))
    return (h_c >= run_max - tol, h_c, run_max)

def last_neg_green_event(hist: pd.Series, allow_pending=True):
    if len(hist) < 5:
        return None

    def _neg_run_bounds(end_loc: int):
        # find start of contiguous negative run that ends at end_loc-1
        j = end_loc - 1
        if j < 0:
            return None, None
        # walk back until we hit a non-negative (>=0) bar
        start_loc = 0
        for t in range(j, -1, -1):
            if hist.iloc[t] >= 0:
                start_loc = t + 1
                break
        return start_loc, j  # inclusive start, inclusive end

    # -------- 1) Pending today (still negative & rising) --------
    if allow_pending:
        cand_idx = hist.index[-1]
        h_cand = float(hist.iloc[-1])
        if h_cand <= NEAR_ZERO_EPS and len(hist) >= 2:
            h_prev = float(hist.iloc[-2])
            if h_cand > h_prev:  # rising today
                # ensure today's bar is the highest (least negative) within the current negative run
                start_loc, end_loc = _neg_run_bounds(len(hist))
                if start_loc is not None and start_loc <= end_loc:
                    run = hist.iloc[start_loc:end_loc+1]
                    run = run[run < 0]
                    if not run.empty:
                        run_max = float(run.max())
                        tol = max(RUN_MAX_ABS_TOL, RUN_MAX_REL_TOL * abs(run_max))
                        if h_cand >= run_max - tol:
                            # "green vs previous" should be tested within the run if possible
                            neg_idx = run.index
                            # if cand is not the first bar inside the negative run, check green vs prior negative
                            if len(neg_idx) >= 2 and cand_idx != neg_idx[0]:
                                prev_neg = neg_idx[neg_idx.get_loc(cand_idx) - 1]
                                if not (h_cand > float(hist.loc[prev_neg]) - NEAR_ZERO_EPS):
                                    return None
                            # if cand is first negative (1-bar run), accept
                            return {"event_date": cand_idx,
                                    "cross_date": None,
                                    "hist_event": h_cand,
                                    "pending_cross": True}

    # -------- 2) Latest up-cross: pick the highest-negative inside the run BEFORE the cross --------
    positive = hist > 0
    cross_up = positive & (~positive.shift(1, fill_value=False))
    if not cross_up.any():
        return None

    cross_idx = hist.index[cross_up][-1]
    i = hist.index.get_loc(cross_idx)  # first positive bar of the cross

    # bounds of the just-finished negative run
    start_loc, end_loc = _neg_run_bounds(i)
    if start_loc is None or start_loc > end_loc:
        return None

    run = hist.iloc[start_loc:end_loc+1]
    run = run[run < 0]
    if run.empty:
        return None

    # candidate = highest (least negative) bar inside that run
    cand_idx = run.idxmax()
    h_cand = float(hist.loc[cand_idx])

    # "green vs previous" ONLY within the negative run.
    neg_idx = run.index
    pos_in_run = neg_idx.get_loc(cand_idx)
    if pos_in_run > 0:
        prev_neg = neg_idx[pos_in_run - 1]
        if not (h_cand > float(hist.loc[prev_neg]) - NEAR_ZERO_EPS):
            return None
    # if cand is the first bar of the run (1-bar run), accept

    return {"event_date": cand_idx,
            "cross_date": cross_idx,
            "hist_event": h_cand,
            "pending_cross": False}

def _load_intraday_today_df(ticker: str):
    """
    Return today's 5-min **RTH** bars (NY time) as a DataFrame indexed by NY tz.
    Use this for intraday volume KPIs (open/mid/close sessions).
    """
    fp = os.path.join(INTRADAY_DIR, f"{ticker}.csv")
    if not os.path.exists(fp):
        return None

    df5 = pd.read_csv(fp, parse_dates=["Date"])
    df5.rename(columns={"Date":"dt","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)

    try:
        df5["dt"] = df5["dt"].dt.tz_localize(TZ_EU, ambiguous="infer").dt.tz_convert(TZ_NY)
    except Exception:
        df5["dt"] = df5["dt"].dt.tz_convert(TZ_NY)

    df5.set_index("dt", inplace=True)

    now_ny = _now_ny()
    today_ny = _today_ny_date()

    # Limit to today’s RTH window. If we’re simulating preopen, just return empty.
    if now_ny.time() < dtime(9,30):
        return None

    end_t = min(now_ny.time(), dtime(16,0))
    rth = df5[df5.index.date == today_ny].between_time(dtime(9,30), end_t)
    return None if rth.empty else rth


######ADDING PRE-MARKET SECTION #########################

def _load_premarket_df(ticker: str):
    """
    Load today's pre-market 5m bars from PREMARKET_DIR and return the CET-indexed
    frame converted to New York tz, filtered to 04:00–09:29:59 ET.
    """
    if not PM_USE:
        return None

    fp = os.path.join(PREMARKET_DIR, f"{ticker}.csv")
    if not os.path.exists(fp):
        return None

    df = pd.read_csv(fp, parse_dates=["Date"])
    df.rename(columns={
        "Date":"dt","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    }, inplace=True)

    if df.empty:
        return None

    # localize the CET timestamps you store, then convert to New York
    try:
        df["dt"] = df["dt"].dt.tz_localize(TZ_EU, ambiguous="infer").dt.tz_convert(TZ_NY)
    except Exception:
        # if already tz-aware as CET, just convert
        df["dt"] = df["dt"].dt.tz_convert(TZ_NY)

    df.set_index("dt", inplace=True)

    # now_ny   = pd.Timestamp.now(tz=TZ_NY)
    # today_ny = now_ny.date()

    now_ny   = _now_ny()
    today_ny = _today_ny_date()


    pm = df[df.index.date == today_ny].between_time(dtime(4,0), dtime(9,29,59))
    return None if pm.empty else pm

def _aggregate_pm_ohlcv(pm: pd.DataFrame):
    """Aggregate a pre-market 5m dataframe into a single PM session OHLCV + VWAP."""
    if pm is None or pm.empty:
        return None

    # VWAP using typical price
    tp = (pm["high"] + pm["low"] + pm["close"]) / 3.0
    pv = (tp * pm["volume"]).sum()
    vv = pm["volume"].sum()
    vwap = float(pv / vv) if vv > 0 else float("nan")

    return {
        "open":   float(pm["open"].iloc[0]),
        "high":   float(pm["high"].max()),
        "low":    float(pm["low"].min()),
        "close":  float(pm["close"].iloc[-1]),
        "volume": float(vv),
        "vwap":   vwap,
    }


# ---------- Feature helpers ----------
def overhead_supply_ratio(df, ed, lookback=40):
    """
    'Overhead supply' = fraction of volume in the last `lookback` sessions
    where the CLOSE was ABOVE the event-day close. Holders who paid more than
    today are more likely to sell on rallies to 'get back to even'.
    Lower is better (less potential supply overhead).
    """
    if ed not in df.index: return np.nan
    end_loc = df.index.get_loc(ed)
    start_loc = max(0, end_loc - lookback + 1)
    win = df.iloc[start_loc:end_loc+1]
    if win.empty: return np.nan
    event_close = df.loc[ed, "close"]
    v_above = win.loc[win["close"] > event_close, "volume"].sum()
    v_tot   = win["volume"].sum()
    return float(v_above / v_tot) if v_tot > 0 else np.nan

def find_swing_low_idx(close, ed, lookback=30):
    if ed not in close.index: return None
    end_loc = close.index.get_loc(ed)
    start_loc = max(0, end_loc - lookback)
    window = close.iloc[start_loc:end_loc+1]
    if window.empty: return None
    return window.idxmin()

def anchored_vwap(df, anchor_idx, ed):
    if anchor_idx is None or ed not in df.index or anchor_idx not in df.index: return np.nan
    s = df.loc[anchor_idx:ed]
    tp = (s["high"] + s["low"] + s["close"]) / 3.0
    cum_pv = (tp * s["volume"]).cumsum()
    cum_v  = s["volume"].cumsum()
    return float(cum_pv.iloc[-1] / cum_v.iloc[-1]) if cum_v.iloc[-1] > 0 else np.nan

def bollinger_width_pct(close, n=20):
    m = close.rolling(n).mean(); sd = close.rolling(n).std()
    upper, lower = m + 2*sd, m - 2*sd
    return (upper - lower) / close

# ---------- Volume Related Info --------- #
def _green_red_volume(df5):
    """Return (green_vol, red_vol, green_ratio)."""
    if df5 is None or df5.empty:
        return (np.nan, np.nan, np.nan)
    df5 = df5.copy()
    df5["is_green"] = (df5["close"] > df5["open"]).astype(int)
    g = df5.loc[df5["is_green"]==1, "volume"].sum()
    r = df5.loc[df5["is_green"]==0, "volume"].sum()
    denom = g + r
    return g, r, (g/denom if denom > 0 else np.nan)


def _intraday_vwap(df5):
    """VWAP over a session (intraday bars)."""
    if df5 is None or df5.empty:
        return np.nan
    tp = (df5["high"] + df5["low"] + df5["close"]) / 3.0
    return ((tp * df5["volume"]).sum() / df5["volume"].sum()) if df5["volume"].sum() > 0 else np.nan


def _session_rvol(df5, v20_daily, start_t, end_t):
    """
    Relative volume for a session window vs 20d daily volume.
    Scaled by expected fraction of daily trading minutes (390).
    """
    if df5 is None or df5.empty or not np.isfinite(v20_daily) or v20_daily <= 0:
        return np.nan
    sess = df5.between_time(start_t, end_t)
    if sess is None or sess.empty:
        return np.nan
    sess_vol = float(sess["volume"].sum())
    minutes = len(sess) * 5  # each bar is 5 minutes
    exp_vol = float(v20_daily * (minutes / 390.0))
    return sess_vol / exp_vol if exp_vol > 0 else np.nan


# ---------- Features & scoring ----------
def features_for_event(df: pd.DataFrame, bench_df: pd.DataFrame | None, ticker: str, float_map: dict | None):
    global EVENT_TODAY, EVENT_PENDING, FAIL_SAMPLES

    if df.empty or "close" not in df.columns: return None

    macd, sig, hist = macd_hist(df["close"])
    evt = last_neg_green_event(hist, allow_pending=ALLOW_PENDING_TODAY)
    if not evt:
        if DEBUG and len(FAIL_SAMPLES) < DEBUG_SAMPLE:
            #today_idx = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
            today_idx = _now_ny().normalize().tz_localize(None)
            in_idx = today_idx in df.index
            if in_idx and len(hist) >= 2:
                h0 = float(hist.iloc[-1]); h1 = float(hist.iloc[-2])
                ok_run, _, run_max_tmp = _is_highest_of_run(hist, hist.index[-1])
                reasons = []
                if not (h0 <= NEAR_ZERO_EPS): reasons.append("today_not_negative")
                if not (h0 > h1): reasons.append("not_green")
                if not ok_run: reasons.append(f"not_run_max(delta={run_max_tmp - h0:.6f})")
                if not reasons: reasons = ["other"]
                FAIL_SAMPLES.append(f"[{ticker}] today check: h[-2]={h1:.5f}, h[-1]={h0:.5f} | {','.join(reasons)}")
            else:
                FAIL_SAMPLES.append(f"[{ticker}] no-event | today_in_index={in_idx} | tail={', '.join(f'{x:.5f}' for x in hist.tail(3).values)}")
        return None

    ed = evt["event_date"]
    if ed not in df.index: return None

    # mark if event is today / pending
    #today_naive = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
    today_naive = _now_ny().normalize().tz_localize(None)
    if ed == today_naive:
        EVENT_TODAY += 1
        if evt["pending_cross"]: EVENT_PENDING += 1

    if ticker in DEBUG_TICKERS:
        log(f"[DEBUG {ticker}] EVENT={ed.date()}  CROSS={'PENDING' if evt['pending_cross'] else evt['cross_date'].date() if evt['cross_date'] is not None else 'None'} "
            f"| hist_event={evt['hist_event']:.5f} | hist_tail={', '.join(f'{x:.5f}' for x in hist.tail(4).values)}")

    row = df.loc[ed]
    # latest row in your daily file (so you always get the freshest close)
    last_row = df.iloc[-1]
    last_close = float(last_row["close"])
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)

    # --- Liquidity/volumes (all as features; no filtering) ---
    v20 = df["volume"].rolling(20).mean().loc[ed] if len(df) >= 20 else np.nan
    vol = float(row["volume"]) if not pd.isna(row["volume"]) else np.nan
    rel_vol_20 = float(vol / v20) if (np.isfinite(vol) and np.isfinite(v20) and v20>0) else np.nan
    # Highest in recent windows?
    vol_highest_10 = float(vol >= df["volume"].shift(1).rolling(10).max().loc[ed]) if len(df) >= 11 and np.isfinite(vol) else np.nan
    vol_highest_5  = float(vol >= df["volume"].shift(1).rolling(5).max().loc[ed])  if len(df) >= 6  and np.isfinite(vol) else np.nan
    # Rank & percentile
    def _rank_in_window(series, ed, win):
        if ed not in series.index: return np.nan
        end = series.index.get_loc(ed); start = max(0, end - win + 1)
        w = series.iloc[start:end+1]
        if w.empty or pd.isna(series.loc[ed]): return np.nan
        return float((w.rank(ascending=False, method="min").loc[ed]))
    vol_rank_10 = _rank_in_window(df["volume"], ed, 10)
    vol_rank_20 = _rank_in_window(df["volume"], ed, 20)
    vol_pctile_20 = float(100.0 * (21 - vol_rank_20) / 20.0) if vol_rank_20 == vol_rank_20 else np.nan

    # --- Green price bar + green volume bar ---
    prev_close = df["close"].shift(1).loc[ed] if ed in df.index else np.nan
    price_green = float(row["close"] >= row["open"]) if (np.isfinite(row["close"]) and np.isfinite(row["open"])) else np.nan
    vol_green   = float(row["close"] >= prev_close) if (np.isfinite(row["close"]) and np.isfinite(prev_close)) else np.nan
    green_price_and_vol = float(price_green==1.0 and vol_green==1.0) if (price_green==price_green and vol_green==vol_green) else np.nan

    # --- Close position + range expansion + ATR% ---
    close_pct = close_in_range_pct({"high": row["high"], "low": row["low"], "close": row["close"]})
    atr20 = atr(df, 20).loc[ed] if len(df) >= 21 else np.nan
    today_tr = true_range(df).loc[ed] if ed in df.index else np.nan
    tr_ratio = float(today_tr / atr20) if (np.isfinite(today_tr) and np.isfinite(atr20) and atr20 > 0) else np.nan
    atr_pct = float(atr20 / row["close"]) if (np.isfinite(atr20) and row["close"] > 0) else np.nan

    # --- Relative strength vs benchmark (optional) ---
    if bench_df is not None and not bench_df.empty:
        rs_pos = rs_slope_pos(df["close"], bench_df["close"], lookback=15)
    else:
        rs_pos = np.nan

    # --- Room to run / R:R ---
    high20 = df["high"].rolling(20).max().shift(1).loc[ed] if ed in df.index else np.nan
    room_atr = float((high20 - row["close"]) / atr20) if (np.isfinite(high20) and np.isfinite(atr20) and atr20 > 0) else np.nan
    stop = min(float(row["low"]), float(ema20.loc[ed])) if ed in ema20.index else float(row["low"])
    risk = row["close"] - stop
    reward = (high20 - row["close"]) if np.isfinite(high20) else np.nan
    rr = float(reward / risk) if (np.isfinite(reward) and risk > 0 and reward > 0) else np.nan

    # --- Trend / EMA locations ---
    price_above_20 = float(row["close"] > ema20.loc[ed]) if ed in ema20.index else np.nan
    price_above_50 = float(row["close"] > ema50.loc[ed]) if ed in ema50.index else np.nan
    ema20_slope_pos = float(ema20.loc[ed] > ema20.shift(5).loc[ed]) if ed in ema20.index and ed in ema20.shift(5).index else np.nan
    # EMA50 location relative to the bar
    if ed in ema50.index:
        e50 = float(ema50.loc[ed])
        if e50 < row["low"]: ema50_pos = "below_bar"; ema50_below_bar = 1.0; ema50_cut_bar = 0.0
        elif e50 > row["high"]: ema50_pos = "above_bar"; ema50_below_bar = 0.0; ema50_cut_bar = 0.0
        else: ema50_pos = "inside_bar"; ema50_below_bar = 0.0; ema50_cut_bar = 1.0
        dist_close_ema50_atr = float((row["close"] - e50) / atr20) if np.isfinite(atr20) and atr20>0 else np.nan
    else:
        ema50_pos = None; ema50_below_bar = np.nan; ema50_cut_bar = np.nan; dist_close_ema50_atr = np.nan

    # --- Overhead supply + anchored VWAP context ---
    overhead_supply_40 = overhead_supply_ratio(df, ed, lookback=40)
    overhead_supply_20 = overhead_supply_ratio(df, ed, lookback=20)
    anchor = find_swing_low_idx(df["close"], ed, lookback=30)
    avwap  = anchored_vwap(df, anchor, ed)
    price_above_avwap = float(row["close"] > avwap) if np.isfinite(avwap) else np.nan
    dist_avwap_atr = float((row["close"] - avwap) / atr20) if (np.isfinite(avwap) and np.isfinite(atr20) and atr20>0) else np.nan

    # --- Float turnover (from LargeCap.csv) ---
    if float_map is not None and ticker in float_map and float_map[ticker] is not None and float_map[ticker] > 0:
        fs = float(float_map[ticker])
        float_turnover_pct = float(vol / fs) if np.isfinite(vol) else np.nan
        float_turnover20_pct = float(v20 / fs) if (np.isfinite(v20) and fs > 0) else np.nan
        float_shares_millions = float(fs / 1_000_000.0)
    else:
        float_turnover_pct = np.nan; float_turnover20_pct = np.nan; float_shares_millions = np.nan

    # --- Histogram closeness to zero (signal quality) ---
    hist_value = float(evt["hist_event"])
    hist_closeness = float(-hist_value)

    # Score (unchanged logic; just uses some of the above for convenience)
    score = (
        W["price_above_20"] * (0.0 if np.isnan(price_above_20) else price_above_20) +
        W["price_above_50"] * (0.0 if np.isnan(price_above_50) else price_above_50) +
        W["ema20_slope_pos"] * (0.0 if np.isnan(ema20_slope_pos) else ema20_slope_pos) +
        W["vol_surge"] * (1.0 if (np.isfinite(rel_vol_20) and rel_vol_20 >= VOL_SURGE_MIN) else 0.0) +
        W["close_high_pct"] * (1.0 if close_pct >= CLOSE_UPPER_QTL else 0.0) +
        W["tr_expand"] * (1.0 if (np.isfinite(tr_ratio) and tr_ratio >= TR_EXPAND_MIN) else 0.0) +
        W["atr_sweet"] * (1.0 if (np.isfinite(atr_pct) and ATR_PCT_RANGE[0] <= atr_pct <= ATR_PCT_RANGE[1]) else 0.0) +
        W["rs_slope_pos"] * (0.0 if np.isnan(rs_pos) else rs_pos) +
        W["room_run"] * (1.0 if (np.isfinite(room_atr) and room_atr >= ROOM_TO_RUN_ATR) else 0.0) +
        W["rr_ok"] * (1.0 if (np.isfinite(rr) and rr >= RR_MIN) else 0.0) +
        W["hist_closeness"] * np.tanh(max(0.0, hist_closeness) / 0.05) +
        (W["pending_penalty"] if evt["pending_cross"] else 0.0)
    )

    # ------------------------------------------------------------------
    # APPEND: "AS OF TODAY" KPIs (no changes to existing logic above)
    # ------------------------------------------------------------------
    t_idx = df.index[-1]
    row_t = df.iloc[-1]
    # today basics & EMAs
    v20_t = df["volume"].rolling(20).mean().loc[t_idx] if len(df) >= 20 else np.nan
    vol_t = float(row_t["volume"]) if not pd.isna(row_t["volume"]) else np.nan
    rel_vol_20_today = float(vol_t / v20_t) if (np.isfinite(vol_t) and np.isfinite(v20_t) and v20_t > 0) else np.nan

    vol_highest_10_today = float(vol_t >= df["volume"].shift(1).rolling(10).max().loc[t_idx]) if len(df) >= 11 and np.isfinite(vol_t) else np.nan
    vol_highest_5_today  = float(vol_t >= df["volume"].shift(1).rolling(5).max().loc[t_idx])  if len(df) >= 6  and np.isfinite(vol_t) else np.nan
    # rank/percentile today
    def _rank_today(series, idx, win):
        if idx not in series.index: return np.nan
        end = series.index.get_loc(idx); start = max(0, end - win + 1)
        w = series.iloc[start:end+1]
        if w.empty or pd.isna(series.loc[idx]): return np.nan
        return float((w.rank(ascending=False, method="min").loc[idx]))
    vol_rank_20_today = _rank_today(df["volume"], t_idx, 20)
    vol_pctile_20_today = float(100.0 * (21 - vol_rank_20_today) / 20.0) if vol_rank_20_today == vol_rank_20_today else np.nan

    # green price & green volume today
    prev_close_t = df["close"].shift(1).loc[t_idx] if t_idx in df.index else np.nan
    price_green_today = float(row_t["close"] >= row_t["open"]) if (np.isfinite(row_t["close"]) and np.isfinite(row_t["open"])) else np.nan
    vol_green_today   = float(row_t["close"] >= prev_close_t) if (np.isfinite(row_t["close"]) and np.isfinite(prev_close_t)) else np.nan
    green_price_and_vol_today = float(price_green_today==1.0 and vol_green_today==1.0) if (price_green_today==price_green_today and vol_green_today==vol_green_today) else np.nan

    # range / ATR today
    atr20_t = atr(df, 20).loc[t_idx] if len(df) >= 21 else np.nan
    tr_today = true_range(df).loc[t_idx] if t_idx in df.index else np.nan
    tr_ratio_today = float(tr_today / atr20_t) if (np.isfinite(tr_today) and np.isfinite(atr20_t) and atr20_t > 0) else np.nan
    atr_pct_today = float(atr20_t / row_t["close"]) if (np.isfinite(atr20_t) and row_t["close"] > 0) else np.nan

    # close position in range today
    rng = row_t["high"] - row_t["low"]
    close_pct_in_range_today = float((row_t["close"] - row_t["low"]) / rng) if rng > 0 else 1.0

    # RS (today window end)
    if bench_df is not None and not bench_df.empty:
        rs_slope_pos_today = rs_slope_pos(df["close"], bench_df["close"], lookback=15)
    else:
        rs_slope_pos_today = np.nan

    # room & RR today
    high20_today = df["high"].rolling(20).max().shift(1).loc[t_idx] if t_idx in df.index else np.nan
    room_atr_today = float((high20_today - row_t["close"]) / atr20_t) if (np.isfinite(high20_today) and np.isfinite(atr20_t) and atr20_t > 0) else np.nan
    ema20_t = ema20.loc[t_idx] if t_idx in ema20.index else np.nan
    stop_t = min(float(row_t["low"]), float(ema20_t)) if np.isfinite(ema20_t) else float(row_t["low"])
    risk_t = row_t["close"] - stop_t
    reward_t = (high20_today - row_t["close"]) if np.isfinite(high20_today) else np.nan
    rr_today = float(reward_t / risk_t) if (np.isfinite(reward_t) and risk_t > 0 and reward_t > 0) else np.nan

    # EMA50 placement today
    if t_idx in ema50.index:
        e50_t = float(ema50.loc[t_idx])
        if e50_t < row_t["low"]: ema50_pos_today = "below_bar"; ema50_below_bar_today = 1.0; ema50_cut_bar_today = 0.0
        elif e50_t > row_t["high"]: ema50_pos_today = "above_bar"; ema50_below_bar_today = 0.0; ema50_cut_bar_today = 0.0
        else: ema50_pos_today = "inside_bar"; ema50_below_bar_today = 0.0; ema50_cut_bar_today = 1.0
        dist_close_ema50_atr_today = float((row_t["close"] - e50_t) / atr20_t) if np.isfinite(atr20_t) and atr20_t>0 else np.nan
    else:
        ema50_pos_today = ""
        ema50_below_bar_today = np.nan
        ema50_cut_bar_today = np.nan
        dist_close_ema50_atr_today = np.nan

    # overhead supply & AVWAP today
    overhead_supply_40_today = overhead_supply_ratio(df, t_idx, lookback=40)
    overhead_supply_20_today = overhead_supply_ratio(df, t_idx, lookback=20)
    anchor_t = find_swing_low_idx(df["close"], t_idx, lookback=30)
    avwap_t  = anchored_vwap(df, anchor_t, t_idx)
    price_above_avwap_today = float(row_t["close"] > avwap_t) if np.isfinite(avwap_t) else np.nan
    dist_avwap_atr_today = float((row_t["close"] - avwap_t) / atr20_t) if (np.isfinite(avwap_t) and np.isfinite(atr20_t) and atr20_t>0) else np.nan

    # float turnover today
    if float_map is not None and ticker in float_map and float_map[ticker] is not None and float_map[ticker] > 0:
        fs = float(float_map[ticker])
        float_turnover_pct_today = float(vol_t / fs) if np.isfinite(vol_t) else np.nan
        float_turnover20_pct_today = float(v20_t / fs) if (np.isfinite(v20_t) and fs > 0) else np.nan
    else:
        float_turnover_pct_today = np.nan
        float_turnover20_pct_today = np.nan

    # power_score_today (parallel to your existing formula)
    def _clip01_local(x):
        try:
            return float(max(0.0, min(1.0, x)))
        except Exception:
            return 0.0
    def _sigmoid_local(x):
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except Exception:
            return 0.0

    energy_t = 0.6 * _sigmoid_local((rel_vol_20_today - 1.8)) \
             + 0.4 * _sigmoid_local(((tr_ratio_today - 1.2) / 0.4) if np.isfinite(tr_ratio_today) else -999)
    room_t = _clip01_local(((room_atr_today if np.isfinite(room_atr_today) else 0.0) / 1.5)) \
           + 0.4 * _clip01_local(1.0 - (overhead_supply_40_today if np.isfinite(overhead_supply_40_today) else 0.5))
    room_t = min(1.0, room_t)

    ema_bonus_t = _clip01_local(max(0.0, dist_close_ema50_atr_today if np.isfinite(dist_close_ema50_atr_today) else 0.0) / 1.0)
    avwap_bonus_t = _clip01_local(max(0.0, dist_avwap_atr_today if np.isfinite(dist_avwap_atr_today) else 0.0) / 0.7)
    ema_pen_t = 0.15 if str(ema50_pos_today) == "above_bar" else 0.0
    support_t = _clip01_local(0.5 * ema_bonus_t + 0.5 * avwap_bonus_t - ema_pen_t)

    ft_t = float_turnover_pct_today if np.isfinite(float_turnover_pct_today) else 0.0
    flow_turnover_t = _clip01_local((ft_t / 0.015) if ft_t > 0 else 0.0)
    flow_vol_t = _clip01_local((vol_pctile_20_today or 0.0) / 100.0 if np.isfinite(vol_pctile_20_today) else 0.0)
    flow_t = max(flow_turnover_t, flow_vol_t)

    power_today = 0.35*energy_t + 0.30*room_t + 0.20*support_t + 0.15*flow_t
    if green_price_and_vol_today == 1.0: power_today += 0.05
    power_score_today = _clip01_local(power_today)

    # ------------------------------------------------------------------

    # -------------------------Volume KPIS -------------------------
    rth_df = _load_intraday_today_df(ticker)

    if rth_df is not None:
        # --- Full day ---
        fd_g, fd_r, fd_gr = _green_red_volume(rth_df)
        fd_dom = float(fd_gr >= 0.55) if np.isfinite(fd_gr) else np.nan
        fd_vwap = _intraday_vwap(rth_df)

        # --- Open session (09:30–11:00) ---
        open_df = rth_df.between_time("09:30", "11:00")
        op_g, op_r, op_gr = _green_red_volume(open_df)
        op_dom = float(op_gr >= 0.55) if np.isfinite(op_gr) else np.nan
        op_rvol = _session_rvol(rth_df, v20_t, dtime(9,30), dtime(11,0))

        # --- Midday (11:00–14:00) ---
        mid_df = rth_df.between_time("11:00", "14:00")
        mid_g, mid_r, mid_gr = _green_red_volume(mid_df)
        mid_dom = float(mid_gr >= 0.55) if np.isfinite(mid_gr) else np.nan
        mid_rvol = _session_rvol(rth_df, v20_t, dtime(11,0), dtime(14,0))

        # --- Close (14:00–16:00) ---
        close_df = rth_df.between_time("14:00", "16:00")
        cl_g, cl_r, cl_gr = _green_red_volume(close_df)
        cl_dom = float(cl_gr >= 0.55) if np.isfinite(cl_gr) else np.nan
        cl_rvol = _session_rvol(rth_df, v20_t, dtime(14,0), dtime(16,0))
    else:
        fd_g = fd_r = fd_gr = fd_dom = fd_vwap = np.nan
        op_g = op_r = op_gr = op_dom = op_rvol = np.nan
        mid_g = mid_r = mid_gr = mid_dom = mid_rvol = np.nan
        cl_g = cl_r = cl_gr = cl_dom = cl_rvol = np.nan


    # ================= PRE-MARKET KPIs (new) =================
    pm_df = _load_premarket_df(ticker)
    pm_bar = _aggregate_pm_ohlcv(pm_df) if pm_df is not None else None

    # reference yesterday bar/levels for gaps & breakouts
    # NOTE: if today (RTH) was appended, t_idx is today; get yesterday safely.
    if len(df) >= 2:
        y_row = df.iloc[-2] if df.index[-1] == t_idx else df.iloc[-1]
    else:
        y_row = df.iloc[-1]  # fallback

    y_close = float(y_row["close"]) if "close" in y_row else np.nan
    y_high  = float(y_row["high"])  if "high"  in y_row else np.nan

    # --- Reference 'yesterday' consistently (do NOT include today) ---
    # If df has at least 2 rows, yesterday is the penultimate row.
    # This stays correct even if you later append a provisional 'today' row.
    if len(df) >= 2:
        y_idx = df.index[-2]
    else:
        y_idx = df.index[-1]  # fallback when there is only 1 row

    y_low = float(df.loc[y_idx, "low"]) if "low" in df.columns else np.nan

    # Windows that END at yesterday:
    # 2-day window = [day_before_yesterday, yesterday]
    y_loc = df.index.get_loc(y_idx)
    start2 = max(0, y_loc - 1)
    win2 = df["low"].iloc[start2:y_loc+1]   # includes y_idx

    # 5-day window = last up to 5 trading days ending at yesterday
    start5 = max(0, y_loc - 4)
    win5 = df["low"].iloc[start5:y_loc+1]   # includes y_idx

    prev_low_2d = float(np.isfinite(y_low) and (y_low == float(win2.min())))
    prev_low_5d = float(np.isfinite(y_low) and (y_low == float(win5.min())))


    pm_feats = {}  # ADD THIS before any pm_feats.update(...)
    if pm_df is not None:
        pm_g, pm_r, pm_gr = _green_red_volume(pm_df)
        pm_feats.update({
            "pm_green_vol": pm_g,
            "pm_red_vol": pm_r,
            "pm_green_ratio": pm_gr,
            "pm_buyer_dominance": float(pm_gr >= 0.55) if np.isfinite(pm_gr) else np.nan,
        })
    else:
        pm_feats.update({
            "pm_green_vol": np.nan,
            "pm_red_vol": np.nan,
            "pm_green_ratio": np.nan,
            "pm_buyer_dominance": np.nan,
        })

    if pm_bar is not None and np.isfinite(y_close):
        pm_range = pm_bar["high"] - pm_bar["low"]
        pm_close_pos = float((pm_bar["close"] - pm_bar["low"]) / pm_range) if pm_range > 0 else 1.0
        pm_gap_pct = float((pm_bar["close"] - y_close) / y_close)

        # PM range relative to ATR20 (using yesterday ATR, atr20 already computed)
        pm_tr_atr = float(pm_range / atr20_t) if (np.isfinite(atr20_t) and atr20_t > 0) else np.nan

        # PM volume vs 20d avg daily volume (coarse but predictive)
        pm_rel_vol_20 = float(pm_bar["volume"] / v20_t) if (np.isfinite(v20_t) and v20_t > 0) else np.nan

        # Float-based PM turnover if available
        if float_map is not None and ticker in float_map and float_map[ticker] and float_map[ticker] > 0:
            fs = float(float_map[ticker])
            pm_float_turnover = float(pm_bar["volume"] / fs)
        else:
            pm_float_turnover = np.nan

        # Levels / support context at PM close vs. daily EMAs (yesterday)
        ema20_y = float(ema(df["close"], 20).iloc[-1]) if len(df) >= 20 else np.nan
        ema50_y = float(ema(df["close"], 50).iloc[-1]) if len(df) >= 50 else np.nan
        pm_above_ema20 = float(pm_bar["close"] > ema20_y) if np.isfinite(ema20_y) else np.nan
        pm_above_ema50 = float(pm_bar["close"] > ema50_y) if np.isfinite(ema50_y) else np.nan
        pm_above_y_high = float(pm_bar["close"] >= y_high) if np.isfinite(y_high) else np.nan
        pm_near_y_high  = float(pm_bar["close"] >= (y_high * 0.995)) if np.isfinite(y_high) else np.nan
        pm_above_pm_vwap= float(pm_bar["close"] >= pm_bar["vwap"]) if np.isfinite(pm_bar["vwap"]) else np.nan

        # --- compact PM power score (0..1) ---
        def _clip01_local(x):
            try: return float(max(0.0, min(1.0, x)))
            except: return 0.0
        def _sigm(x):
            try: return 1.0 / (1.0 + np.exp(-x))
            except: return 0.0

        pm_energy = 0.55 * _sigm((pm_gap_pct - PM_GAP_MIN) / 0.01) \
                  + 0.45 * _sigm((pm_tr_atr - PM_TR_ATR_MIN) / 0.10 if np.isfinite(pm_tr_atr) else -999)

        pm_flow = max(
            _clip01_local((pm_rel_vol_20 / PM_REL_VOL_MIN) if (np.isfinite(pm_rel_vol_20) and PM_REL_VOL_MIN>0) else 0.0),
            _clip01_local((pm_float_turnover / PM_FLOAT_TURN_MIN) if (np.isfinite(pm_float_turnover) and PM_FLOAT_TURN_MIN>0) else 0.0)
        )

        pm_levels = _clip01_local(
            0.50*(pm_near_y_high if np.isfinite(pm_near_y_high) else 0.0) +
            0.25*(pm_above_ema20 if np.isfinite(pm_above_ema20) else 0.0) +
            0.25*(pm_above_ema50 if np.isfinite(pm_above_ema50) else 0.0)
        )

        pm_position = _clip01_local(pm_close_pos) * (1.0 if (pm_above_pm_vwap==1.0) else 0.8)

        pm_power = (PM_W_ENERGY*pm_energy + PM_W_FLOW*pm_flow + PM_W_LEVELS*pm_levels + PM_W_POSITION*pm_position)
        pm_power_score = _clip01_local(pm_power)

        # expose PM features
        pm_feats.update({
        "pm_open": pm_bar["open"],
        "pm_high": pm_bar["high"],
        "pm_low": pm_bar["low"],
        "pm_close": pm_bar["close"],
        "pm_volume": pm_bar["volume"],
        "pm_vwap": pm_bar["vwap"],
        "pm_gap_pct": pm_gap_pct,
        "pm_close_pct_in_range": pm_close_pos,
        "pm_tr_atr": pm_tr_atr,
        "pm_rel_vol_20": pm_rel_vol_20,
        "pm_float_turnover": pm_float_turnover,
        "pm_above_y_high": pm_above_y_high,
        "pm_near_y_high": pm_near_y_high,
        "pm_above_ema20": pm_above_ema20,
        "pm_above_ema50": pm_above_ema50,
        "pm_above_pm_vwap": pm_above_pm_vwap,
        "pm_power_score": pm_power_score,
        })
    else:
        pm_feats.update({k: np.nan for k in [
            "pm_open","pm_high","pm_low","pm_close","pm_volume","pm_vwap","pm_gap_pct",
            "pm_close_pct_in_range","pm_tr_atr","pm_rel_vol_20","pm_float_turnover",
            "pm_above_y_high","pm_near_y_high","pm_above_ema20","pm_above_ema50",
            "pm_above_pm_vwap","pm_power_score"
        ]})

    # add PM features to output row
    for k,v in pm_feats.items():
        # 'return {...}' happens below; just place them into a dict we return
        pass


    return {
        # core event
        "ticker": ticker,
        "event_date": ed.date().isoformat(),
        "cross_date": (evt["cross_date"].date().isoformat() if evt["cross_date"] is not None else None),
        "pending_cross": bool(evt["pending_cross"]),
        "hist_event": hist_value,
        "score": float(score),

        # price/volume basics
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": last_close,
        "volume": int(row["volume"]) if not pd.isna(row["volume"]) else np.nan,
        "rel_vol_20": float(rel_vol_20) if np.isfinite(rel_vol_20) else np.nan,
        "vol_highest_10": float(vol_highest_10) if np.isfinite(vol_highest_10) else np.nan,
        "vol_highest_5": float(vol_highest_5) if np.isfinite(vol_highest_5) else np.nan,
        "vol_rank_10": float(vol_rank_10) if np.isfinite(vol_rank_10) else np.nan,
        "vol_rank_20": float(vol_rank_20) if np.isfinite(vol_rank_20) else np.nan,
        "vol_pctile_20": float(vol_pctile_20) if np.isfinite(vol_pctile_20) else np.nan,

        # green bar diagnostics
        "price_green": float(price_green) if np.isfinite(price_green) else np.nan,
        "vol_green": float(vol_green) if np.isfinite(vol_green) else np.nan,
        "green_price_and_vol": float(green_price_and_vol) if np.isfinite(green_price_and_vol) else np.nan,

        # range/volatility
        "close_pct_in_range": float(close_pct),
        "tr_ratio": float(tr_ratio) if np.isfinite(tr_ratio) else np.nan,
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,

        # trend/location & EMA50 placement
        "price_above_20": price_above_20,
        "price_above_50": price_above_50,
        "ema20_slope_pos": ema20_slope_pos,
        "ema50_pos": ema50_pos if ema50_pos is not None else "",
        "ema50_below_bar": float(ema50_below_bar) if ema50_below_bar==ema50_below_bar else np.nan,
        "ema50_cut_bar": float(ema50_cut_bar) if ema50_cut_bar==ema50_cut_bar else np.nan,
        "dist_close_ema50_atr": float(dist_close_ema50_atr) if np.isfinite(dist_close_ema50_atr) else np.nan,

        # room / R:R
        "room_atr": float(room_atr) if np.isfinite(room_atr) else np.nan,
        "rr": float(rr) if np.isfinite(rr) else np.nan,

        # overhead supply & AVWAP
        "overhead_supply_40": float(overhead_supply_40) if np.isfinite(overhead_supply_40) else np.nan,
        "overhead_supply_20": float(overhead_supply_20) if np.isfinite(overhead_supply_20) else np.nan,
        "price_above_avwap": float(price_above_avwap) if np.isfinite(price_above_avwap) else np.nan,
        "dist_avwap_atr": float(dist_avwap_atr) if np.isfinite(dist_avwap_atr) else np.nan,

        # float / turnover
        "float_shares_millions": float(float_shares_millions) if np.isfinite(float_shares_millions) else np.nan,
        "float_turnover_pct": float(float_turnover_pct) if np.isfinite(float_turnover_pct) else np.nan,
        "float_turnover20_pct": float(float_turnover20_pct) if np.isfinite(float_turnover20_pct) else np.nan,

        # ================= TODAY KPIs (appended) =================
        "rel_vol_20_today": float(rel_vol_20_today) if np.isfinite(rel_vol_20_today) else np.nan,
        "vol_highest_10_today": float(vol_highest_10_today) if np.isfinite(vol_highest_10_today) else np.nan,
        "vol_highest_5_today": float(vol_highest_5_today) if np.isfinite(vol_highest_5_today) else np.nan,
        "vol_pctile_20_today": float(vol_pctile_20_today) if np.isfinite(vol_pctile_20_today) else np.nan,

        "price_green_today": float(price_green_today) if np.isfinite(price_green_today) else np.nan,
        "vol_green_today": float(vol_green_today) if np.isfinite(vol_green_today) else np.nan,
        "green_price_and_vol_today": float(green_price_and_vol_today) if np.isfinite(green_price_and_vol_today) else np.nan,

        "close_pct_in_range_today": float(close_pct_in_range_today) if np.isfinite(close_pct_in_range_today) else np.nan,
        "tr_ratio_today": float(tr_ratio_today) if np.isfinite(tr_ratio_today) else np.nan,
        "atr_pct_today": float(atr_pct_today) if np.isfinite(atr_pct_today) else np.nan,

        "room_atr_today": float(room_atr_today) if np.isfinite(room_atr_today) else np.nan,
        "rr_today": float(rr_today) if np.isfinite(rr_today) else np.nan,

        "ema50_pos_today": ema50_pos_today,
        "ema50_below_bar_today": float(ema50_below_bar_today) if ema50_below_bar_today==ema50_below_bar_today else np.nan,
        "ema50_cut_bar_today": float(ema50_cut_bar_today) if ema50_cut_bar_today==ema50_cut_bar_today else np.nan,
        "dist_close_ema50_atr_today": float(dist_close_ema50_atr_today) if np.isfinite(dist_close_ema50_atr_today) else np.nan,

        "overhead_supply_40_today": float(overhead_supply_40_today) if np.isfinite(overhead_supply_40_today) else np.nan,
        "overhead_supply_20_today": float(overhead_supply_20_today) if np.isfinite(overhead_supply_20_today) else np.nan,
        "price_above_avwap_today": float(price_above_avwap_today) if np.isfinite(price_above_avwap_today) else np.nan,
        "dist_avwap_atr_today": float(dist_avwap_atr_today) if np.isfinite(dist_avwap_atr_today) else np.nan,

        "float_turnover_pct_today": float(float_turnover_pct_today) if np.isfinite(float_turnover_pct_today) else np.nan,
        "float_turnover20_pct_today": float(float_turnover20_pct_today) if np.isfinite(float_turnover20_pct_today) else np.nan,

        "power_score_today": float(power_score_today) if np.isfinite(power_score_today) else np.nan,
        # =========================================================

        # ================= PRE-MARKET KPIs =================
        "pm_open": pm_feats["pm_open"],
        "pm_high": pm_feats["pm_high"],
        "pm_low": pm_feats["pm_low"],
        "pm_close": pm_feats["pm_close"],
        "pm_volume": pm_feats["pm_volume"],
        "pm_vwap": pm_feats["pm_vwap"],
        "pm_gap_pct": pm_feats["pm_gap_pct"],
        "pm_close_pct_in_range_pm": pm_feats["pm_close_pct_in_range"],
        "pm_tr_atr": pm_feats["pm_tr_atr"],
        "pm_rel_vol_20": pm_feats["pm_rel_vol_20"],
        "pm_float_turnover": pm_feats["pm_float_turnover"],
        "pm_above_y_high": pm_feats["pm_above_y_high"],
        "pm_near_y_high": pm_feats["pm_near_y_high"],
        "pm_above_ema20_pm": pm_feats["pm_above_ema20"],
        "pm_above_ema50_pm": pm_feats["pm_above_ema50"],
        "pm_above_pm_vwap": pm_feats["pm_above_pm_vwap"],
        "pm_power_score": pm_feats["pm_power_score"],

        # ------------ Volume Keys ------------------ #
        # Pre-market
        "pm_green_vol": pm_feats["pm_green_vol"],
        "pm_red_vol": pm_feats["pm_red_vol"],
        "pm_green_ratio": pm_feats["pm_green_ratio"],
        "pm_buyer_dominance": pm_feats["pm_buyer_dominance"],

        # RTH full day
        "fd_green_vol": fd_g,
        "fd_red_vol": fd_r,
        "fd_green_ratio": fd_gr,
        "fd_buyer_dominance": fd_dom,
        "fd_vwap": fd_vwap,

        # RTH open
        "open_green_vol": op_g,
        "open_red_vol": op_r,
        "open_green_ratio": op_gr,
        "open_buyer_dominance": op_dom,
        "open_rvol": op_rvol,

        # RTH midday
        "mid_green_vol": mid_g,
        "mid_red_vol": mid_r,
        "mid_green_ratio": mid_gr,
        "mid_buyer_dominance": mid_dom,
        "mid_rvol": mid_rvol,

        # RTH close
        "close_green_vol": cl_g,
        "close_red_vol": cl_r,
        "close_green_ratio": cl_gr,
        "close_buyer_dominance": cl_dom,
        "close_rvol": cl_rvol,

        # context: yesterday low exhaustion flags
        "prev_low_2d": prev_low_2d,
        "prev_low_5d": prev_low_5d,

    }

for sym in ["NEGG", "DLO"]:
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    print(f"[CHECK] {sym} daily file exists in {DATA_DIR}: {os.path.exists(path)}")


# ---------- Main ----------
def main():
    try:
        tickers, float_map = load_tickers_and_floats(TICKERS_CSV)
    except Exception as e:
        log(f"ERROR loading tickers: {e}")
        sys.exit(1)
    log(f"Scanning {len(tickers)} tickers...")

    bench_path = os.path.join(DATA_DIR, f"{BENCH_TICKER}.csv")
    bench_df = load_daily(BENCH_TICKER) if os.path.exists(bench_path) else None

    rows = []
    for i, tk in enumerate(tickers, 1):
        if i % 150 == 0:
            log(f"Progress: {i}/{len(tickers)}")
        try:
            df = load_daily(tk)
            if tk in {"NEGG","DLO"}:
                print(f"[DBG {tk}] len(df)={len(df)}  first={df.index.min()} last={df.index.max()}")
                if len(df) >= 5:
                    mc, sg, hs = macd_hist(df["close"])
                    print(f"[DBG {tk}] hist tail: " + ", ".join(f"{x:.4f}" for x in hs.tail(8).values))
                    pos = (hs > 0)
                    cross_up = pos & (~pos.shift(1, fill_value=False))
                    print(f"[DBG {tk}] cross_up_any={bool(cross_up.any())}  last_cross={str(hs.index[cross_up][-1].date()) if cross_up.any() else 'None'}")
            if df.empty or len(df) < 25:   # need enough bars to warm MACD/ATR
                continue
            feats = features_for_event(df, bench_df, tk, float_map)
            if feats is None:
                continue
            rows.append(feats)
        except Exception as e:
            if DEBUG and len(FAIL_SAMPLES) < DEBUG_SAMPLE:
                FAIL_SAMPLES.append(f"[{tk}] EXCEPTION: {e}")
            continue
        if feats is None and tk in {"NEGG","DLO"}:
            print(f"[DBG {tk}] no event returned by detector (likely no negative run in window or 'green' test failed).")

    # ---------- Debug summary ----------
    #today_naive = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
    today_naive = _now_ny().normalize().tz_localize(None)
    log(f"Today augmentation (RTH only): attempted={TODAY_ATTEMPTED}  appended={TODAY_APPENDED}  empty={TODAY_EMPTY}  beforeRTH={TODAY_BEFORE_RTH}")
    log(f"Events with event_date == today({today_naive.date()}): {EVENT_TODAY}  (pending={EVENT_PENDING})")

    if DEBUG and FAIL_SAMPLES:
        log("Sample failures (today-specific reasons for no event):")
        for line in FAIL_SAMPLES:
            log("  " + line)

    if not rows:
        log("No candidates found with current settings.")
        return

    out = pd.DataFrame(rows).sort_values(["score", "event_date"], ascending=[False, False])

    import numpy as np

    def _clip01(x): 
        return float(np.nanmax([0.0, np.nanmin([1.0, x])])) if np.isfinite(x) else 0.0

    def _sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x))

    def compute_power_score(r):
        # Pillar: Energy
        energy = 0.6 * _sigmoid((r.get("rel_vol_20", np.nan) - 1.8)) \
            + 0.4 * _sigmoid((r.get("tr_ratio", np.nan) - 1.2) / 0.4)

        # Pillar: Room
        room = _clip01((r.get("room_atr", np.nan) or 0.0) / 1.5) \
            + 0.4 * _clip01(1.0 - (r.get("overhead_supply_40", 0.5) or 0.5))  # reward lower overhead
        room = min(1.0, room)

        # Pillar: Support
        ema_bonus = _clip01(max(0.0, r.get("dist_close_ema50_atr", 0.0)) / 1.0)
        avwap_bonus = _clip01(max(0.0, r.get("dist_avwap_atr", 0.0)) / 0.7)
        ema_pen = 0.15 if str(r.get("ema50_pos","")) == "above_bar" else 0.0
        support = 0.5 * ema_bonus + 0.5 * avwap_bonus - ema_pen
        support = _clip01(support)

        # Pillar: Flow/Fuel
        # Map float turnover: 0.5%+ ≈ strong (cap at 1.5%)
        ft = r.get("float_turnover_pct", np.nan)
        flow_turnover = _clip01((ft or 0.0) / 0.015 if np.isfinite(ft) else 0.0)  # 1.5% → 1.0
        flow_vol = _clip01((r.get("vol_pctile_20", 0.0) or 0.0) / 100.0)
        flow = max(flow_turnover, flow_vol)

        # Blend pillars (weights sum to 1)
        power = 0.35*energy + 0.30*room + 0.20*support + 0.15*flow

        # small boosts/penalties
        if r.get("green_price_and_vol", 0) == 1: power += 0.05
        if r.get("pending_cross", False): power -= 0.05  # optional

        return _clip01(power)

    def passes_rules(r):
        reasons = []

        # Must: green price & volume
        if r.get("green_price_and_vol", 0) != 1:
            return False, ["not_green_price_and_volume"]

        # Energy
        cond_energy = (r.get("tr_ratio", 0) >= 1.2) or (r.get("rel_vol_20", 0) >= 1.8) or (r.get("vol_highest_10", 0) == 1)
        if not cond_energy: reasons.append("low_energy")

        # Room
        cond_room = (r.get("room_atr", 0) >= 1.0) or (r.get("overhead_supply_40", 1.0) <= 0.35)
        if not cond_room: reasons.append("low_room")

        # Support
        cond_support = (str(r.get("ema50_pos","")) in {"below_bar","inside_bar"}) and (r.get("price_above_avwap", 0) == 1)
        if not cond_support: reasons.append("weak_support")

        # Flow
        cond_flow = (r.get("float_turnover_pct", 0) >= 0.005) or (r.get("vol_pctile_20", 0) >= 80)  # 0.5% or vol in top 20%
        if not cond_flow: reasons.append("weak_flow")

        ok = (cond_energy + cond_room + cond_support + cond_flow) >= 3  # require 3 of 4 pillars
        if not ok and not reasons:
            reasons.append("fell_below_pillar_threshold")

        return ok, reasons
    
    def pm_passes_rules(r):
        reasons = []

        # Gap (energy)
        if not np.isfinite(r.get("pm_gap_pct", np.nan)) or r["pm_gap_pct"] < PM_GAP_MIN:
            reasons.append("gap_small")

        # Flow (volume/float)
        flow_ok = (
            (np.isfinite(r.get("pm_rel_vol_20", np.nan)) and r["pm_rel_vol_20"] >= PM_REL_VOL_MIN) or
            (np.isfinite(r.get("pm_float_turnover", np.nan)) and r["pm_float_turnover"] >= PM_FLOAT_TURN_MIN)
        )
        if not flow_ok:
            reasons.append("weak_pm_flow")

        # Energy 2: PM range vs ATR
        if not np.isfinite(r.get("pm_tr_atr", np.nan)) or r["pm_tr_atr"] < PM_TR_ATR_MIN:
            reasons.append("pm_range_small_vs_atr")

        # Position & levels
        pos_ok = (np.isfinite(r.get("pm_close_pct_in_range_pm", np.nan)) and r["pm_close_pct_in_range_pm"] >= PM_CLOSE_POS_MIN)
        lvl_ok = (r.get("pm_near_y_high", 0) == 1.0) or (r.get("pm_above_y_high", 0) == 1.0) or (r.get("pm_above_ema20_pm", 0) == 1.0)
        if not pos_ok:
            reasons.append("pm_close_not_strong")
        if not lvl_ok:
            reasons.append("not_near_key_level")

        ok = (len([x for x in ["gap_small","weak_pm_flow","pm_range_small_vs_atr","pm_close_not_strong","not_near_key_level"] if x not in reasons]) >= 3)
        # equivalently: require at least 3 of 5 checks to be good
        return ok, reasons

    # --- apply to your output DF ---
    out["power_score"] = out.apply(compute_power_score, axis=1)

    sel_flags = []
    sel_reasons = []
    for _, r in out.iterrows():
        ok, why = passes_rules(r)
        sel_flags.append(1 if ok else 0)
        sel_reasons.append(",".join(why) if why else "")

    out["selected"] = sel_flags
    out["why_not_selected"] = sel_reasons

    out["pm_selected"] = 0
    out["pm_why_not_selected"] = ""
    out["pm_rank"] = -out["pm_power_score"]  # for sorting (higher power → smaller rank)

    pm_flags = []
    pm_reasons = []
    for _, r in out.iterrows():
        ok_pm, why_pm = pm_passes_rules(r)
        pm_flags.append(1 if ok_pm else 0)
        pm_reasons.append(",".join(why_pm) if why_pm else "")
    out["pm_selected"] = pm_flags
    out["pm_why_not_selected"] = pm_reasons

    # A combined view: prioritize PM first, then your existing day metrics
    view = out.sort_values(
        ["pm_selected","pm_power_score","selected","power_score","score"],
        ascending=[False, False, False, False, False]
    )
    print("Top premarket-selected:", ", ".join(view[view["pm_selected"]==1].head(15)["ticker"].tolist()))

    # For convenience, keep a filtered view (but still save full CSV)
    view = out.sort_values(["selected","power_score","score"], ascending=[False, False, False])
    print("Top selected (power_score):", ", ".join(view[view["selected"]==1].head(15)["ticker"].tolist()))

    out.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(out)} candidates to {OUTPUT_CSV}")
    topn = min(15, len(out))
    log("Top tickers: " + ", ".join(out.head(topn)["ticker"].tolist()))

    SLIM_COLS = [
    "ticker",
    # End-of-day features
    "close",
    "rel_vol_20",
    "close_green_ratio", "close_buyer_dominance", "close_rvol", "room_atr_today", "atr_pct_today",
    "close_pct_in_range", "fd_buyer_dominance", "float_turnover_pct_today",

    # Pre-market features
    "pm_gap_pct",
    "pm_rel_vol_20", "pm_float_turnover", "pm_volume", "pm_tr_atr", "pm_above_y_high", "pm_near_y_high",
    "pm_green_ratio", "pm_buyer_dominance", "prev_low_2d", "prev_low_5d",
    "pm_above_pm_vwap",
    "pm_power_score",
    ]

    slim_cols = [c for c in SLIM_COLS if c in out.columns]
    slim_df = out[slim_cols].copy()

    slim_outfile = OUTPUT_CSV.replace(".csv", "_SLIM.csv")
    slim_df.to_csv(slim_outfile, index=False)
    print(f"✅ Slim CSV written: {slim_outfile}")


if __name__ == "__main__":
    main()