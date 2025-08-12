# File: Rank_MACD_PreCross_1to2Day.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import time as dtime

# ===================== CONFIG =====================
DATA_DIR      = "ALPACA_DAILY_DATA"
TICKERS_CSV   = "LargeCap.csv"
BENCH_TICKER  = "SPY"                 # for relative strength; if SPY.csv not present, RS features are skipped
OUTPUT_CSV    = "macd_pre_cross_ranked.csv"

# Lookback window (limit work to recent data; keeps MACD warmed and fast)
LOOKBACK_CAL_DAYS = 75                # ~2.5 months calendar, fine for MACD(12,26,9)

# MACD params
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

# Signal detection
ALLOW_PENDING_TODAY = True            # include today's bar if it's the best negative green bar but cross not yet happened

# Optional relaxed checks for pending-today evaluation (start strict)
NEAR_ZERO_EPS   = 0.0     # e.g., 0.0005 treats tiny positive as "still negative"
RUN_MAX_ABS_TOL = 0.0     # e.g., 0.0005 absolute wiggle room to be "highest of run"
RUN_MAX_REL_TOL = 0.0     # e.g., 0.05 (=5%) relative wiggle vs run_max

# Hard filters (liquidity / price)
MIN_PRICE        = 5.0
MIN_ADTV_SHARES  = 1_000_000          # 20D avg shares

# Short-term quality thresholds / ranges (used in scoring & optional soft checks)
VOL_SURGE_MIN    = 1.5                # event-day volume / 20D avg
CLOSE_UPPER_QTL  = 0.70               # close in top X of day's range; 0.70 ~ top 30%
TR_EXPAND_MIN    = 1.20               # event-day TrueRange / ATR(20)
ATR_PCT_RANGE    = (0.02, 0.07)       # sweet spot ATR% (2%-7%) for 1-2 day bursts
ROOM_TO_RUN_ATR  = 1.0                # room to 20D high should be >= 1 x ATR(20)
RR_MIN           = 1.5                # simple reward:risk using event low/EMA20 as stop

# Build a provisional "today" daily candle from 5-min RTH data ONLY (no premarket)
INCLUDE_TODAY_FROM_INTRADAY = True
INTRADAY_DIR = "ALPACA_DATA"
TZ_EU = "Europe/Amsterdam"        # how your minute CSVs were saved
TZ_NY = "America/New_York"

# Weights for ranking score (tune to taste; higher = more influence)
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
    "hist_closeness": 0.5,  # closer to 0 (less negative) gets a small bump
    "pending_penalty": -0.2 # slight penalty if cross is still pending (today)
}

# Debug controls
DEBUG = True
DEBUG_TICKERS = []          # e.g. ["PHM","COIN"]
DEBUG_SAMPLE  = 12          # print up to N sample failure reasons
# ===================================================

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii","replace").decode("ascii"), flush=True)

def load_tickers(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in ["Ticker","ticker","Symbol","symbol","SYM"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            return s[(s!="") & s.notna()].unique().tolist()
    raise ValueError(f"No ticker column in {csv_path}. Columns: {list(df.columns)}")

# ---------- diagnostics ----------
TODAY_ATTEMPTED = 0
TODAY_APPENDED  = 0
TODAY_EMPTY     = 0
TODAY_BEFORE_RTH= 0
EVENT_TODAY     = 0
EVENT_PENDING   = 0
FAIL_SAMPLES    = []

def _load_intraday_today_ohlcv(ticker: str):
    """
    Aggregate today's 5-min RTH bars (09:30–now ET) into a provisional daily OHLCV.
    Returns dict or {"before_rth": True} if run before RTH; None if nothing.
    """
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

    now_ny   = pd.Timestamp.now(tz=TZ_NY)
    today_ny = now_ny.date()

    if now_ny.time() < dtime(9,30):
        return {"before_rth": True}

    day = dfm[dfm.index.date == today_ny]
    # RTH only and up to now
    day = day.between_time(dtime(9,30), now_ny.time())

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
    """Read daily CSV and optionally augment with today's intraday RTH bar."""
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
        cutoff = (pd.Timestamp.now(tz=TZ_NY).normalize() - pd.Timedelta(days=LOOKBACK_CAL_DAYS)).tz_localize(None)
        df = df[df.index >= cutoff]

    return df

def ema(s, span): 
    return s.ewm(span=span, adjust=False).mean()

def macd_hist(close):
    fast = ema(close, MACD_FAST)
    slow = ema(close, MACD_SLOW)
    macd = fast - slow
    signal = ema(macd, MACD_SIG)
    return macd, signal, macd - signal

def true_range(df: pd.DataFrame):
    prev_close = df["close"].shift(1)
    x1 = (df["high"] - df["low"]).abs()
    x2 = (df["high"] - prev_close).abs()
    x3 = (df["low"] - prev_close).abs()
    return pd.concat([x1, x2, x3], axis=1).max(axis=1)

def atr(df, n=20):
    return true_range(df).rolling(n).mean()

def close_in_range_pct(row):
    rng = row["high"] - row["low"]
    if rng <= 0:
        return 1.0
    return (row["close"] - row["low"]) / rng

def rs_slope_pos(price: pd.Series, bench: pd.Series, lookback=15):
    # Simple RS momentum: RS_t / RS_{t-lookback} - 1 > 0
    aligned = price.align(bench, join="inner")
    if aligned[0].empty or len(aligned[0]) < lookback + 1:
        return np.nan
    rs = aligned[0] / aligned[1]
    try:
        v = (rs.iloc[-1] / rs.iloc[-1 - lookback]) - 1.0
        return float(v > 0)
    except Exception:
        return np.nan

# ---------- Event detection (fixed & enhanced) ----------
def _is_highest_of_run(hist: pd.Series, cand):
    """Return (ok, h_c, run_max). ok True if cand is highest (closest to zero) in its negative run."""
    h_c = float(hist.loc[cand])

    # find start of negative run: bar after the last non-negative before cand
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
    if neg_run.empty:
        return (False, h_c, np.nan)

    run_max = float(neg_run.max())
    tol = max(RUN_MAX_ABS_TOL, RUN_MAX_REL_TOL * abs(run_max))
    return (h_c >= run_max - tol, h_c, run_max)

def last_neg_green_event(hist: pd.Series, allow_pending=True):
    """
    Pick the most recent valid event:
      1) If today's hist <= NEAR_ZERO_EPS, prefer a PENDING-TODAY candidate (negative, green, highest-of-run).
      2) Else, use the most recent UP cross and test t-1 (negative, green, highest-of-run).
    """
    if len(hist) < 5:
        return None

    today_idx = hist.index[-1]
    h_last = float(hist.iloc[-1])

    # 1) Pending today path (prefer today if still negative / near zero and rising)
    if allow_pending and (h_last <= NEAR_ZERO_EPS):
        if len(hist) >= 2:
            prev_idx = hist.index[-2]
            h_prev = float(hist.loc[prev_idx])
            neg_and_green = (h_last < NEAR_ZERO_EPS) and (h_last > h_prev)
            if neg_and_green:
                ok, h_c, run_max = _is_highest_of_run(hist, today_idx)
                if ok:
                    return {
                        "event_date": today_idx,
                        "cross_date": None,
                        "hist_event": h_c,
                        "pending_cross": True,
                    }
        # fall through to cross-based evaluation if not ok

    # 2) Most recent UP cross path
    positive = hist > 0
    cross_up = positive & (~positive.shift(1, fill_value=False))
    if not cross_up.any():
        return None

    cross_idx = hist.index[cross_up][-1]
    i = hist.index.get_loc(cross_idx)
    if i - 1 < 1:
        return None
    cand = hist.index[i - 1]
    prev = hist.index[i - 2]
    h_c, h_p = float(hist.loc[cand]), float(hist.loc[prev])

    # negative & green
    if not (h_c < 0 and h_c > h_p):
        return None

    ok, h_c, run_max = _is_highest_of_run(hist, cand)
    if not ok:
        return None

    return {
        "event_date": cand,
        "cross_date": cross_idx,
        "hist_event": h_c,
        "pending_cross": False,   # FIX: real cross → not pending
    }

# ---------- Features & scoring ----------
def features_for_event(df: pd.DataFrame, bench_df: pd.DataFrame | None, ticker: str):
    global EVENT_TODAY, EVENT_PENDING, FAIL_SAMPLES

    if df.empty or "close" not in df.columns:
        return None

    macd, sig, hist = macd_hist(df["close"])
    evt = last_neg_green_event(hist, allow_pending=ALLOW_PENDING_TODAY)
    if not evt:
        if DEBUG and len(FAIL_SAMPLES) < DEBUG_SAMPLE:
            # Show why last few failed TODAY specifically (if today exists)
            today_idx = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
            in_idx = today_idx in df.index
            if in_idx and len(hist) >= 2:
                h0 = float(hist.iloc[-1]); h1 = float(hist.iloc[-2])
                ok_run, h_c_tmp, run_max_tmp = _is_highest_of_run(hist, hist.index[-1])
                reasons = []
                if not (h0 <= NEAR_ZERO_EPS): reasons.append("today_not_negative")
                if not (h0 > h1): reasons.append("not_green")
                if not ok_run: reasons.append(f"not_run_max(delta={run_max_tmp - h0:.6f})")
                if not reasons: reasons = ["other"]
                FAIL_SAMPLES.append(
                    f"[{ticker}] today check: h[-2]={h1:.5f}, h[-1]={h0:.5f}, run_max={run_max_tmp:.5f} | reasons={','.join(reasons)}"
                )
            else:
                FAIL_SAMPLES.append(f"[{ticker}] no-event | today_in_index={in_idx} | hist_tail={', '.join(f'{x:.5f}' for x in hist.tail(3).values)}")
        return None

    ed = evt["event_date"]
    if ed not in df.index:
        return None

    # mark if event is today / pending
    today_naive = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
    if ed == today_naive:
        EVENT_TODAY += 1
        if evt["pending_cross"]:
            EVENT_PENDING += 1

    if ticker in DEBUG_TICKERS:
        log(f"[DEBUG {ticker}] EVENT={ed.date()}  CROSS={'PENDING' if evt['pending_cross'] else evt['cross_date'].date() if evt['cross_date'] is not None else 'None'} "
            f"| hist_event={evt['hist_event']:.5f} | hist_tail={', '.join(f'{x:.5f}' for x in hist.tail(4).values)}")

    row = df.loc[ed]
    # EMAs
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)

    # Liquidity
    v20 = df["volume"].rolling(20).mean().loc[ed] if len(df) >= 20 and ed in df.index else np.nan
    adtv_ok = float(v20 >= MIN_ADTV_SHARES) if np.isfinite(v20) else 0.0
    price_ok = float(row["close"] >= MIN_PRICE)
    if not (price_ok and adtv_ok):
        return None

    # Volume surge
    vol_surge = float(row["volume"] / v20) if (np.isfinite(v20) and v20 > 0) else np.nan

    # Close position in range
    close_pct = close_in_range_pct({"high": row["high"], "low": row["low"], "close": row["close"]})

    # Range expansion
    atr20 = atr(df, 20).loc[ed] if len(df) >= 21 else np.nan
    today_tr = true_range(df).loc[ed] if ed in df.index else np.nan
    tr_ratio = float(today_tr / atr20) if (np.isfinite(today_tr) and np.isfinite(atr20) and atr20 > 0) else np.nan

    # ATR percent
    atr_pct = float(atr20 / row["close"]) if (np.isfinite(atr20) and row["close"] > 0) else np.nan
    atr_ok = float(np.isfinite(atr_pct) and ATR_PCT_RANGE[0] <= atr_pct <= ATR_PCT_RANGE[1])

    # Relative strength vs bench
    if bench_df is not None and not bench_df.empty:
        rs_pos = rs_slope_pos(df["close"], bench_df["close"], lookback=15)
    else:
        rs_pos = np.nan

    # Room to run to recent high
    high20 = df["high"].rolling(20).max().shift(1).loc[ed] if ed in df.index else np.nan
    room_atr = float((high20 - row["close"]) / atr20) if (np.isfinite(high20) and np.isfinite(atr20) and atr20 > 0) else np.nan
    room_ok = float(np.isfinite(room_atr) and room_atr >= ROOM_TO_RUN_ATR)

    # Simple R/R: target 20D high, stop = min(event low, EMA20)
    stop = min(float(row["low"]), float(ema20.loc[ed])) if ed in ema20.index else float(row["low"])
    risk = row["close"] - stop
    reward = (high20 - row["close"]) if np.isfinite(high20) else np.nan
    rr = float(reward / risk) if (np.isfinite(reward) and risk > 0 and reward > 0) else np.nan
    rr_ok = float(np.isfinite(rr) and rr >= RR_MIN)

    # Trend / location
    price_above_20 = float(row["close"] > ema20.loc[ed]) if ed in ema20.index else 0.0
    price_above_50 = float(row["close"] > ema50.loc[ed]) if ed in ema50.index else 0.0
    ema20_slope_pos = float(ema20.loc[ed] > ema20.shift(5).loc[ed]) if ed in ema20.index and ed in ema20.shift(5).index else 0.0

    # Hist closeness to zero (less negative is better)
    hist_value = float(evt["hist_event"])
    hist_closeness = float(-hist_value)  # more negative → smaller; we will squash later

    # Score (combine features) — unchanged
    score = (
        W["price_above_20"] * price_above_20 +
        W["price_above_50"] * price_above_50 +
        W["ema20_slope_pos"] * ema20_slope_pos +
        W["vol_surge"] * (1.0 if (np.isfinite(vol_surge) and vol_surge >= VOL_SURGE_MIN) else 0.0) +
        W["close_high_pct"] * (1.0 if close_pct >= CLOSE_UPPER_QTL else 0.0) +
        W["tr_expand"] * (1.0 if (np.isfinite(tr_ratio) and tr_ratio >= TR_EXPAND_MIN) else 0.0) +
        W["atr_sweet"] * (1.0 if (np.isfinite(atr_ok) and atr_ok == 1.0) else 0.0) +
        W["rs_slope_pos"] * (rs_pos if np.isfinite(rs_pos) else 0.0) +
        W["room_run"] * (1.0 if (np.isfinite(room_ok) and room_ok == 1.0) else 0.0) +
        W["rr_ok"] * (1.0 if (np.isfinite(rr_ok) and rr_ok == 1.0) else 0.0) +
        W["hist_closeness"] * np.tanh(max(0.0, hist_closeness) / 0.05) +  # normalize closeness
        (W["pending_penalty"] if evt["pending_cross"] else 0.0)
    )

    return {
        "event_date": ed.date().isoformat(),
        "cross_date": (evt["cross_date"].date().isoformat() if evt["cross_date"] is not None else None),
        "pending_cross": bool(evt["pending_cross"]),
        "hist_event": hist_value,
        "score": float(score),
        # context
        "close": float(row["close"]),
        "volume": int(row["volume"]) if not pd.isna(row["volume"]) else np.nan,
        "vol_surge": float(vol_surge) if np.isfinite(vol_surge) else np.nan,
        "close_pct_in_range": float(close_pct),
        "tr_ratio": float(tr_ratio) if np.isfinite(tr_ratio) else np.nan,
        "atr_pct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
        "room_atr": float(room_atr) if np.isfinite(room_atr) else np.nan,
        "rr": float(rr) if np.isfinite(rr) else np.nan,
        "price_above_20": price_above_20,
        "price_above_50": price_above_50,
        "ema20_slope_pos": ema20_slope_pos,
        "rs_slope_pos": float(rs_pos) if np.isfinite(rs_pos) else np.nan,
    }

# ---------- Main ----------
def main():
    # Load tickers
    try:
        tickers = load_tickers(TICKERS_CSV)
    except Exception as e:
        log(f"ERROR loading tickers: {e}")
        sys.exit(1)
    log(f"Scanning {len(tickers)} tickers...")

    # Bench for RS
    bench_path = os.path.join(DATA_DIR, f"{BENCH_TICKER}.csv")
    bench_df = load_daily(BENCH_TICKER) if os.path.exists(bench_path) else None

    rows = []
    scanned = 0
    for i, tk in enumerate(tickers, 1):
        if i % 150 == 0:
            log(f"Progress: {i}/{len(tickers)}")
        try:
            df = load_daily(tk)
            scanned += 1
            if df.empty or len(df) < 25:  # need enough bars to warm MACD/ATR
                continue

            feats = features_for_event(df, bench_df, tk)
            if feats is None:
                continue

            rows.append({"ticker": tk, **feats})
        except Exception as e:
            if DEBUG and len(FAIL_SAMPLES) < DEBUG_SAMPLE:
                FAIL_SAMPLES.append(f"[{tk}] EXCEPTION: {e}")
            continue

    # ---------- Debug summary ----------
    today_naive = pd.Timestamp.now(tz=TZ_NY).normalize().tz_localize(None)
    log(f"Today augmentation (RTH only): attempted={TODAY_ATTEMPTED}  appended={TODAY_APPENDED}  empty={TODAY_EMPTY}  beforeRTH={TODAY_BEFORE_RTH}")
    log(f"Events with event_date == today({today_naive.date()}): {EVENT_TODAY}  (pending={EVENT_PENDING})")

    if DEBUG and FAIL_SAMPLES:
        log("Sample failures (today-specific reasons for no event):")
        for line in FAIL_SAMPLES:
            log("  " + line)

    if not rows:
        log("No candidates found with current settings.")
        return

    out = pd.DataFrame(rows)
    # Sort by score (desc) then by event_date (desc)
    out = out.sort_values(["score", "event_date"], ascending=[False, False])
    out.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved {len(out)} candidates to {OUTPUT_CSV}")
    # Print quick list
    topn = min(15, len(out))
    log("Top tickers: " + ", ".join(out.head(topn)["ticker"].tolist()))

if __name__ == "__main__":
    main()
