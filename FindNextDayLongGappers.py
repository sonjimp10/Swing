### find_tickers_ema50_vwap.py

import os
import sys
import pandas as pd
import numpy as np
from datetime import time as dtime, datetime
from zoneinfo import ZoneInfo

### ====== CONFIG ======
DATA_DIR   = "/Users/jimutmukhopadhyay/Dummy Trading/Intraday Trading/ALPACA_DATA"
TICKERS_CSV = "/Users/jimutmukhopadhyay/Dummy Trading/Intraday Trading/LargeCap.csv"
TARGET_DATE = "2025-08-07"

TZ_NY = ZoneInfo("America/New_York")
TZ_EU = ZoneInfo("Europe/Berlin")

### Thresholds used elsewhere (kept for cond2)
VWAP_NEAR_FRAC    = 0.003     ### 0.3% closeness for the older "near VWAP" fraction in cond2
GREEN_MAJ_FRAC    = 0.55
LOWER_VWAP_TOUCH_FRAC = 0.003 ### not required now, kept for optional use

### Decision timing
CUT_OFF_TIME     = dtime(15,50)   ### only use bars up to this time
PREF_LATE_START  = dtime(15,25)   ### preferred start for "late" improvement slice
FALLBACK_LAST_N  = 6              ### if preferred slice too short, use last N bars up to 15:50

### Single proximity + improvement rule (no strict/relaxed split)
UPPER_VWAP_MAX_FRAC = 0.018       ### must be within 1.8% of Upper VWAP at finish (last 3 bars mean)
GAP_IMPROVE_MIN     = 0.002       ### must reduce gap to Upper VWAP by >= 0.2% over the late slice

### Sorting preference: small opening gap bucket
OPEN_GAP_PREFERRED_MIN = 0.00     ### 0%
OPEN_GAP_PREFERRED_MAX = 0.01     ### 1%

### Toggles
DEBUG = True                      ### set False to silence logs
REQUIRE_LOWER_VWAP_TOUCH = False  ### optional; not used in cond3 now
ENFORCE_SMALL_GAP_FILTER = False  ### optional extra filter; off by default

### Optional gap filter params (used only if ENFORCE_SMALL_GAP_FILTER=True)
GAP_MIN_PCT = 0.00
GAP_MAX_PCT = 0.05
CLOSE_NEAR_OPEN_FRAC = 0.01
CLOSE_ABOVE_PREV_MAX_PCT = 0.02

### Close strength & blow-off guards
CLOSE_STRENGTH_FRAC = 0.98      ### 98% threshold vs (max(day open, max open/close up to 15:30))
DAY_HIGH_TO_CLOSE_MAX = 1.03    ### day high must be <= 1.03 * day closing price


### ----------------------
### EXACT USER-SPEC VWAP
### ----------------------
def calculate_vwap(df, window=20):
    ### df columns expected: Datetime, High, Low, Close, Volume
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Date'] = df['Datetime'].dt.date

    df['VWAP'] = 0.0
    df['Upper_VWAP'] = 0.0
    df['Lower_VWAP'] = 0.0

    outs = []
    for d, g in df.groupby('Date'):
        g = g.copy()
        g['Cum_TPV'] = (g['Typical'] * g['Volume']).cumsum()
        g['Cum_Vol'] = g['Volume'].cumsum()
        g['VWAP']    = g['Cum_TPV'] / g['Cum_Vol']
        std = g['Typical'].std()
        k = 2
        g['Upper_VWAP'] = g['VWAP'] + k*std
        g['Lower_VWAP'] = g['VWAP'] - k*std
        outs.append(g)

    r = pd.concat(outs).reset_index(drop=True)
    return r.drop(columns=['Cum_TPV','Cum_Vol','Typical','Date'])


def load_minute_data(ticker: str) -> pd.DataFrame:
    fp = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(fp, parse_dates=["Date"])
    df.rename(columns={"Date":"dt","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
    df["dt"] = df["dt"].dt.tz_localize(TZ_EU, ambiguous="infer").dt.tz_convert(TZ_NY)
    df.set_index("dt", inplace=True)
    return df.between_time(dtime(9,30), dtime(16,0))


def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["day"] = out.index.tz_convert(TZ_NY).date

    ### EMA50 exactly as you requested
    tmp = out.copy()
    tmp["Close"] = tmp["close"]
    out["EMA50"] = tmp["Close"].ewm(span=50, adjust=False).mean()

    ### VWAP + bands exactly as you requested
    src = pd.DataFrame({
        "Datetime": out.index.tz_convert(TZ_NY),
        "High": out["high"].values,
        "Low": out["low"].values,
        "Close": out["close"].values,
        "Volume": out["volume"].values,
    })
    vdf = calculate_vwap(src)
    out["VWAP"]        = vdf["VWAP"].values
    out["Upper_VWAP"]  = vdf["Upper_VWAP"].values
    out["Lower_VWAP"]  = vdf["Lower_VWAP"].values

    return out


def session_for_date(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    mask = df.index.tz_convert(TZ_NY).date == target
    return df[mask]


def get_prev_session_close(full_df: pd.DataFrame, date_str: str):
    target = datetime.strptime(date_str, "%Y-%m-%d").date()
    dts = full_df.index.tz_convert(TZ_NY).date
    days = sorted(pd.unique(dts))
    prev = [d for d in days if d < target]
    if not prev:
        return None
    p = prev[-1]
    prev_df = full_df[dts == p]
    if prev_df.empty:
        return None
    return float(prev_df["close"].iloc[-1])


### ------------- helper: dynamic persistent body-above-EMA50 -------------
def first_persistent_body_above_ema50_after_2pm(day_df: pd.DataFrame):
    if day_df.empty:
        return None
    idx_ny = day_df.index.tz_convert(TZ_NY)
    after2 = day_df[idx_ny.time >= dtime(14,0)]
    if after2.empty:
        return None
    body_above = (after2["open"] > after2["EMA50"]) & (after2["close"] > after2["EMA50"])
    starts = body_above & (~body_above.shift(1, fill_value=False))
    for t in after2.index[starts]:
        if body_above.loc[t:].all():
            return t
    return None


### ------------------------- conditions (unchanged ones) -------------------------
def condition_1(day_df: pd.DataFrame):
    t_cross = first_persistent_body_above_ema50_after_2pm(day_df)
    if t_cross is None:
        return None
    after_230 = t_cross.timetz().hour > 14 or (t_cross.timetz().hour == 14 and t_cross.timetz().minute >= 30)
    return {"cross_time": t_cross, "after_230pm": bool(after_230)}


def condition_2(day_df: pd.DataFrame):
    t_cross = first_persistent_body_above_ema50_after_2pm(day_df)
    if t_cross is None:
        return None
    seg = day_df.loc[t_cross:]
    if seg.empty:
        return None
    green_frac = (seg["close"] > seg["open"]).mean()
    near_vwap_frac = (np.abs(seg["close"] - seg["VWAP"]) / seg["close"] <= VWAP_NEAR_FRAC).mean()
    if green_frac >= GREEN_MAJ_FRAC and near_vwap_frac >= 0.50:
        return {"cross_time": t_cross, "green_frac": float(green_frac), "near_vwap_frac": float(near_vwap_frac)}
    return None


### ------------------------- MAIN condition (simplified) -------------------------
def condition_3(day_df: pd.DataFrame, prev_close: float | None = None, ticker: str = ""):
    ### Keep if:
    ###  - After >=14:00, there exists a time from which all candle bodies (open & close) stay above EMA50
    ###  - Up to 15:50, the gap to Upper VWAP shrinks by >= GAP_IMPROVE_MIN over the late slice
    ###  - The last 3 bars’ mean (<= 15:50) is within UPPER_VWAP_MAX_FRAC of Upper VWAP, and above VWAP
    ###  - Close-strength: (last day close OR avg of last 3 <= 15:50) >= CLOSE_STRENGTH_FRAC * max(day open, max(open/close) up to 15:30)
    ###  - No blow-off: day high <= DAY_HIGH_TO_CLOSE_MAX * day close

    def log(msg: str):
        if DEBUG:
            print(f"[{ticker}] {msg}")

    if day_df.empty:
        log("cond3 FAIL @DATA: empty")
        return None

    ### 1) dynamic persistent body-above-EMA50 from >= 14:00
    idx_ny = day_df.index.tz_convert(TZ_NY)
    after2 = day_df[idx_ny.time >= dtime(14, 0)]
    if after2.empty:
        log("cond3 FAIL @AFTER14: no bars >= 14:00")
        return None

    body_above = (after2["open"] > after2["EMA50"]) & (after2["close"] > after2["EMA50"])
    starts = body_above & (~body_above.shift(1, fill_value=False))
    cross_time = None
    for t in after2.index[starts]:
        if body_above.loc[t:].all():
            cross_time = t
            break
    if cross_time is None:
        log("cond3 FAIL @CROSS: never achieved persistent body-above-EMA50")
        return None
    log(f"cond3 CROSS time={cross_time.tz_convert(TZ_NY).timetz()}")

    ### 2) restrict to <= 15:50
    up_to_cut = day_df[idx_ny.time <= CUT_OFF_TIME]
    if up_to_cut.empty:
        log("cond3 FAIL @CUTOFF: no bars <= 15:50")
        return None

    ### 3) choose late slice for improvement: preferred 15:25–15:50, else last N bars
    preferred = up_to_cut[up_to_cut.index.tz_convert(TZ_NY).time >= PREF_LATE_START]
    late = preferred if len(preferred) >= 3 else up_to_cut.tail(FALLBACK_LAST_N)
    if len(late) < 3:
        log("cond3 FAIL @LATE: insufficient bars")
        return None
    log(f"cond3 LATE bars={len(late)} (preferred={len(preferred)})")

    ### 4) improvement of gap-to-upper over late slice
    eps = 1e-9
    gaps = (late["Upper_VWAP"].values - late["close"].values)
    gaps = np.maximum(gaps, 0.0) / np.maximum(late["Upper_VWAP"].values, eps)
    first_gap = float(gaps[0])
    last_gap  = float(gaps[-1])
    gap_improve = first_gap - last_gap

    ### slope (for info/sorting)
    x = np.arange(len(late))
    late_slope = float(np.polyfit(x, late["close"].values, 1)[0])
    log(f"cond3 IMPROVE: first_gap={first_gap:.4f} last_gap={last_gap:.4f} improve={gap_improve:.4f} slope={late_slope:.6f}")

    ### 5) finish proximity using last 3 bars up to 15:50
    last3 = up_to_cut.tail(3)
    mean_close = float(last3["close"].mean())
    mean_upper = float(last3["Upper_VWAP"].mean())
    mean_vwap  = float(last3["VWAP"].mean())

    cut_gap = abs(mean_close - mean_upper) / max(mean_upper, eps)
    near_upper = (cut_gap <= UPPER_VWAP_MAX_FRAC) and (mean_close >= mean_vwap)
    log(f"cond3 NEAR UPPER: cut_gap={cut_gap:.4f} thr={UPPER_VWAP_MAX_FRAC} above_vwap={mean_close >= mean_vwap}")

    ### 6) close-strength vs (max(day open, max body pre-15:30))
    day_open = float(day_df["open"].iloc[0])
    pre1530_mask = idx_ny.time <= dtime(15,30)
    pre1530 = day_df[pre1530_mask] if pre1530_mask.any() else day_df
    max_body_pre1530 = max(float(pre1530["open"].max()), float(pre1530["close"].max()))
    base_ref = max(day_open, max_body_pre1530)

    last_day_close = float(day_df["close"].iloc[-1])                  ### session close
    avg_last3_cut  = float(up_to_cut["close"].tail(3).mean())         ### last-3 up to 15:50

    close_strength = (last_day_close >= CLOSE_STRENGTH_FRAC * base_ref) or \
                     (avg_last3_cut  >= CLOSE_STRENGTH_FRAC * base_ref)
    log(f"cond3 CLOSE STRENGTH: base_ref={base_ref:.4f} last_close={last_day_close:.4f} "
        f"avg_last3_cut={avg_last3_cut:.4f} thr={CLOSE_STRENGTH_FRAC} -> {close_strength}")

    ### 7) blow-off top guard
    day_high = float(day_df["high"].max())
    blowoff_ok = day_high <= DAY_HIGH_TO_CLOSE_MAX * last_day_close
    log(f"cond3 BLOWOFF: day_high={day_high:.4f} close={last_day_close:.4f} "
        f"limit={DAY_HIGH_TO_CLOSE_MAX*last_day_close:.4f} -> {blowoff_ok}")

    ### 8) simple proximity diagnostics you asked for (also saved to CSV)
    close_vs_open_pct = (last_day_close / day_open - 1.0) * 100.0
    close_vs_max1530_pct = (last_day_close / max_body_pre1530 - 1.0) * 100.0
    log(f"cond3 PROX: close_vs_open={close_vs_open_pct:+.2f}% | close_vs_max1530={close_vs_max1530_pct:+.2f}%")

    ### 9) accept/deny
    if not (near_upper and (gap_improve >= GAP_IMPROVE_MIN) and close_strength and blowoff_ok):
        log("cond3 FAIL (composite rule)")
        return None

    ### 10) opening gap vs previous close (for sorting)
    open_gap = None
    if prev_close and prev_close > 0:
        open_gap = (day_open / prev_close) - 1.0

    log("cond3 PASS")
    return {
        "cross_time": cross_time,
        "gap_to_upper": float(cut_gap),
        "gap_improve": float(gap_improve),
        "late_slope": float(late_slope),
        "open_gap": float(open_gap) if open_gap is not None else None,
        "close_vs_open_pct": float(close_vs_open_pct),
        "close_vs_max1530_pct": float(close_vs_max1530_pct),
    }



def condition_4(day_df: pd.DataFrame):
    if (day_df["close"] > day_df["EMA50"]).all():
        return {"always_above": True}
    return None


### ------------------------- utilities & driver -------------------------
def get_tickers_from_csv(path: str):
    df = pd.read_csv(path)
    if df.empty:
        return []
    candidates = ["Symbol","SYMBOL","symbol","Ticker","ticker","Tickers","tickers","Security","security"]
    for c in candidates:
        if c in df.columns:
            s = df[c].dropna().astype(str).str.strip()
            return [x for x in s.tolist() if x]
    s = df[df.columns[0]].dropna().astype(str).str.strip()
    return [x for x in s.tolist() if x]


def analyze_ticker(ticker: str, date_str: str) -> dict:
    try:
        df = load_minute_data(ticker)
    except FileNotFoundError:
        return {"ticker": ticker, "error": "data_not_found"}
    if df.empty:
        return {"ticker": ticker, "error": "empty_data"}

    df = compute_intraday_indicators(df)
    day = session_for_date(df, date_str)
    if day.empty:
        return {"ticker": ticker, "error": f"no_data_for_{date_str}"}

    prev_close = get_prev_session_close(df, date_str)

    res = {"ticker": ticker}
    c1 = condition_1(day)
    c2 = condition_2(day)
    c3 = condition_3(day, prev_close=prev_close, ticker=ticker)
    c4 = condition_4(day)
    if c1: res["cond1"] = c1
    if c2: res["cond2"] = c2
    if c3: res["cond3"] = c3
    if c4: res["cond4"] = c4
    return res

### --- ATR(7) helper: build daily OHLC from minutes and compute 7-day ATR up to TARGET_DATE ---
def compute_atr7_from_minutes(ticker: str, target_date: str) -> float | None:
    """
    Returns the 7-day ATR for `ticker` as of `target_date` (YYYY-MM-DD),
    computed from daily bars aggregated out of your minute data.
    """
    try:
        dfm = load_minute_data(ticker)  ### uses your timezone conversions
    except FileNotFoundError:
        return None
    if dfm.empty:
        return None

    # Get NY trading dates on the minute index
    idx_ny = dfm.index.tz_convert(TZ_NY)
    dfm = dfm.copy()
    dfm["date_ny"] = idx_ny.date

    # Aggregate to daily OHLC (close = last close of the day)
    daily = (
        dfm.groupby("date_ny")
           .agg(high=("high","max"), low=("low","min"), close=("close","last"))
           .sort_index()
    )

    if daily.empty:
        return None

    # True Range: max( high-low, |high - prev_close|, |low - prev_close| )
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"]  - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    # ATR(7): simple rolling mean over last 7 trading days
    atr7 = tr.rolling(window=7, min_periods=2).mean()

    # Return ATR as of target_date (if present)
    try:
        tgt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    if tgt in atr7.index:
        val = atr7.loc[tgt]
    else:
        # If target date is missing (holiday/empty), use the last available prior date
        prior = atr7.loc[atr7.index < tgt]
        if prior.empty:
            return None
        val = prior.iloc[-1]

    return float(val) if pd.notna(val) else None


def main():
    ### Allow CLI override of TARGET_DATE
    date_str = sys.argv[1] if len(sys.argv) > 1 else TARGET_DATE

    ### Load tickers
    tickers = get_tickers_from_csv(TICKERS_CSV)
    if not tickers:
        print(f"No tickers found in {TICKERS_CSV}. Columns present: {pd.read_csv(TICKERS_CSV).columns.tolist()}")
        return

    ### Load FloatShares (robust to column names)
    lc = pd.read_csv(TICKERS_CSV)
    tick_col = "Ticker" if "Ticker" in lc.columns else ("Symbol" if "Symbol" in lc.columns else lc.columns[0])
    lc[tick_col] = lc[tick_col].astype(str).str.strip()
    float_col = "FloatShares" if "FloatShares" in lc.columns else None
    if float_col:
        lc = lc[[tick_col, float_col]].drop_duplicates(subset=[tick_col])
        float_map = lc.set_index(tick_col)[float_col]
    else:
        float_map = None

    ### Analyze all tickers
    results = [analyze_ticker(t, date_str) for t in tickers]

    ### Build condition summaries and metrics lookup for cond3
    conds = {f"cond{i}":[] for i in range(1,5)}
    metrics_by_ticker = {}  ### only for those that PASS cond3

    for r in results:
        for k in conds.keys():
            if k in r:
                conds[k].append(r["ticker"])
        if "cond3" in r:
            c3 = r["cond3"]
            metrics_by_ticker[r["ticker"]] = {
                "gap_to_upper": c3.get("gap_to_upper", float("inf")),
                "gap_improve":  c3.get("gap_improve", 0.0),
                "late_slope":   c3.get("late_slope", 0.0),
                "open_gap":     c3.get("open_gap", None),
                "close_vs_open_pct": c3.get("close_vs_open_pct", None),
                "close_vs_max1530_pct": c3.get("close_vs_max1530_pct", None),
            }

    ### Sort cond3:
    ###   1) prefer 0–1% opening gap
    ###   2) closest to Upper VWAP at ~15:50
    ###   3) larger improvement toward Upper
    ###   4) stronger positive slope
    ###   5) ticker
    def band_rank(og):
        return 0 if (og is not None and (OPEN_GAP_PREFERRED_MIN <= og <= OPEN_GAP_PREFERRED_MAX)) else 1

    cond3_sorted = sorted(
        metrics_by_ticker.keys(),
        key=lambda t: (
            band_rank(metrics_by_ticker[t]["open_gap"]),
            metrics_by_ticker[t]["gap_to_upper"],
            -metrics_by_ticker[t]["gap_improve"],
            -metrics_by_ticker[t]["late_slope"],
            t
        )
    )
    conds["cond3"] = cond3_sorted[:]

    ### Save detailed results (all tickers)
    rows = []
    for r in results:
        row = {"ticker": r["ticker"]}
        for k in ("cond1","cond2","cond3","cond4","error"):
            row[k] = str(r.get(k, ""))
        rows.append(row)
    out_df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    detailed_path = os.path.join("outputs", f"ema50_vwap_scan_{date_str}.csv")
    out_df.to_csv(detailed_path, index=False)

    ### Save sorted cond3 with metrics + FloatShares + ATR7
    cond3_rows = []
    for t in cond3_sorted:
        m = metrics_by_ticker[t]
        ### ATR(7) from minute data aggregated to daily
        atr7 = compute_atr7_from_minutes(t, date_str)

        cond3_rows.append({
            "Ticker": t,
            "FloatShares": (float_map.get(t) if float_map is not None and t in float_map.index else None),
            "Open_Gap": m["open_gap"],
            "Gap_to_UpperVWAP": m["gap_to_upper"],
            "Gap_Improve": m["gap_improve"],
            "Late_Slope": m["late_slope"],
            "Close_vs_Open_Pct": m["close_vs_open_pct"],
            "Close_vs_Max1530_Pct": m["close_vs_max1530_pct"],
            "ATR7": atr7,
        })

    cond3_df = pd.DataFrame(cond3_rows)
    cond3_path = os.path.join("outputs", f"cond3_sorted_tickers_{date_str}.csv")
    cond3_df.to_csv(cond3_path, index=False)

    ### Console summary
    print(f"Scanning date: {date_str}")
    for i in range(1,5):
        print(f"=== Condition {i} ===")
        print(conds[f"cond{i}"])
    print(f"Detailed results saved to: {detailed_path}")
    print(f"Saved cond3-sorted list to: {cond3_path}")

if __name__ == "__main__":
    main()
