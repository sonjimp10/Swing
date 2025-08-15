# File: Extract_Selected_Tickers.py
import pandas as pd
import numpy as np
import sys
import os

# -------- CONFIG --------
INPUT_CSV  = "macd_pre_cross_ranked.csv"

# thresholds (as requested)
VOL_PCTL_MIN       = 60.0
ROOM_ATR_MIN       = 1.5
DIST_EMA50_MIN     = -0.2     # strictly greater than -0.2
REQUIRE_GREEN_PV   = True     # green_price_and_vol == 1

# output
OUT_MERGED         = "selected_merged.csv"

# optional: also write the intermediate sets (set to True if you want them)
WRITE_INTERMEDIATE = False
OUT_LATEST_EVENT   = "selected_latest_event.csv"
OUT_PREV_TODAY     = "selected_prev_today.csv"
# ------------------------

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii","replace").decode("ascii"), flush=True)

def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

def is_one(series):
    # robust "true" check for float-encoded flags (1.0/0.0) or booleans
    return (series.fillna(0).astype(float) >= 0.5)

def main():
    if not os.path.exists(INPUT_CSV):
        log(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    if "event_date" not in df.columns:
        log("ERROR: 'event_date' column not found in input.")
        sys.exit(1)

    # parse event_date to date
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    if df["event_date"].isna().all():
        log("ERROR: could not parse any event_date values.")
        sys.exit(1)

    # find latest and previous distinct event_date
    unique_dates = sorted([d for d in df["event_date"].unique() if pd.notna(d)])
    if not unique_dates:
        log("ERROR: no valid event dates found.")
        sys.exit(1)

    latest_date = unique_dates[-1]
    prev_date = unique_dates[-2] if len(unique_dates) >= 2 else None

    log(f"Latest event date: {latest_date}")
    if prev_date:
        log(f"Previous event date: {prev_date}")
    else:
        log("Previous event date: [not available]")

    # ---------- Filter set 1: latest date using EVENT-day KPIs ----------
    req_cols_event = [
        "ticker", "event_date",
        "green_price_and_vol", "vol_pctile_20", "room_atr", "dist_close_ema50_atr"
    ]
    require_cols(df, req_cols_event)

    latest_mask = (df["event_date"] == latest_date)
    latest = df.loc[latest_mask].copy()

    conds = []
    if REQUIRE_GREEN_PV:
        conds.append(is_one(latest["green_price_and_vol"]))
    conds.append(pd.to_numeric(latest["vol_pctile_20"], errors="coerce") > VOL_PCTL_MIN)
    conds.append(pd.to_numeric(latest["room_atr"], errors="coerce") > ROOM_ATR_MIN)
    #conds.append(pd.to_numeric(latest["dist_close_ema50_atr"], errors="coerce") > DIST_EMA50_MIN)

    filt_latest = latest[np.logical_and.reduce(conds)].copy()
    filt_latest["selected_on"] = "latest_event"  # mark source

    # ---------- Filter set 2: previous date using TODAY KPIs ----------
    if prev_date is not None:
        req_cols_today = [
            "ticker", "event_date",
            "green_price_and_vol_today", "vol_pctile_20_today",
            "room_atr_today", "dist_close_ema50_atr_today"
        ]
        missing_today = [c for c in req_cols_today if c not in df.columns]
        if missing_today:
            log(f"WARNING: Missing today columns: {missing_today}. Skipping previous-date-today filter.")
            filt_prev_today = pd.DataFrame(columns=df.columns)
        else:
            prev_mask = (df["event_date"] == prev_date)
            prev = df.loc[prev_mask].copy()

            conds2 = []
            if REQUIRE_GREEN_PV:
                conds2.append(is_one(prev["green_price_and_vol_today"]))
            conds2.append(pd.to_numeric(prev["vol_pctile_20_today"], errors="coerce") > VOL_PCTL_MIN)
            conds2.append(pd.to_numeric(prev["room_atr_today"], errors="coerce") > ROOM_ATR_MIN)
            #conds2.append(pd.to_numeric(prev["dist_close_ema50_atr_today"], errors="coerce") > DIST_EMA50_MIN)

            filt_prev_today = prev[np.logical_and.reduce(conds2)].copy()
            filt_prev_today["selected_on"] = "prev_event_using_today_kpis"
    else:
        filt_prev_today = pd.DataFrame(columns=df.columns)

    # ---------- Optional: write intermediate sets ----------
    if WRITE_INTERMEDIATE:
        filt_latest.to_csv(OUT_LATEST_EVENT, index=False)
        log(f"Saved latest-date selection: {len(filt_latest)} rows -> {OUT_LATEST_EVENT}")
        if prev_date is not None and not filt_prev_today.empty:
            filt_prev_today.to_csv(OUT_PREV_TODAY, index=False)
            log(f"Saved prev-date (using _today KPIs) selection: {len(filt_prev_today)} rows -> {OUT_PREV_TODAY}")

    # ---------- Merge to one dataset ----------
    merged = pd.concat([filt_latest, filt_prev_today], ignore_index=True, sort=False)

    # Nice ordering of columns: key fields first, then the rest
    preferred = [
        "ticker","event_date","selected_on","pending_cross","hist_event",
        "power_score","score",
        # event-day KPIs
        "green_price_and_vol","vol_pctile_20","room_atr","dist_close_ema50_atr",
        # today KPIs
        "green_price_and_vol_today","vol_pctile_20_today","room_atr_today","dist_close_ema50_atr_today"
    ]
    preferred = [c for c in preferred if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in preferred]
    merged = merged[preferred + other_cols]

    # Sort: newest first, then by power_score/score
    sort_cols = [c for c in ["event_date","power_score","score"] if c in merged.columns]
    if sort_cols:
        asc = [False] * len(sort_cols)
        merged = merged.sort_values(sort_cols, ascending=asc)

    # Save single output
    merged.to_csv(OUT_MERGED, index=False)
    log(f"Saved merged selection: {len(merged)} rows -> {OUT_MERGED}")

    # Print quick tickers
    if not merged.empty:
        tickers_str = ", ".join(merged["ticker"].astype(str).head(25).tolist())
        log("Merged tickers (first 25): " + tickers_str)

if __name__ == "__main__":
    main()
