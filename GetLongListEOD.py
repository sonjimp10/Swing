# File: Select_Latest_ByRoomThenVolume.py
import pandas as pd
import numpy as np
import os
import sys

INPUT_CSV = "macd_pre_cross_ranked.csv"
OUT_CSV   = "selected_latest_prioritized.csv"

# thresholds
VOL_PCTL_MIN   = 60.0
ROOM_ATR_MIN   = 1.5
DIST_EMA50_MIN = 0.0   # must be > 0 (above EMA50)
REQUIRE_GREEN  = True  # green_price_and_vol == 1

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
    # robust boolean (for 1/0 or True/False)
    return series.fillna(0).astype(float) >= 0.5

def main():
    if not os.path.exists(INPUT_CSV):
        log(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    require_cols(df, ["event_date", "ticker",
                      "green_price_and_vol", "dist_close_ema50_atr",
                      "vol_pctile_20", "room_atr"])

    # Parse event_date to date
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    if df["event_date"].isna().all():
        log("ERROR: could not parse any event_date.")
        sys.exit(1)

    # Latest event_date
    unique_dates = sorted([d for d in df["event_date"].unique() if pd.notna(d)])
    if not unique_dates:
        log("ERROR: no valid event dates found.")
        sys.exit(1)
    latest_date = unique_dates[-1]
    log(f"Latest event date: {latest_date}")

    latest = df.loc[df["event_date"] == latest_date].copy()

    # Coerce numerics safely
    latest["vol_pctile_20"]      = pd.to_numeric(latest["vol_pctile_20"], errors="coerce")
    latest["room_atr"]           = pd.to_numeric(latest["room_atr"], errors="coerce")
    latest["dist_close_ema50_atr"] = pd.to_numeric(latest["dist_close_ema50_atr"], errors="coerce")

    # Filters (strictly greater)
    conds = []
    if REQUIRE_GREEN:
        conds.append(is_one(latest["green_price_and_vol"]))
    conds.append(latest["dist_close_ema50_atr"] > DIST_EMA50_MIN)
    conds.append(latest["vol_pctile_20"] > VOL_PCTL_MIN)
    conds.append(latest["room_atr"] > ROOM_ATR_MIN)

    mask = np.logical_and.reduce(conds)
    sel = latest.loc[mask].copy()

    if sel.empty:
        log("No rows passed the filters for the latest date.")
        # still write an empty CSV with headers for convenience
        sel.to_csv(OUT_CSV, index=False)
        log(f"Wrote empty selection -> {OUT_CSV}")
        return

    # Sort: 1) room_atr desc, 2) vol_pctile_20 desc
    sort_cols = ["room_atr", "vol_pctile_20"]
    sel = sel.sort_values(sort_cols, ascending=[False, False])

    # Nice column ordering (keep everything, just front-load key fields)
    preferred = [c for c in [
        "ticker", "event_date",
        "room_atr", "vol_pctile_20",
        "dist_close_ema50_atr", "green_price_and_vol",
        "power_score", "score",
        "rel_vol_20", "tr_ratio", "overhead_supply_40", "price_above_avwap"
    ] if c in sel.columns]
    other = [c for c in sel.columns if c not in preferred]
    sel = sel[preferred + other]

    sel.to_csv(OUT_CSV, index=False)
    log(f"Saved {len(sel)} rows -> {OUT_CSV}")

    # Quick view
    tickers_preview = ", ".join(sel["ticker"].astype(str).head(25).tolist())
    log("Top (by room_atr then vol_pctile_20): " + tickers_preview)

if __name__ == "__main__":
    main()
