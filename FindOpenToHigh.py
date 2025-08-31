#!/usr/bin/env python3
import os
import pandas as pd
from datetime import datetime, time as dtime
import pytz

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR      = "/Users/jimutmukhopadhyay/Dummy Trading/Intraday Trading/ALPACA_DATA"
TICKERS_CSV   = "/Users/jimutmukhopadhyay/Dummy Trading/Intraday Trading/LargeCap.csv"
TZ_NY         = pytz.timezone("America/New_York")
MAX_GAPS      = 7          # up to 7 overnight gaps
GAP_THRESHOLD = 0.0001       # 1% gap

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def load_minute_data(ticker: str) -> pd.DataFrame:
    fp = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(fp, parse_dates=["Date"])
    df.rename(columns={
        "Date":"dt","Open":"open","High":"high",
        "Low":"low","Close":"close","Volume":"volume"
    }, inplace=True)
    df["dt"] = (
        df["dt"]
          .dt.tz_localize("Europe/Berlin", ambiguous="infer")
          .dt.tz_convert(TZ_NY)
    )
    df.set_index("dt", inplace=True)
    return df.between_time(dtime(9,30), dtime(16,0))

def compute_daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df.resample("D").agg({
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last"
    })
    return ohlc.dropna(subset=["open","close"])

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
def main():
    tickers = pd.read_csv(TICKERS_CSV, dtype=str)["Ticker"]\
                .dropna().str.upper().str.strip().unique()
    out = []

    for T in tickers:
        try:
            df = load_minute_data(T)
        except FileNotFoundError:
            continue

        # show what date-range you actually have
        start, end = df.index.min().date(), df.index.max().date()
        print(f"[{T}] data from {start} to {end}")

        daily = compute_daily_ohlc(df)
        dates = daily.index.normalize().unique()
        if len(dates) < 2:
            print("   → not enough days, skipping\n")
            continue

        # only keep the last MAX_GAPS+1 trading days (or all if fewer)
        trimmed = dates[-(MAX_GAPS+1):]

        # for each overnight gap
        for i in range(1, len(trimmed)):
            today_dt = trimmed[i]
            prev_dt  = trimmed[i-1]
            o = daily.at[today_dt, "open"]
            c = daily.at[today_dt,  "high"]
            gap = (c - o)/o

            if gap >= GAP_THRESHOLD:
                out.append({
                    "RunDate":   today_dt.date().isoformat(),
                    "Ticker":    T,
                    "Open":      o,
                    "High": c,
                    "Gap%":      gap * 100
                })

        print("")

    df_out = pd.DataFrame(out)
    if df_out.empty:
        print(f"No tickers met ≥{GAP_THRESHOLD*100:.1f}% gap in the last {MAX_GAPS} gaps.")
    else:
        df_out.sort_values(["RunDate","Gap%"], ascending=[False,False], inplace=True)
        df_out.to_csv("daily_High_gap_candidates_LC.csv", index=False)
        print(f"→ wrote {len(df_out)} records to daily_High_gap_candidates_LC.csv\n")
        print(df_out)

if __name__ == "__main__":
    main()
