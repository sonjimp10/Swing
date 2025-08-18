# File: Extract_Selected_ByCrossAndPending_to_Excel.py
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "macd_pre_cross_ranked.csv"
OUT_XLSX   = "selected_merged.xlsx"

ROOM_MIN_EVENT   = 1.5   # event-day room_atr for latest event_date
ROOM_MIN_TODAY   = 1.5   # today room_atr_today for crossed lane
LAST_N_CROSS_DATES = 15  # last 15 distinct cross_date values (incl. latest)
# ──────────────────────────────────────────────────────────────────────────────

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii","replace").decode("ascii"), flush=True)

def is_one(series):
    """Robust boolean for 0/1 or True/False columns stored as floats/ints."""
    return series.fillna(0).astype(float) >= 0.5

def to_date_col(s):
    """Parse to datetime.date (not tz-aware); leaves NaT as NaN."""
    return pd.to_datetime(s, errors="coerce").dt.date

def first_existing_col(df, *names):
    """Return the first column name that exists in df from the provided list, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None

def num_series(df, colname):
    """Return a numeric series (float) with NaN for non-numeric/missing."""
    if colname is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[colname], errors="coerce")

def bool_series(df, *candidate_names, default_if_missing=True):
    """
    Return a boolean series from the first existing candidate column (interpreted via is_one).
    If none exists, return all-True (or all-False) per default_if_missing.
    """
    col = first_existing_col(df, *candidate_names)
    if col is None:
        log(f"WARNING: none of {candidate_names} found; not filtering on that condition.")
        return pd.Series(default_if_missing, index=df.index)
    return is_one(df[col])

def main():
    if not os.path.exists(INPUT_CSV):
        log(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    # Normalize dates
    if "event_date" not in df.columns:
        log("ERROR: 'event_date' is missing in input.")
        sys.exit(1)
    df["event_date"] = to_date_col(df["event_date"])
    df["cross_date"] = to_date_col(df["cross_date"]) if "cross_date" in df.columns else pd.NaT

    # Latest event date
    good_dates = [d for d in df["event_date"].dropna().unique()]
    if not good_dates:
        log("ERROR: no valid event_date values found.")
        sys.exit(1)
    latest_event_date = max(good_dates)
    log(f"Latest event_date in file: {latest_event_date}")

    # ── Lane A: Latest EVENT-DATE ────────────────────────────────────────────
    laneA = df[df["event_date"] == latest_event_date].copy()

    room_ok_A   = num_series(laneA, "room_atr") > ROOM_MIN_EVENT
    green_ok_A  = bool_series(laneA, "vol_green", "green_price_and_vol")

    vol_col_A   = first_existing_col(laneA, "vol_pctile_20")
    vol80_A     = num_series(laneA, vol_col_A) >= 60

    close_col_A = first_existing_col(
        laneA, "close", "close_price", "event_close", "close_event", "close_today"
    )
    price_ok_A  = num_series(laneA, close_col_A) < 50

    close_pct_col_A = first_existing_col(laneA, "close_pct_in_range")
    close_pct_ok_A  = num_series(laneA, close_pct_col_A) > 0.25

    tr_col_A    = first_existing_col(laneA, "tr_ratio", "tr_ratio_event")
    tr_ok_A     = num_series(laneA, tr_col_A) > 0.1

    cond_A = room_ok_A & green_ok_A & (vol80_A | close_pct_ok_A) & price_ok_A & tr_ok_A
    laneA  = laneA[cond_A].copy()
    laneA["selected_on"] = "event_latest"

    # ── Lane B: Crossed in LAST 15 distinct cross_date values ────────────────
    valid_cross_mask = df["cross_date"].notna() & (df["cross_date"] <= latest_event_date)
    unique_cross_dates = (
        pd.Series(df.loc[valid_cross_mask, "cross_date"].unique())
        .dropna()
        .sort_values(ascending=False)
        .tolist()
    )
    last15_dates = set(unique_cross_dates[:LAST_N_CROSS_DATES])

    laneB = df[df["cross_date"].isin(last15_dates)].copy()

    room_col_B  = first_existing_col(laneB, "room_atr_today", "room_atr")
    room_ok_B   = num_series(laneB, room_col_B) > ROOM_MIN_TODAY

    green_ok_B  = bool_series(laneB, "vol_green_today", "green_price_and_vol_today")

    vol_col_B   = first_existing_col(laneB, "vol_pctile_20_today", "vol_pctile_20")
    vol80_B     = num_series(laneB, vol_col_B) >= 60

    close_col_B = first_existing_col(laneB, "close_today", "close", "close_price")
    price_ok_B  = num_series(laneB, close_col_B) < 50 

    close_pct_col_B = first_existing_col(laneB, "close_pct_in_range_today", "close_pct_in_range")
    close_pct_ok_B  = num_series(laneB, close_pct_col_B) > 0.25

    tr_col_B    = first_existing_col(laneB, "tr_ratio_today", "tr_ratio")
    tr_ok_B     = num_series(laneB, tr_col_B) > 0.1

    cond_B = room_ok_B & green_ok_B & (vol80_B | close_pct_ok_B) & price_ok_B & tr_ok_B
    laneB  = laneB[cond_B].copy()
    laneB["selected_on"] = "crossed_last15_using_today"

    # ── Merge lanes ──────────────────────────────────────────────────────────
    merged = pd.concat([laneA, laneB], ignore_index=True, sort=False)

    def room_for_sort(row):
        if row.get("selected_on") == "crossed_last15_using_today":
            return pd.to_numeric(row.get("room_atr_today", np.nan), errors="coerce")
        return pd.to_numeric(row.get("room_atr", np.nan), errors="coerce")

    if not merged.empty:
        merged["_room_sort"] = merged.apply(room_for_sort, axis=1)
        merged["_sort_date"] = np.where(
            merged["selected_on"] == "crossed_last15_using_today",
            merged["cross_date"],
            merged["event_date"],
        )

        # Robust vol key construction
        if "vol_pctile_20_today" in merged.columns:
            vol_today = pd.to_numeric(merged["vol_pctile_20_today"], errors="coerce")
        else:
            vol_today = pd.Series(np.nan, index=merged.index)
        if "vol_pctile_20" in merged.columns:
            vol_event = pd.to_numeric(merged["vol_pctile_20"], errors="coerce")
        else:
            vol_event = pd.Series(np.nan, index=merged.index)

        merged["_vol_sort"] = np.where(
            merged["selected_on"] == "crossed_last15_using_today",
            vol_today,
            vol_event,
        )

        merged = merged.sort_values(
            by=["_sort_date", "_room_sort", "_vol_sort"],
            ascending=[False, False, False]
        ).drop(columns=["_sort_date", "_room_sort", "_vol_sort"], errors="ignore")

    # Ensure Ticker column
    if "ticker" in merged.columns and "Ticker" not in merged.columns:
        merged = merged.rename(columns={"ticker": "Ticker"})
    elif "Ticker" not in merged.columns:
        for guess in ["symbol", "Symbol", "SYM"]:
            if guess in merged.columns:
                merged = merged.rename(columns={guess: "Ticker"})
                break

    # Key columns first
    preferred = [
        "Ticker","event_date","cross_date","selected_on",
        # Prices
        "close","close_price","event_close","close_event","close_today",
        # Event-day KPIs
        "room_atr","vol_pctile_20","close_pct_in_range","tr_ratio",
        "vol_green","green_price_and_vol",
        # Today KPIs
        "room_atr_today","vol_pctile_20_today","close_pct_in_range_today","tr_ratio_today",
        "vol_green_today","green_price_and_vol_today",
        # Extras
        "pending_cross","hist_event","power_score","score",
        "dist_close_ema50_atr","dist_close_ema50_atr_today",
    ]
    preferred = [c for c in preferred if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in preferred]
    if not merged.empty:
        merged = merged[preferred + other_cols]

    # Save ONE Excel file (single sheet)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="Merged", index=False)

    log(f"Saved merged selection: {len(merged)} rows -> {OUT_XLSX}")
    if not merged.empty and "Ticker" in merged.columns:
        sample = ", ".join(merged["Ticker"].astype(str).head(25).tolist())
        log("Tickers (first 25): " + sample)

if __name__ == "__main__":
    main()
