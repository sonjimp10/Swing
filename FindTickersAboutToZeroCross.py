# File: FindMACD_AlmostCrossers.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------- CONFIG --------------------
DATA_DIR      = "ALPACA_DAILY_DATA"
TICKERS_CSV   = "LargeCap.csv"
OUTPUT_CSV    = "macd_last_neg_green_before_cross.csv"

# MACD params (Yahoo default)
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9

# Only consider crosses that happened within the last N calendar days
LOOKBACK_CAL_DAYS = 62   # ~ 2 months; change to 31 for ~1 month

# Floating-point tolerance
EPS = 1e-10

# Optional: print detailed reasons for these tickers
DEBUG_TICKERS = []  # e.g., ["SHC", "COIN"]
# ------------------------------------------------

def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)

def load_daily(ticker: str) -> pd.DataFrame:
    """Read daily CSV written by your fetcher; return DataFrame indexed by date."""
    fp = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()

    df = pd.read_csv(fp, parse_dates=["Date"])
    # normalize columns
    df.rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"},
              inplace=True)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # limit to recent window if desired
    if LOOKBACK_CAL_DAYS is not None:
        cutoff = (
            pd.Timestamp.now(tz="America/New_York").normalize()
            - pd.Timedelta(days=LOOKBACK_CAL_DAYS)
        ).tz_localize(None)   # drop tz → naive timestamp

        df = df[df.index >= cutoff]

    return df

def compute_macd(close: pd.Series, fast=12, slow=26, sig=9):
    """Return macd, signal, hist Series (same index as close)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def last_negative_green_before_cross(hist: pd.Series):
    """
    Find the most recent index where:
      - hist crosses up through zero at t (hist[t] > 0 and hist[t-1] <= 0)
      - candidate is t-1 (still negative)
      - candidate is 'green' (hist[t-1] > hist[t-2])
      - candidate is the highest (max) value within the contiguous negative run before t
    Returns dict with details, or None.
    """
    if hist.shape[0] < 5:
        return None

    pos = hist > 0
    cross_up = pos & (~pos.shift(1, fill_value=False))  # first positive after non-positive

    if not cross_up.any():
        return None

    # Consider only crosses inside our data window; take the most recent
    cross_idx = hist.index[cross_up].tolist()[-1]
    i = hist.index.get_loc(cross_idx)  # index of the cross day (positive)

    # Candidate is the prior day
    if i - 1 < 1:  # need at least two prior bars (i-1 and i-2)
        return None

    cand_idx = hist.index[i - 1]
    prev_idx = hist.index[i - 2]

    h_cand = float(hist.loc[cand_idx])
    h_prev = float(hist.loc[prev_idx])

    # Must be negative (strict) and "green" vs previous bar
    if not (h_cand < 0 and h_cand > h_prev + EPS):
        return None

    # Confirm it's the highest (closest to zero) of the negative run before the cross
    # Find start of this negative run: last index before i where hist >= 0
    nonneg_before = (hist[:cand_idx] >= 0)
    if nonneg_before.any():
        # last non-negative index before cand
        run_start_pos = nonneg_before[::-1].idxmax()  # last True when reversed
        # idxmax trick needs at least one True; ensure it's really True
        if not nonneg_before.loc[run_start_pos]:
            # fallback: no prior non-negative
            neg_run_start = hist.index[0]
        else:
            # next index after the last non-negative is the start of the negative run
            try:
                run_start_loc = hist.index.get_loc(run_start_pos)
                neg_run_start = hist.index[run_start_loc + 1]
            except Exception:
                neg_run_start = hist.index[0]
    else:
        neg_run_start = hist.index[0]

    neg_run = hist.loc[neg_run_start:cand_idx]
    # From that run, take only negative bars
    neg_run = neg_run[neg_run < 0]
    if neg_run.empty:
        return None

    run_max = float(neg_run.max())  # closest to zero among negatives
    if not (h_cand >= run_max - 1e-9):
        return None

    return {
        "event_date": cand_idx.date().isoformat(),
        "cross_date": cross_idx.date().isoformat(),
        "hist_event": h_cand,
        "hist_prev": h_prev,
        "hist_next": float(hist.loc[cross_idx]),
        "days_to_cross": (cross_idx - cand_idx).days
    }

def load_tickers(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in ["Ticker","ticker","Symbol","symbol","SYM"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            return s[(s!="") & s.notna()].unique().tolist()
    raise ValueError(f"No ticker column in {csv_path}. Columns: {list(df.columns)}")

def evaluate_ticker(tk: str):
    df = load_daily(tk)
    if df.empty or "close" not in df.columns:
        return None

    macd, signal, hist = compute_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIG)
    evt = last_negative_green_before_cross(hist)

    if tk in DEBUG_TICKERS:
        # Minimal debug print
        last_row = df.tail(1).index[0].date().isoformat()
        if evt:
            log(f"[DEBUG {tk}] evt={evt['event_date']} -> cross={evt['cross_date']} hist={evt['hist_event']:.4f}")
        else:
            log(f"[DEBUG {tk}] no event in window; last day {last_row}")

    if evt is None:
        return None

    # enrich with price context on event day
    ed = pd.to_datetime(evt["event_date"])
    row = df.loc[ed] if ed in df.index else None
    price = float(row["close"]) if row is not None else np.nan

    return {
        "ticker": tk,
        "event_date": evt["event_date"],
        "cross_date": evt["cross_date"],
        "days_to_cross": evt["days_to_cross"],
        "hist_event": evt["hist_event"],
        "hist_prev": evt["hist_prev"],
        "hist_next": evt["hist_next"],
        "close_on_event": price
    }

def main():
    try:
        tickers = load_tickers(TICKERS_CSV)
    except Exception as e:
        log(f"ERROR loading tickers: {e}")
        sys.exit(1)

    results = []
    for i, tk in enumerate(tickers, 1):
        if i % 150 == 0:
            log(f"Progress: {i}/{len(tickers)}")
        try:
            r = evaluate_ticker(tk)
            if r:
                results.append(r)
        except Exception:
            continue

    if results:
        out = pd.DataFrame(results).sort_values(
            ["event_date","hist_event"], ascending=[False, False]
        )
        out.to_csv(OUTPUT_CSV, index=False)
        log(f"Found {len(out)} tickers. Saved to {OUTPUT_CSV}")
        log("Tickers: " + ", ".join(out["ticker"].tolist()))
    else:
        log("No matches in the current lookback window.")
        log("Tip: increase LOOKBACK_CAL_DAYS or widen your daily data extraction window.")

if __name__ == "__main__":
    main()
