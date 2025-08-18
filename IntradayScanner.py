
# IntradayScanner.py (fixed)
# - Scans all tickers from LargeCap.csv using 5-minute CSVs in ALPACA_DATA/.
# - Robust CSV loader (Date/time/timestamp/datetime; flexible formats).
# - Signals per 2025-08-15 session (US/Eastern):
#     * RET_RVOL: 5m return >= 2% and 5m RVOL >= 1.5
#     * VWAP_RECLAIM_RVOL1.5
#     * PDH_BREAK_RVOL1.5  (bar CLOSE > prior-day high + RVOL)
# - Prints earliest time per detector and saves full results CSV.

import sys, glob
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
DATA_DIR = Path("ALPACA_DATA")
LARGECAP_FILE = Path("LargeCap.csv")
DATE_TO_CHECK = "2025-08-15"          # US/Eastern session date
ASSUME_INPUT_TZ = "Europe/Berlin"        # or "UTC" if your CSV times are UTC
OUTPUT_CSV = f"LargeCap_diagnostics_{DATE_TO_CHECK}.csv"

# Thresholds
RET_WIN_MIN = 5
RET_THRESH = 0.02          # +2% in 5 minutes
RVOL_THRESH = 1.5          # >= 1.5
PDH_RVOL_THRESH = 1.5
VWAP_RVOL_THRESH = 1.5

# Avoid Windows console encoding issues
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
# ----------------------------------------


# ---------- Loader helpers ----------
def find_csv_for_ticker(tic: str) -> Optional[Path]:
    patterns = [
        str(DATA_DIR / f"{tic}.csv"),
        str(DATA_DIR / f"{tic}_5m.csv"),
        str(DATA_DIR / f"{tic}-5m*.csv"),
        str(DATA_DIR / f"{tic}*5min*.csv"),
        str(DATA_DIR / f"{tic}*5*.csv"),
        str(DATA_DIR / tic / "ticker.csv"),
        str(DATA_DIR / tic / f"{tic}.csv"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return Path(hits[0])
    return None


def _detect_ts_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower() in ("date","timestamp","time","datetime","dt","ts")]
    return cands[0] if cands else df.columns[0]


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """
    Robust datetime parser. Tries explicit 'YYYYMMDD HH:MM:SS' first,
    then falls back to pandas' flexible parser.
    """
    s = s.astype(str).str.strip()
    # explicit '20250815 20:35:00'
    try:
        dt = pd.to_datetime(s, format="%Y%m%d %H:%M:%S", errors="raise")
        return dt
    except Exception:
        pass
    # fallback (ISO, etc.)
    return pd.to_datetime(s, errors="coerce")


def load_bars_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # map OHLCV (case-insensitive)
    lower = {c.lower(): c for c in df.columns}
    need_map: Dict[str, str] = {}
    for want, cands in {
        "Open":  ["open","o"],
        "High":  ["high","h"],
        "Low":   ["low","l"],
        "Close": ["close","c","adjclose","adj_close","price"],
        "Volume":["volume","v","vol"],
    }.items():
        for cand in cands:
            if cand in lower:
                need_map[want] = lower[cand]
                break
    missing = [k for k in ("Open","High","Low","Close","Volume") if k not in need_map]
    if missing:
        raise ValueError(f"{path.name}: missing OHLCV columns {missing}")

    # detect timestamp column (Date/timestamp/time/datetime/etc.)
    ts_col = _detect_ts_column(df)
    ts = _parse_datetime_series(df[ts_col])

    # drop unparsable rows
    df = df.loc[~ts.isna()].copy()
    ts = ts.loc[~ts.isna()]

    # --- LOCALIZE TO SOURCE TIMEZONE (your CSV's timezone) ---
    # If timestamps are naive, localize to ASSUME_INPUT_TZ (e.g., "Europe/Paris").
    # If already tz-aware, keep as-is.
    try:
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(ASSUME_INPUT_TZ, nonexistent="shift_forward", ambiguous="NaT")
    except AttributeError:
        # older pandas: check via .tz on first element
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(ASSUME_INPUT_TZ, nonexistent="shift_forward", ambiguous="NaT")

    df["ts"] = ts

    # standardize columns
    df = df.rename(columns={
        need_map["Open"]: "Open",
        need_map["High"]: "High",
        need_map["Low"]: "Low",
        need_map["Close"]: "Close",
        need_map["Volume"]: "Volume",
    })[["ts","Open","High","Low","Close","Volume"]]

    return df.sort_values("ts").reset_index(drop=True)


# ---------- Indicators / detectors ----------
def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_v = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_v


def rvol_5m(volume: pd.Series, window_bars: int = 1) -> pd.Series:
    chunk = volume.rolling(window_bars).sum()
    base = chunk.rolling(12).mean().shift(1)  # 1 hour of 5m windows
    return chunk / base


def analyze_one_day(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    d = df.copy()
    d["ts_et"] = d["ts"].dt.tz_convert("US/Eastern")
    d["date_et"] = d["ts_et"].dt.strftime("%Y-%m-%d")
    d["time_et"] = d["ts_et"].dt.strftime("%H:%M")

    day = d[d["date_et"] == date_str].copy()
    if day.empty:
        # return an explicitly empty frame with expected columns
        return pd.DataFrame(columns=["detector","ts_et","time_et","rvol","ret","price"])

    # indicators
    day["VWAP"] = vwap(day)
    day["ret_5m"] = day["Close"].pct_change(1)
    day["rvol_5m"] = rvol_5m(day["Volume"], 1)

    # prior-day high (confirmation by close > PDH)
    prev = d[d["date_et"] < date_str].tail(78)  # ~ 1 session of 5m bars
    pdh = prev["High"].max() if not prev.empty else np.nan
    day["PDH_BREAK"] = False if np.isnan(pdh) else (
        (day["Close"] > pdh) & (day["Close"].shift(1) <= pdh)
    )

    # VWAP reclaim confirmation
    day["VWAP_RECLAIM"] = (day["Close"] > day["VWAP"]) & (day["Close"].shift(1) <= day["VWAP"].shift(1))

    hits: List[Dict] = []

    # RET_RVOL
    mask_rr = (day["ret_5m"] >= RET_THRESH) & (day["rvol_5m"] >= RVOL_THRESH)
    if mask_rr.any() and mask_rr.any():
        idx = mask_rr.idxmax()
        if bool(mask_rr.loc[idx]):
            r = day.loc[idx]
            hits.append({
                "detector": "RET_RVOL",
                "ts_et": r["ts_et"], "time_et": r["time_et"],
                "rvol": float(r["rvol_5m"]), "ret": float(r["ret_5m"]),
                "price": float(r["Close"]),
            })

    # VWAP_RECLAIM_RVOL1.5
    mask_vr = day["VWAP_RECLAIM"] & (day["rvol_5m"] >= VWAP_RVOL_THRESH)
    if mask_vr.any():
        idx = mask_vr.idxmax()
        if bool(mask_vr.loc[idx]):
            r = day.loc[idx]
            hits.append({
                "detector": "VWAP_RECLAIM_RVOL1.5",
                "ts_et": r["ts_et"], "time_et": r["time_et"],
                "rvol": float(r["rvol_5m"]), "ret": None,
                "price": float(r["Close"]),
            })

    # PDH_BREAK_RVOL1.5
    mask_pdh = day["PDH_BREAK"] & (day["rvol_5m"] >= PDH_RVOL_THRESH)
    if mask_pdh.any():
        idx = mask_pdh.idxmax()
        if bool(mask_pdh.loc[idx]):
            r = day.loc[idx]
            hits.append({
                "detector": "PDH_BREAK_RVOL1.5",
                "ts_et": r["ts_et"], "time_et": r["time_et"],
                "rvol": float(r["rvol_5m"]), "ret": None,
                "price": float(r["Close"]),
            })

    # ---- safe return even if hits is empty ----
    if not hits:
        return pd.DataFrame(columns=["detector","ts_et","time_et","rvol","ret","price"])

    out = pd.DataFrame(hits)
    # Ensure ts_et exists before sorting (it will, given above, but be defensive)
    if "ts_et" not in out.columns:
        out["ts_et"] = pd.NaT
    out = out.sort_values("ts_et").reset_index(drop=True)
    return out


def load_largecap_tickers(path: Path) -> List[str]:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() in ("ticker","symbol","symbols")]
    if cols:
        col = cols[0]
        return (
            df[col].dropna().astype(str).str.upper().str.strip().unique().tolist()
        )
    if df.shape[1] == 1:
        return (
            df.iloc[:,0].dropna().astype(str).str.upper().str.strip().unique().tolist()
        )
    raise ValueError("LargeCap.csv must have a Ticker/Symbol column or a single column of tickers.")


# ---------- main ----------
def main():
    print(f"\nDiagnostics for {DATE_TO_CHECK} (US/Eastern)…\n")

    if not LARGECAP_FILE.exists():
        raise SystemExit(f"Missing {LARGECAP_FILE}. Put your tickers there.")

    tickers = load_largecap_tickers(LARGECAP_FILE)
    if not tickers:
        raise SystemExit("No tickers found in LargeCap.csv")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    for tic in tickers:
        p = find_csv_for_ticker(tic)
        if not p:
            print(f"[WARN] {tic}: file not found in {DATA_DIR}.")
            continue

        try:
            bars = load_bars_csv(p)
        except Exception as e:
            print(f"[ERR] {tic}: {e}")
            continue

        res = analyze_one_day(bars, DATE_TO_CHECK)
        if res.empty:
            print(f"{tic:<6} no alerts on {DATE_TO_CHECK}")
            continue

        # Print earliest per detector
        print(f"{tic}:")
        for det in res["detector"].unique():
            sub = res[res["detector"] == det].sort_values("ts_et")
            r = sub.iloc[0]
            if det == "RET_RVOL":
                print(f"  {det:<20} {r['time_et']}  ret>={RET_THRESH*100:.0f}%  "
                      f"rvol>={RVOL_THRESH:.1f}  ret={r.get('ret',0)*100:4.1f}%  rvol={r.get('rvol',np.nan):.2f}")
            else:
                print(f"  {det:<20} {r['time_et']}  rvol={r.get('rvol',np.nan):.2f}")

        # keep all rows
        res = res.copy()
        res["ticker"] = tic
        res["ts_utc"] = pd.to_datetime(res["ts_et"]).dt.tz_convert("UTC")
        all_rows.append(res[["ticker","detector","ts_utc","ts_et","time_et","price","ret","rvol"]])

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).sort_values(["ticker","ts_utc"])
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results for {out['ticker'].nunique()} tickers to {OUTPUT_CSV}")
    else:
        print("\nNo qualifying alerts across the list (try lower thresholds or verify files).")


if __name__ == "__main__":
    main()
