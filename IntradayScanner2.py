
# IntradayScanner.py — resistance-gated, anti-fade + N-day breakout + gap filters

import sys, glob
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

# ================= CONFIG =================
DATA_DIR        = Path("ALPACA_DATA")          # where your 5m CSVs live
LARGECAP_FILE   = Path("LargeCap.csv")         # tickers list (column Ticker/Symbol or single column)
DATE_TO_CHECK   = "2025-08-15"                 # session date (US/Eastern)
ASSUME_INPUT_TZ = "Europe/Paris"               # your CSV timezone (handles DST); use "UTC" if needed
OUTPUT_CSV      = f"LargeCap_diagnostics_{DATE_TO_CHECK}.csv"

# Opening range window (for ORH)
OR_MINUTES = 30                                 # first 30 minutes (6×5m bars)

# Momentum detectors + filters
RET_5M_MIN   = 0.02                             # +2% intrabar vs prev close
RET_10M_MIN  = 0.015                            # +1.5% close-to-close over 10m (steady)
RVOL_MIN     = 1.5
RVOL_STEADY  = 1.2
RVOL_MAX     = 8.0                              # ignore blow-offs (often fade)

# Anti-fade (forward retention after signal)
ANTI_FADE_FWD_MIN = 15                          # minutes forward
ANTI_FADE_MIN_RET = 0.003                       # +0.3% forward
ANTI_FADE_MAX_DD  = 0.02                        # ≤2% max drawdown allowed after signal

# -------- NEW: N-day breakout (no RVOL needed) --------
NDH_LOOKBACK_DAYS = 4                           # “last 3–4 days” => use 4
NDH_BREAK_MODE    = "close"                     # "close" (confirmed) or "high" (earlier)

# -------- NEW: Gap / Room filters --------
MAX_GAP_UP             = 0.06                   # 6%: if open > prior close by more than this…
MIN_ROOM_TO_NDH        = 0.02                   # 2%: require at least this room from open to N-day high
REQUIRE_BASE_AFTER_BIG_GAP = True               # if gapped above NDH, wait for base
BASE_OK_AFTER_ET       = "10:00"                # accept signals only after this ET time when above NDH
RVOL_MAX_AFTER_GAP     = 6.0                    # cap RVOL for first acceptance after big gap
# ==========================================

# Avoid Windows console encoding issues
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass


# --------------- Loader ---------------
def _detect_ts_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower() in ("date","timestamp","time","datetime","dt","ts")]
    return cands[0] if cands else df.columns[0]

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    try:
        return pd.to_datetime(s, format="%Y%m%d %H:%M:%S", errors="raise")  # explicit format
    except Exception:
        return pd.to_datetime(s, errors="coerce")                           # fallback parser

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
                need_map[want] = lower[cand]; break
    missing = [k for k in ("Open","High","Low","Close","Volume") if k not in need_map]
    if missing:
        raise ValueError(f"{path.name}: missing OHLCV columns {missing}")

    ts_col = _detect_ts_column(df)
    ts = _parse_datetime_series(df[ts_col])

    df = df.loc[~ts.isna()].copy()
    ts = ts.loc[~ts.isna()]

    # ✅ localize to source tz (your CSV tz)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ASSUME_INPUT_TZ, nonexistent="shift_forward", ambiguous="NaT")

    df["ts"] = ts
    df = df.rename(columns={
        need_map["Open"]:"Open", need_map["High"]:"High",
        need_map["Low"]:"Low",   need_map["Close"]:"Close",
        need_map["Volume"]:"Volume",
    })[["ts","Open","High","Low","Close","Volume"]].sort_values("ts").reset_index(drop=True)

    return df


# --------------- Indicators ---------------
def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_v = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_v

def rvol_chunk(vol: pd.Series, bars: int) -> pd.Series:
    chunk = vol.rolling(bars).sum()
    base  = chunk.rolling(12).mean().shift(1)  # prior hour of similar chunks
    return chunk / base

def forward_ret_and_dd(close: pd.Series, fwd_bars: int) -> pd.DataFrame:
    fwd = (close.shift(-fwd_bars) / close) - 1.0
    dd  = []
    for i in range(len(close)):
        seg = close.iloc[i : i + fwd_bars + 1]
        if len(seg) < 2: dd.append(np.nan); continue
        dd.append(min(0.0, seg.min()/seg.iloc[0] - 1.0))
    return pd.DataFrame({"fwd_ret": fwd, "fwd_dd": dd}, index=close.index)


# -------- Prior High helpers --------
def prior_day_high(d: pd.DataFrame, date_str: str) -> float:
    prev = d[d["date_et"] < date_str].tail(78)
    return prev["High"].max() if not prev.empty else np.nan

def opening_range_high(day: pd.DataFrame, minutes: int) -> float:
    bars = max(1, minutes // 5)
    return day.iloc[:bars]["High"].max() if not day.empty else np.nan

def ndays_prior_high(d: pd.DataFrame, date_str: str, n_days: int) -> float:
    all_dates = d["date_et"].unique().tolist()
    prev_dates = sorted([x for x in all_dates if x < date_str])
    if not prev_dates: return np.nan
    pick = prev_dates[-n_days:] if len(prev_dates) >= n_days else prev_dates
    prev_bars = d[d["date_et"].isin(pick)]
    return prev_bars["High"].max() if not prev_bars.empty else np.nan


# --------------- Detectors ---------------
def analyze_one_day(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    d = df.copy()
    d["ts_et"]   = d["ts"].dt.tz_convert("US/Eastern")
    d["date_et"] = d["ts_et"].dt.strftime("%Y-%m-%d")
    d["time_et"] = d["ts_et"].dt.strftime("%H:%M")

    day = d[d["date_et"] == date_str].copy()
    if day.empty:
        return pd.DataFrame(columns=["detector","gate","ts_et","time_et","price","ret","rvol","gap_up","room_to_ndh"])

    # indicators
    day["VWAP"]  = vwap(day)
    day["ret_5m_high_vs_prev_close"] = (day["High"] / day["Close"].shift(1)) - 1.0
    day["ret_10m_close"]             =  day["Close"] / day["Close"].shift(2) - 1.0
    day["rvol_5m"] = rvol_chunk(day["Volume"], 1)

    # references
    pdh = prior_day_high(d, date_str)
    orh = opening_range_high(day, OR_MINUTES)
    ndh = ndays_prior_high(d, date_str, NDH_LOOKBACK_DAYS)
    prev = d[d["date_et"] < date_str].tail(78)
    pdc  = prev["Close"].iloc[-1] if not prev.empty else np.nan

    # cumulative vs prior close
    day["cum_vs_pdc"] = (day["Close"] / pdc - 1.0) if pd.notna(pdc) else np.nan

    # forward retention
    fwd = forward_ret_and_dd(day["Close"], ANTI_FADE_FWD_MIN//5)
    day["fwd15_ret"], day["fwd15_dd"] = fwd["fwd_ret"], fwd["fwd_dd"]

    # ---- Gap / Room measurements ----
    day_open     = day["Open"].iloc[0]
    gap_up       = (day_open / pdc - 1.0) if pd.notna(pdc) else np.nan
    room_to_ndh  = ((ndh - day_open) / day_open) if pd.notna(ndh) else np.nan

    # helper: accept only after a base time (for big-gap-above-NDH)
    def _after_base(ix) -> bool:
        try:
            hh, mm = map(int, str(day.loc[ix, "time_et"]).split(":"))
            bh, bm = map(int, BASE_OK_AFTER_ET.split(":"))
            return (hh, mm) >= (bh, bm)
        except Exception:
            return True

    def keep(ix) -> bool:
        r = day.loc[ix]
        return (pd.notna(r["fwd15_ret"]) and pd.notna(r["fwd15_dd"])
                and r["fwd15_ret"] >= ANTI_FADE_MIN_RET
                and r["fwd15_dd"]  >= -ANTI_FADE_MAX_DD)

    # unified gap/room gate used by all detectors
    def _gap_room_allows(ix) -> bool:
        # missing refs → don't block
        if pd.isna(gap_up) or pd.isna(room_to_ndh) or pd.isna(ndh):
            return True
        # huge gap with no room → reject
        if gap_up > MAX_GAP_UP and room_to_ndh < MIN_ROOM_TO_NDH:
            return False
        # gapped above NDH → wait for base and avoid blow-off
        if REQUIRE_BASE_AFTER_BIG_GAP and gap_up > MAX_GAP_UP and day_open > ndh:
            if not _after_base(ix):
                return False
            if not (RVOL_MIN <= day.loc[ix, "rvol_5m"] <= RVOL_MAX_AFTER_GAP):
                return False
        return True

    hits: List[Dict] = []

    # 1) PDH breakout (confirmed by close), sane RVOL, 5m spike
    if pd.notna(pdh):
        above_pdh = (day["Close"] > pdh)
        mask_pdh  = (above_pdh
                     & (day["rvol_5m"].between(RVOL_MIN, RVOL_MAX))
                     & (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN))
        if mask_pdh.any():
            idx = mask_pdh.idxmax()
            if bool(mask_pdh.loc[idx]) and keep(idx) and _gap_room_allows(idx):
                r = day.loc[idx]
                hits.append({"detector":"PDH_BREAK_RET_RVOL","gate":"PDH",
                            "ts_et": r["ts_et"], "time_et": r["time_et"],
                            "price": float(r["Close"]),
                            "ret": float(r["ret_5m_high_vs_prev_close"]),
                            "rvol": float(r["rvol_5m"]),
                            "gap_up": float(gap_up) if pd.notna(gap_up) else np.nan,
                            "room_to_ndh": float(room_to_ndh) if pd.notna(room_to_ndh) else np.nan})

    # 2) ORH breakout (confirmed by close), sane RVOL, 5m spike
    if pd.notna(orh):
        above_orh = (day["Close"] > orh)
        mask_orh  = (above_orh
                     & (day["rvol_5m"].between(RVOL_MIN, RVOL_MAX))
                     & (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN))
        if mask_orh.any():
            idx = mask_orh.idxmax()
            if bool(mask_orh.loc[idx]) and keep(idx) and _gap_room_allows(idx):
                r = day.loc[idx]
                hits.append({"detector":"ORH_BREAK_RET_RVOL","gate":"ORH",
                            "ts_et": r["ts_et"], "time_et": r["time_et"],
                            "price": float(r["Close"]),
                            "ret": float(r["ret_5m_high_vs_prev_close"]),
                            "rvol": float(r["rvol_5m"]),
                            "gap_up": float(gap_up) if pd.notna(gap_up) else np.nan,
                            "room_to_ndh": float(room_to_ndh) if pd.notna(room_to_ndh) else np.nan})

    # 3) Steady trend (no blow-off): above VWAP & ORH, 10m push, modest RVOL, up vs PDC
    if pd.notna(orh):
        mask_steady = ((day["Close"] > day["VWAP"]) & (day["Close"] > orh)
                       & (day["ret_10m_close"] >= RET_10M_MIN)
                       & (day["rvol_5m"].between(RVOL_STEADY, 6.0))
                       & (day["cum_vs_pdc"] >= 0.01))
        if mask_steady.any():
            idx = mask_steady.idxmax()
            if bool(mask_steady.loc[idx]) and keep(idx) and _gap_room_allows(idx):
                r = day.loc[idx]
                hits.append({"detector":"STEADY_TREND","gate":"VWAP+ORH",
                            "ts_et": r["ts_et"], "time_et": r["time_et"],
                            "price": float(r["Close"]),
                            "ret": float(r["ret_10m_close"]),
                            "rvol": float(r["rvol_5m"]),
                            "gap_up": float(gap_up) if pd.notna(gap_up) else np.nan,
                            "room_to_ndh": float(room_to_ndh) if pd.notna(room_to_ndh) else np.nan})

    # 4) N-day high breakout (no RVOL requirement, still respects gap rules)
    if pd.notna(ndh):
        if NDH_BREAK_MODE.lower() == "high":
            broke = (day["High"] > ndh) & (day["High"].shift(1) <= ndh)
        else:  # "close"
            broke = (day["Close"] > ndh) & (day["Close"].shift(1) <= ndh)
        if broke.any():
            idx = broke.idxmax()
            if bool(broke.loc[idx]) and _gap_room_allows(idx):
                r = day.loc[idx]
                hits.append({"detector":f"NDH_BREAK_{NDH_LOOKBACK_DAYS}D","gate":"NDH",
                            "ts_et": r["ts_et"], "time_et": r["time_et"],
                            "price": float(r["Close"]),
                            "ret": np.nan,
                            "rvol": float(day.loc[idx, "rvol_5m"]) if pd.notna(day.loc[idx, "rvol_5m"]) else np.nan,
                            "gap_up": float(gap_up) if pd.notna(gap_up) else np.nan,
                            "room_to_ndh": float(room_to_ndh) if pd.notna(room_to_ndh) else np.nan})

    if not hits:
        return pd.DataFrame(columns=["detector","gate","ts_et","time_et","price","ret","rvol","gap_up","room_to_ndh"])

    out = pd.DataFrame(hits).sort_values("ts_et").reset_index(drop=True)
    return out


# --------------- Utils ---------------
def find_csv_for_ticker(tic: str) -> Optional[Path]:
    pats = [
        str(DATA_DIR / f"{tic}.csv"),
        str(DATA_DIR / f"{tic}_5m.csv"),
        str(DATA_DIR / f"{tic}-5m*.csv"),
        str(DATA_DIR / f"{tic}*5min*.csv"),
        str(DATA_DIR / f"{tic}*5*.csv"),
        str(DATA_DIR / tic / "ticker.csv"),
        str(DATA_DIR / tic / f"{tic}.csv"),
    ]
    for p in pats:
        hits = glob.glob(p)
        if hits: return Path(hits[0])
    return None

def load_largecap_tickers(path: Path) -> List[str]:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.lower() in ("ticker","symbol","symbols")]
    if cols:
        col = cols[0]
        return (df[col].dropna().astype(str).str.upper().str.strip().unique().tolist())
    if df.shape[1] == 1:
        return (df.iloc[:,0].dropna().astype(str).str.upper().str.strip().unique().tolist())
    raise ValueError("LargeCap.csv must have a Ticker/Symbol column or a single column of tickers.")


# --------------- Main ---------------
def main():
    print(f"\nDiagnostics for {DATE_TO_CHECK} (US/Eastern)…\n")
    tickers = load_largecap_tickers(LARGECAP_FILE)
    all_rows: List[pd.DataFrame] = []

    for tic in tickers:
        p = find_csv_for_ticker(tic)
        if not p:
            print(f"[WARN] {tic}: file not found in {DATA_DIR}."); continue

        try:
            bars = load_bars_csv(p)
        except Exception as e:
            print(f"[ERR]  {tic}: {e}"); continue

        res = analyze_one_day(bars, DATE_TO_CHECK)
        if res.empty:
            print(f"{tic:<6} no qualified alerts")
            continue

        print(f"{tic}:")
        for det in res["detector"].unique():
            sub = res[res["detector"] == det].sort_values("ts_et")
            r = sub.iloc[0]
            rr = "" if pd.isna(r.get("ret")) else f" ret={r['ret']*100:4.1f}%"
            rv = "" if pd.isna(r.get("rvol")) else f" rvol={r['rvol']:.2f}"
            print(f"  {det:<22} {r['time_et']}  gate={r['gate']:<8}"
                  f"{rr}{rv}  gap={r.get('gap_up',np.nan):+.1%}"
                  f"  roomNDH={r.get('room_to_ndh',np.nan):+.1%}")

        res = res.copy()
        res["ticker"] = tic
        res["ts_utc"] = pd.to_datetime(res["ts_et"]).dt.tz_convert("UTC")
        all_rows.append(res[["ticker","detector","gate","ts_utc","ts_et","time_et",
                             "price","ret","rvol","gap_up","room_to_ndh"]])

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).sort_values(["ticker","ts_utc"])
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results for {out['ticker'].nunique()} tickers to {OUTPUT_CSV}")
    else:
        print("\nNo qualifying alerts. Consider tweaking thresholds or modes.")


if __name__ == "__main__":
    main()
