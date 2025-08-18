# IntradayScanner.py – resistance-gated, anti-fade diagnostics

import sys, glob
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

# ------------- CONFIG -------------
DATA_DIR = Path("ALPACA_DATA")
LARGECAP_FILE = Path("LargeCap.csv")
DATE_TO_CHECK = "2025-08-15"            # US/Eastern session date
ASSUME_INPUT_TZ = "Europe/Paris"        # your CSV timezone (handles DST)
OUTPUT_CSV = f"LargeCap_diagnostics_{DATE_TO_CHECK}.csv"

# resistance windows
OR_MINUTES = 30                         # opening range length (minutes)

# detectors & filters
RET_5M_MIN   = 0.02                     # +2% in 5 min
RET_10M_MIN  = 0.015                    # +1.5% in 10 min (steady)
RVOL_MIN     = 1.5                      # require participation
RVOL_STEADY  = 1.2
RVOL_MAX     = 8.0                      # ignore blow-offs
ANTI_FADE_FWD_MIN = 15                  # forward retention window
ANTI_FADE_MIN_RET = 0.003               # +0.3% forward
ANTI_FADE_MAX_DD  = 0.02                # ≤ 2% drawdown allowed

# console encoding
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
# ----------------------------------


# ---------- CSV loader ----------
def _detect_ts_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower() in ("date","timestamp","time","datetime","dt","ts")]
    return cands[0] if cands else df.columns[0]

def _parse_datetime_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    try:
        return pd.to_datetime(s, format="%Y%m%d %H:%M:%S", errors="raise")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def load_bars_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # map OHLCV (case-insensitively)
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
    df = df.loc[~ts.isna()].copy(); ts = ts.loc[~ts.isna()]

    # ✅ localize to source tz (your CSV tz)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ASSUME_INPUT_TZ, nonexistent="shift_forward", ambiguous="NaT")

    df["ts"] = ts
    df = df.rename(columns={
        need_map["Open"]: "Open", need_map["High"]: "High",
        need_map["Low"]: "Low",  need_map["Close"]: "Close",
        need_map["Volume"]: "Volume",
    })[["ts","Open","High","Low","Close","Volume"]]

    return df.sort_values("ts").reset_index(drop=True)


# ---------- indicators ----------
def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_v = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_v

def rvol_chunk(vol: pd.Series, bars: int) -> pd.Series:
    chunk = vol.rolling(bars).sum()
    base  = chunk.rolling(12).mean().shift(1)           # prior hour of similar chunks
    return chunk / base

def forward_ret_and_dd(close: pd.Series, fwd_bars: int) -> pd.DataFrame:
    """Forward % return and worst drawdown over next fwd_bars (from start close)."""
    fwd = (close.shift(-fwd_bars) / close) - 1.0
    dd  = []
    for i in range(len(close)):
        seg = close.iloc[i : i + fwd_bars + 1]
        if len(seg) < 2: dd.append(np.nan); continue
        dd.append(min(0.0, seg.min()/seg.iloc[0] - 1.0))
    return pd.DataFrame({"fwd_ret": fwd, "fwd_dd": dd}, index=close.index)


# ---------- detectors ----------
def analyze_one_day(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    d = df.copy()
    d["ts_et"] = d["ts"].dt.tz_convert("US/Eastern")
    d["date_et"] = d["ts_et"].dt.strftime("%Y-%m-%d")
    d["time_et"] = d["ts_et"].dt.strftime("%H:%M")

    day = d[d["date_et"] == date_str].copy()
    if day.empty:
        return pd.DataFrame(columns=["detector","ts_et","time_et","price","ret","rvol","gate"])

    # base indicators
    day["VWAP"] = vwap(day)
    day["ret_5m_close"] = day["Close"].pct_change(1)
    day["ret_5m_high_vs_prev_close"] = (day["High"] / day["Close"].shift(1)) - 1.0
    day["ret_10m_close"] = day["Close"] / day["Close"].shift(2) - 1.0
    day["rvol_5m"] = rvol_chunk(day["Volume"], 1)
    day["ma20"] = day["Close"].rolling(20).mean()

    # prior-day stats
    prev = d[d["date_et"] < date_str].tail(78)
    pdh = prev["High"].max() if not prev.empty else np.nan
    pdc = prev["Close"].iloc[-1] if not prev.empty else np.nan

    # opening-range high (first 30 min = 6 bars)
    day_first = day.iloc[:max(1, OR_MINUTES//5)]
    orh = day_first["High"].max()

    # gates
    day["cum_vs_pdc"] = (day["Close"] / pdc - 1.0) if pd.notna(pdc) else np.nan
    day["above_pdh"]  = (day["Close"] > pdh) if pd.notna(pdh) else False
    day["above_orh"]  = (day["Close"] > orh) if pd.notna(orh) else False
    day["above_vwap"] = (day["Close"] > day["VWAP"])

    # forward retention (anti-fade)
    fwd = forward_ret_and_dd(day["Close"], ANTI_FADE_FWD_MIN//5)
    day["fwd15_ret"] = fwd["fwd_ret"]
    day["fwd15_dd"]  = fwd["fwd_dd"]

    hits: List[Dict] = []

    def keep(ix) -> bool:
        r = day.loc[ix]
        return (pd.notna(r["fwd15_ret"]) and pd.notna(r["fwd15_dd"])
                and r["fwd15_ret"] >= ANTI_FADE_MIN_RET
                and r["fwd15_dd"]  >= -ANTI_FADE_MAX_DD)

    # --- 1) Breakout over PDH (confirmation: close > PDH), sane RVOL, spike in 5m ---
    mask_pdh = (day["above_pdh"]
                & (day["rvol_5m"].between(RVOL_MIN, RVOL_MAX))
                & (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN))
    if mask_pdh.any():
        idx = mask_pdh.idxmax()
        if bool(mask_pdh.loc[idx]) and keep(idx):
            r = day.loc[idx]
            hits.append({"detector":"PDH_BREAK_RET_RVOL",
                        "ts_et": r["ts_et"], "time_et": r["time_et"],
                        "price": float(r["Close"]),
                        "ret": float(r["ret_5m_high_vs_prev_close"]),
                        "rvol": float(r["rvol_5m"]),
                        "gate":"PDH"})

    # --- 2) Breakout over ORH (confirmation: close > ORH), sane RVOL, spike in 5m ---
    mask_orh = (day["above_orh"]
                & (day["rvol_5m"].between(RVOL_MIN, RVOL_MAX))
                & (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN))
    if mask_orh.any():
        idx = mask_orh.idxmax()
        if bool(mask_orh.loc[idx]) and keep(idx):
            r = day.loc[idx]
            hits.append({"detector":"ORH_BREAK_RET_RVOL",
                        "ts_et": r["ts_et"], "time_et": r["time_et"],
                        "price": float(r["Close"]),
                        "ret": float(r["ret_5m_high_vs_prev_close"]),
                        "rvol": float(r["rvol_5m"]),
                        "gate":"ORH"})

    # --- 3) Steady trenders (no blow-off): above VWAP & ORH, 10m push, modest RVOL ---
    mask_steady = (day["above_vwap"] & (day["Close"] > orh)
                   & (day["ret_10m_close"] >= RET_10M_MIN)
                   & (day["rvol_5m"].between(RVOL_STEADY, 6.0))
                   & (day["cum_vs_pdc"] >= 0.01))
    if mask_steady.any():
        idx = mask_steady.idxmax()
        if bool(mask_steady.loc[idx]) and keep(idx):
            r = day.loc[idx]
            hits.append({"detector":"STEADY_TREND",
                        "ts_et": r["ts_et"], "time_et": r["time_et"],
                        "price": float(r["Close"]),
                        "ret": float(r["ret_10m_close"]),
                        "rvol": float(r["rvol_5m"]),
                        "gate":"VWAP+ORH"})

    if not hits:
        return pd.DataFrame(columns=["detector","ts_et","time_et","price","ret","rvol","gate"])

    out = pd.DataFrame(hits).sort_values("ts_et").reset_index(drop=True)
    return out


# ---------- utils ----------
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


# ---------- main ----------
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
            print(f"{tic:<6} no qualified alerts (post-resistance & anti-fade)")
            continue

        print(f"{tic}:")
        for det in res["detector"].unique():
            r = res[res["detector"]==det].iloc[0]
            print(f"  {det:<20} {r['time_et']}  gate={r['gate']:<8} "
                  f"ret={'' if pd.isna(r['ret']) else f'{r['ret']*100:4.1f}%'}  "
                  f"rvol={r['rvol']:.2f}")

        res = res.copy()
        res["ticker"] = tic
        res["ts_utc"] = pd.to_datetime(res["ts_et"]).dt.tz_convert("UTC")
        all_rows.append(res[["ticker","detector","gate","ts_utc","ts_et","time_et","price","ret","rvol"]])

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).sort_values(["ticker","ts_utc"])
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results for {out['ticker'].nunique()} tickers to {OUTPUT_CSV}")
    else:
        print("\nNo qualifying alerts. Consider lowering thresholds slightly.")

if __name__ == "__main__":
    main()
