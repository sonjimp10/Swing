# IntradayScanner_v5.py — early-but-confirmed entries + micro-confirm + early R→G
# + explicit volume diagnostics (raw volume, two RVOLs, spike multiple)

import sys, glob
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

# ================= CONFIG =================
DATA_DIR        = Path("ALPACA_DATA")
LARGECAP_FILE   = Path("LargeCap.csv")
DATE_TO_CHECK   = "2025-08-20"                 # session date (US/Eastern)
ASSUME_INPUT_TZ = "Europe/Paris"               # or "UTC"
OUTPUT_CSV      = f"LargeCap_diagnostics_{DATE_TO_CHECK}.csv"

# Opening range window
OR_MINUTES = 30

# Momentum thresholds
RET_5M_MIN   = 0.02         # +2% intrabar spike vs prev close
RET_10M_MIN  = 0.012        # steady 10m push
RVOL_MIN     = 1.3
RVOL_STEADY  = 1.2
RVOL_MAX     = 8.0

# Anti-fade (forward retention after signal)
ANTI_FADE_FWD_MIN = 10      # minutes forward
ANTI_FADE_MIN_RET = 0.0025  # +0.25%
ANTI_FADE_MAX_DD  = 0.02    # -2% max DD

# N-day breakout
NDH_LOOKBACK_DAYS = 2
NDH_BREAK_MODE    = "high"  # "high" is earlier than "close"

# Gap/Room filters
MAX_GAP_UP             = 0.06
MIN_ROOM_TO_NDH        = 0.02
REQUIRE_BASE_AFTER_BIG_GAP = True
BASE_OK_AFTER_ET       = "10:00"
RVOL_MAX_AFTER_GAP     = 6.0

# Early-entry helpers
PREZONE_MAX_BELOW   = 0.010      # within 1% under PDH/NDH
EARLY_RVOL_MIN      = 1.1        # a touch easier early
FLAT_OPEN_BOUNDS    = (-0.005, 0.005)
RTG_GAP_BOUNDS      = (-0.03, -0.001)
IGNITE_SPIKE_MIN    = 0.012
RET_10M_EARLY_MIN   = 0.008
LEVEL_EPS           = 0.001

# Micro-confirm (for fast acceptance)
MICRO_CONFIRM_BARS   = 2
MICRO_HOLD_ABOVE_BP  = 0.001      # hold ≥ +0.10% above trigger level
MICRO_MIN_EXT        = 0.003      # extend ≥ +0.30%
MICRO_MAX_DD         = 0.01       # ≤ -1% dip allowed
EARLY_CUTOFF_ET      = "10:35"

# Volume confirmation knobs
ABS_VOL_MIN       = 50_000        # min shares per 5m bar
VOL_SPIKE_FACTOR  = 2.0           # ≥ 2× avg of last 5 bars
VOLUME_MODE       = "any2"        # "all" (strict) or "any2" (any two of the three)
RVOL_COMBINE      = "max"         # "max" of rvol_5m and rvol_tod, or "rvol_5m" or "rvol_tod" explicitly
RVOL_TOD_LOOKBACK = 10            # sessions used for time-of-day RVOL baseline

# Score gate (optional)
USE_SCORE_GATE      = True
SCORE_ACCEPT_THRESH = 4

# ==========================================

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
        return pd.to_datetime(s, format="%Y%m%d %H:%M:%S", errors="raise")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def load_bars_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

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
    """RVOL robust for early bars: rolling(12) mean with expanding fallback."""
    chunk = vol.rolling(bars, min_periods=bars).sum()
    base_roll = chunk.rolling(12, min_periods=3).mean().shift(1)
    base_exp  = chunk.expanding(min_periods=3).mean().shift(1)
    base = base_roll.combine_first(base_exp).fillna(chunk)
    out = (chunk / base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out

def forward_ret_and_dd(close: pd.Series, fwd_bars: int) -> pd.DataFrame:
    fwd = (close.shift(-fwd_bars) / close) - 1.0
    dd  = []
    for i in range(len(close)):
        seg = close.iloc[i : i + fwd_bars + 1]
        if len(seg) < 2: dd.append(np.nan); continue
        dd.append(min(0.0, seg.min()/seg.iloc[0] - 1.0))
    return pd.DataFrame({"fwd_ret": fwd, "fwd_dd": dd}, index=close.index)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def higher_lows(series_low: pd.Series, n: int = 3) -> pd.Series:
    return series_low.rolling(n).apply(lambda x: all(np.diff(x) > 0), raw=True).astype(bool)

def tightening_range(high: pd.Series, low: pd.Series, n: int = 3) -> pd.Series:
    rng = (high - low)
    return (rng <= rng.rolling(n).max().shift(1)) & (rng <= rng.rolling(n).mean())

def opening_range_high(day: pd.DataFrame, minutes: int) -> float:
    bars = max(1, minutes // 5)
    return day.iloc[:bars]["High"].max() if not day.empty else np.nan

def opening_range_low(day: pd.DataFrame, minutes: int) -> float:
    bars = max(1, minutes // 5)
    return day.iloc[:bars]["Low"].min() if not day.empty else np.nan

def or_midpoint(day: pd.DataFrame, minutes: int) -> float:
    oh = opening_range_high(day, minutes)
    ol = opening_range_low(day, minutes)
    return np.nan if (pd.isna(oh) or pd.isna(ol)) else (oh + ol) / 2.0

# Running AVWAP from session low (no future leakage)
def avwap_from_running_low(tp: pd.Series, vol: pd.Series, low_series: pd.Series) -> pd.Series:
    av = np.empty(len(tp)); av[:] = np.nan
    start = 0
    cur_min = np.inf
    for i in range(len(tp)):
        lv = low_series.iloc[i]
        if lv < cur_min - 1e-12:
            cur_min = lv
            start = i
        num = (tp * vol).iloc[start:i+1].sum()
        den = vol.iloc[start:i+1].sum()
        av[i] = num / den if den else np.nan
    return pd.Series(av, index=tp.index)

# -------- Prior High helpers --------
def prior_day_high(d: pd.DataFrame, date_str: str) -> float:
    prev = d[d["date_et"] < date_str].tail(78)
    return prev["High"].max() if not prev.empty else np.nan

def ndays_prior_high(d: pd.DataFrame, date_str: str, n_days: int) -> float:
    all_dates = d["date_et"].unique().tolist()
    prev_dates = sorted([x for x in all_dates if x < date_str])
    if not prev_dates: return np.nan
    pick = prev_dates[-n_days:] if len(prev_dates) >= n_days else prev_dates
    prev_bars = d[d["date_et"].isin(pick)]
    return prev_bars["High"].max() if not prev_bars.empty else np.nan

# -------- Time-of-day RVOL baseline --------
def rvol_time_of_day(d: pd.DataFrame, date_str: str, lookback_days: int) -> pd.Series:
    """Compute time-of-day RVOL: today's bar volume / mean volume at the same clock time over past sessions."""
    # past sessions only
    prev = d[d["date_et"] < date_str].copy()
    # keep same 5-min clock labels
    prev["time_et"] = prev["ts"].dt.tz_convert("US/Eastern").dt.strftime("%H:%M")
    day  = d[d["date_et"] == date_str].copy()
    day["time_et"] = day["ts"].dt.tz_convert("US/Eastern").dt.strftime("%H:%M")

    # last N sessions
    unique_dates = sorted(prev["date_et"].unique())[-lookback_days:]
    prevN = prev[prev["date_et"].isin(unique_dates)].copy()

    if prevN.empty:
        return pd.Series(1.0, index=day.index)

    mean_by_time = prevN.groupby("time_et")["Volume"].mean()
    # map today's times to their mean; fallback to overall mean to avoid NaN
    fallback = prevN["Volume"].mean()
    mapped = day["time_et"].map(lambda t: mean_by_time.get(t, fallback))
    rvol_tod = day["Volume"] / mapped.replace(0, np.nan)
    rvol_tod = rvol_tod.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return rvol_tod

# --------------- Scoring ---------------
def early_score(row, vwap_val, pdh, ndh) -> int:
    score = 0
    if not np.isnan(vwap_val) and row["Close"] > vwap_val: score += 2
    if pd.notna(pdh):
        dist_pdh = (pdh / row["Close"]) - 1.0
        if 0.0 <= dist_pdh <= PREZONE_MAX_BELOW: score += 1
    if pd.notna(ndh):
        dist_ndh = (ndh / row["Close"]) - 1.0
        if 0.0 <= dist_ndh <= PREZONE_MAX_BELOW: score += 1
    if row.get("rvol_used", np.nan) >= EARLY_RVOL_MIN: score += 2
    if row.get("ema9", np.nan) > row.get("ema20", np.nan): score += 1
    if row.get("hl3", False): score += 1
    if row.get("tight", False): score += 1
    if row.get("cum_vs_pdc", -1) >= 0.0: score += 1
    return score

# --------------- Micro-confirm helpers ---------------
def _time_le(day: pd.DataFrame, ix: int, et_str: str) -> bool:
    hh, mm = map(int, str(day.loc[ix, "time_et"]).split(":"))
    eh, em = map(int, et_str.split(":"))
    return (hh, mm) <= (eh, em)

def micro_confirm(day: pd.DataFrame, ix: int, level: float) -> bool:
    j = day.index.get_loc(ix)
    tail = day.iloc[j : j + MICRO_CONFIRM_BARS + 1]
    if len(tail) < 2:
        return False
    p0 = day.loc[ix, "Close"]
    maxp = tail["High"].max()
    minp = tail["Low"].min()
    if (tail["Low"].min() < level * (1 - MICRO_HOLD_ABOVE_BP)):
        return False
    ext = maxp / p0 - 1.0
    dd  = minp / p0 - 1.0
    return (ext >= MICRO_MIN_EXT) and (dd >= -MICRO_MAX_DD)

# --------------- Volume confirmation builder ---------------
def build_volume_confirms(day: pd.DataFrame) -> pd.Series:
    # choose RVOL source
    if RVOL_COMBINE == "rvol_5m":
        rvol_used = day["rvol_5m"]
    elif RVOL_COMBINE == "rvol_tod":
        rvol_used = day["rvol_tod"]
    else:
        rvol_used = pd.concat([day["rvol_5m"], day["rvol_tod"]], axis=1).max(axis=1)
    day["rvol_used"] = rvol_used

    rvol_ok = rvol_used >= RVOL_MIN
    abs_ok  = day["Volume"] >= ABS_VOL_MIN
    spike_ok = day["Volume"] >= (day["Volume"].shift(1).rolling(5, min_periods=2).mean() * VOL_SPIKE_FACTOR)

    day["rvol_ok"] = rvol_ok
    day["abs_vol_ok"] = abs_ok
    day["vol_spike_ok"] = spike_ok
    day["vol_spike_mult"] = day["Volume"] / day["Volume"].shift(1).rolling(5, min_periods=2).mean()

    if VOLUME_MODE.lower() == "all":
        return rvol_ok & abs_ok & spike_ok
    else:
        conds = pd.concat([rvol_ok, abs_ok, spike_ok], axis=1).astype(int).sum(axis=1)
        return conds >= 2

# --------------- Detectors ---------------
def analyze_one_day(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    d = df.copy()
    d["ts_et"]   = d["ts"].dt.tz_convert("US/Eastern")
    d["date_et"] = d["ts_et"].dt.strftime("%Y-%m-%d")
    d["time_et"] = d["ts_et"].dt.strftime("%H:%M")

    day = d[d["date_et"] == date_str].copy()
    if day.empty:
        return pd.DataFrame(columns=["detector","gate","ts_et","time_et","price","ret",
                                     "rvol","rvol_tod","rvol_used","bar_volume","vol_spike_mult",
                                     "gap_up","room_to_ndh","score","ret_vs_pdc","ret_vs_open",
                                     "rvol_ok","abs_vol_ok","vol_spike_ok"])

    # indicators
    day["VWAP"]  = vwap(day)
    day["ret_5m_high_vs_prev_close"] = (day["High"] / day["Close"].shift(1)) - 1.0
    day["ret_10m_close"]             =  day["Close"] / day["Close"].shift(2) - 1.0
    day["rvol_5m"] = rvol_chunk(day["Volume"], 1)
    day["ema9"]  = ema(day["Close"], 9)
    day["ema20"] = ema(day["Close"], 20)
    day["hl3"]   = higher_lows(day["Low"], 3)
    day["tight"] = tightening_range(day["High"], day["Low"], 3)

    # references from full data
    pdh = prior_day_high(d, date_str)
    orh = opening_range_high(day, OR_MINUTES)
    orl = opening_range_low(day, OR_MINUTES)
    orm = or_midpoint(day, OR_MINUTES)
    prev = d[d["date_et"] < date_str].tail(78)
    pdc  = prev["Close"].iloc[-1] if not prev.empty else np.nan
    tp   = (day["High"] + day["Low"] + day["Close"]) / 3.0
    ndh = ndays_prior_high(d, date_str, NDH_LOOKBACK_DAYS)

    # cumulative vs prior close (safe even if pdc is NaN)
    day["cum_vs_pdc"] = (day["Close"] / pdc - 1.0) if pd.notna(pdc) else np.nan
    # time-of-day RVOL
    day["rvol_tod"] = rvol_time_of_day(d, date_str, RVOL_TOD_LOOKBACK)

    # running AVWAP from current session low
    day["AVWAP_low"] = avwap_from_running_low(tp, day["Volume"], day["Low"])

    # universal returns
    day["ret_vs_pdc"]  = (day["Close"] / pdc - 1.0) if pd.notna(pdc) else np.nan
    day["ret_vs_open"] = (day["Close"] / day["Open"].iloc[0]) - 1.0

    # forward retention
    fwd = forward_ret_and_dd(day["Close"], ANTI_FADE_FWD_MIN//5)
    day["fwd_ret"], day["fwd_dd"] = fwd["fwd_ret"], fwd["fwd_dd"]

    # gap / room
    day_open     = day["Open"].iloc[0]
    gap_up       = (day_open / pdc - 1.0) if pd.notna(pdc) else np.nan
    room_to_ndh  = ((ndh - day_open) / day_open) if pd.notna(ndh) else np.nan

    # helpers
    def _after_base(ix) -> bool:
        try:
            hh, mm = map(int, str(day.loc[ix, "time_et"]).split(":"))
            bh, bm = map(int, BASE_OK_AFTER_ET.split(":"))
            return (hh, mm) >= (bh, bm)
        except Exception:
            return True

    def keep(ix) -> bool:
        r = day.loc[ix]
        return (pd.notna(r["fwd_ret"]) and pd.notna(r["fwd_dd"])
                and r["fwd_ret"] >= ANTI_FADE_MIN_RET
                and r["fwd_dd"]  >= -ANTI_FADE_MAX_DD)

    def _gap_room_allows(ix) -> bool:
        if pd.isna(gap_up) or pd.isna(room_to_ndh) or pd.isna(ndh):
            return True
        if gap_up > MAX_GAP_UP and room_to_ndh < MIN_ROOM_TO_NDH:
            return False
        if REQUIRE_BASE_AFTER_BIG_GAP and gap_up > MAX_GAP_UP and day_open > ndh:
            if not _after_base(ix):
                return False
            if not (RVOL_MIN <= day.loc[ix, "rvol_used"] <= RVOL_MAX_AFTER_GAP):
                return False
        return True

    hits: List[Dict] = []

    # Build shared volume confirm once (and store rvol_used/booleans)
    vol_confirm = build_volume_confirms(day)

    # ========================== CLASSIC DETECTORS (intrabar cross + micro-confirm) ==========================

    def _append_hit(detector: str, gate: str, idx: int, ret_val: float = np.nan):
        r = day.loc[idx]
        sc = early_score(r, r["VWAP"], pdh, ndh) if USE_SCORE_GATE else np.nan
        if (not USE_SCORE_GATE) or (sc >= SCORE_ACCEPT_THRESH):
            hits.append({
                "detector": detector, "gate": gate,
                "ts_et": r["ts"], "time_et": r["time_et"],
                "price": float(r["Close"]), "ret": (float(ret_val) if pd.notna(ret_val) else np.nan),
                "rvol": float(r["rvol_5m"]) if pd.notna(r["rvol_5m"]) else np.nan,
                "rvol_tod": float(r["rvol_tod"]) if pd.notna(r["rvol_tod"]) else np.nan,
                "rvol_used": float(r["rvol_used"]) if pd.notna(r["rvol_used"]) else np.nan,
                "bar_volume": int(r["Volume"]),
                "vol_spike_mult": float(r["vol_spike_mult"]) if pd.notna(r["vol_spike_mult"]) else np.nan,
                "rvol_ok": bool(r["rvol_ok"]), "abs_vol_ok": bool(r["abs_vol_ok"]), "vol_spike_ok": bool(r["vol_spike_ok"]),
                "gap_up": float(gap_up) if pd.notna(gap_up) else np.nan,
                "room_to_ndh": float(room_to_ndh) if pd.notna(room_to_ndh) else np.nan,
                "score": sc,
                "ret_vs_pdc": float(r["ret_vs_pdc"]) if pd.notna(r["ret_vs_pdc"]) else np.nan,
                "ret_vs_open": float(r["ret_vs_open"]) if pd.notna(r["ret_vs_open"]) else np.nan,
            })

    # 1) PDH intrabar cross
    if pd.notna(pdh):
        cross = (day["High"] > pdh) & (day["High"].shift(1) <= pdh) & vol_confirm & (day["rvol_used"] <= RVOL_MAX)
        cross &= (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN)
        if cross.any():
            idx = cross.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                gate = "PDH_FAST" if (_time_le(day, idx, EARLY_CUTOFF_ET) and micro_confirm(day, idx, pdh)) else "PDH"
                _append_hit("PDH_BREAK_RET_RVOL", gate, idx, day["ret_5m_high_vs_prev_close"].loc[idx])

    # 2) ORH intrabar cross
    orh_val = orh
    if pd.notna(orh_val):
        cross = (day["High"] > orh_val) & (day["High"].shift(1) <= orh_val) & vol_confirm & (day["rvol_used"] <= RVOL_MAX)
        cross &= (day["ret_5m_high_vs_prev_close"] >= RET_5M_MIN)
        if cross.any():
            idx = cross.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                gate = "ORH_FAST" if (_time_le(day, idx, EARLY_CUTOFF_ET) and micro_confirm(day, idx, orh_val)) else "ORH"
                _append_hit("ORH_BREAK_RET_RVOL", gate, idx, day["ret_5m_high_vs_prev_close"].loc[idx])

    # 3) Steady trend
    if pd.notna(orh_val):
        mask = ((day["Close"] > day["VWAP"]) & (day["Close"] > orh_val) &
                (day["ret_10m_close"] >= RET_10M_MIN) &
                (day["rvol_used"].between(RVOL_STEADY, 6.0)) &
                (day["cum_vs_pdc"] >= 0.005) & vol_confirm)
        if mask.any():
            idx = mask.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                _append_hit("STEADY_TREND", "VWAP+ORH", idx, day["ret_10m_close"].loc[idx])

    # 4) NDH intrabar cross
    if pd.notna(ndh):
        if NDH_BREAK_MODE.lower() == "high":
            broke = (day["High"] > ndh) & (day["High"].shift(1) <= ndh)
        else:
            broke = (day["Close"] > ndh) & (day["Close"].shift(1) <= ndh)
        mask = broke & vol_confirm
        if mask.any():
            idx = mask.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                gate = "NDH_FAST" if (_time_le(day, idx, EARLY_CUTOFF_ET) and micro_confirm(day, idx, ndh)) else "NDH"
                _append_hit(f"NDH_BREAK_{NDH_LOOKBACK_DAYS}D", gate, idx, np.nan)

    # ========================== EARLY/CONFIRMED PATTERNS ==========================
    in_pre_pdh_zone = pd.notna(pdh) & ((pdh / day["Close"] - 1.0).between(0.0, PREZONE_MAX_BELOW))
    in_pre_ndh_zone = pd.notna(ndh) & ((ndh / day["Close"] - 1.0).between(0.0, PREZONE_MAX_BELOW))
    rv_up = (day["rvol_used"] > day["rvol_used"].shift(1))

    # A) Pre-Breakout Coil
    mask = ((day["Close"] > day["VWAP"]) & (day["rvol_used"] >= EARLY_RVOL_MIN) & rv_up &
            (day["hl3"] | day["tight"]) & (day["ema9"] > day["ema20"]) &
            (in_pre_pdh_zone | in_pre_ndh_zone) & vol_confirm)
    if mask.any():
        idx = mask.idxmax()
        if _gap_room_allows(idx) and keep(idx):
            _append_hit("PREBREAK_COIL", "VWAP+PREZONE", idx, day["ret_10m_close"].loc[idx])

    # B) First Pullback After Ignition
    ign = ((pd.notna(pdh) & (day["High"] > pdh) & (day["Close"] <= pdh*(1+LEVEL_EPS))) |
           (pd.notna(ndh) & (day["High"] > ndh) & (day["Close"] <= ndh*(1+LEVEL_EPS))))
    ign_prev = ign.astype("boolean").shift(1).fillna(False)
    pullback_ok = (ign_prev &
                   ((day["Low"] >= day["VWAP"]) |
                    (pd.notna(pdh) & (day["Low"] >= pdh*(1-LEVEL_EPS))) |
                    (pd.notna(ndh) & (day["Low"] >= ndh*(1-LEVEL_EPS)))))
    hh_break = pullback_ok & (day["High"] > day["High"].shift(1)) & (day["High"] > day["High"].shift(2)) & vol_confirm
    if hh_break.any():
        idx = hh_break.idxmax()
        if _gap_room_allows(idx) and keep(idx):
            _append_hit("PULLBACK_RECLAIM", "RECLAIM", idx, day["ret_10m_close"].loc[idx])

    # C) Flat-Start Momentum
    prev_close = pdc
    flat_open = pd.notna(prev_close) and (FLAT_OPEN_BOUNDS[0] <= (day_open/prev_close - 1.0) <= FLAT_OPEN_BOUNDS[1])
    if flat_open:
        orl_proxy = day["Open"].rolling(6, min_periods=1).min().bfill().ffill()
        mask = ((day["ret_5m_high_vs_prev_close"] >= IGNITE_SPIKE_MIN) &
                (day["rvol_used"] >= EARLY_RVOL_MIN) &
                (day["Close"] > day["VWAP"]) &
                (day["Close"] > orl_proxy) & vol_confirm)
        if mask.any():
            idx = mask.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                _append_hit("FLAT_IGNITE", "VWAP+IGN", idx, day["ret_10m_close"].loc[idx])

    # ========================== EARLY REVERSALS (R→G) ==========================
    # D1) Early R→G via OR-mid (no VWAP wait)
    if not pd.isna(orm):
        mask = ((day["Close"] > orm) &
                (day["Low"] >= day["Low"].cummin()) &
                (day["rvol_used"] >= EARLY_RVOL_MIN) &
                (day["ret_10m_close"] >= RET_10M_EARLY_MIN) & vol_confirm)
        if mask.any():
            idx = mask.idxmax()
            if _time_le(day, idx, "11:15") and _gap_room_allows(idx) and keep(idx) and micro_confirm(day, idx, orm):
                _append_hit("RTG_ORMID_FAST", "ORMID_RECLAIM", idx, day["ret_10m_close"].loc[idx])

    # D2) Early R→G via running AVWAP-from-low
    mask = ((day["Close"] > day["AVWAP_low"]) &
            (day["rvol_used"] >= EARLY_RVOL_MIN) &
            (day["ret_10m_close"] >= RET_10M_EARLY_MIN) & vol_confirm)
    if mask.any():
        idx = mask.idxmax()
        if _time_le(day, idx, "11:15") and _gap_room_allows(idx) and keep(idx) and micro_confirm(day, idx, float(day.loc[idx,"AVWAP_low"])):
            _append_hit("RTG_AVWAP_FAST", "AVWAPLOW_RECLAIM", idx, day["ret_10m_close"].loc[idx])

    # Classic R→G (VWAP reclaim) as fallback
    gap_dn = pd.notna(pdc) and (RTG_GAP_BOUNDS[0] <= (day_open/pdc - 1.0) <= RTG_GAP_BOUNDS[1])
    if gap_dn:
        mask = ((day["Close"] > day["VWAP"]) &
                (day["ret_10m_close"] >= RET_10M_EARLY_MIN) &
                (day["rvol_used"] >= EARLY_RVOL_MIN) & vol_confirm)
        if mask.any():
            idx = mask.idxmax()
            if _gap_room_allows(idx) and keep(idx):
                _append_hit("RED_TO_GREEN", "VWAP_RECLAIM", idx, day["ret_10m_close"].loc[idx])

    if not hits:
        return pd.DataFrame(columns=["detector","gate","ts_et","time_et","price","ret",
                                     "rvol","rvol_tod","rvol_used","bar_volume","vol_spike_mult",
                                     "gap_up","room_to_ndh","score","ret_vs_pdc","ret_vs_open",
                                     "rvol_ok","abs_vol_ok","vol_spike_ok"])

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

        bars = bars.copy()
        bars["ts_et"]   = bars["ts"].dt.tz_convert("US/Eastern")
        bars["date_et"] = bars["ts_et"].dt.strftime("%Y-%m-%d")

        res = analyze_one_day(bars, DATE_TO_CHECK)
        if res.empty:
            print(f"{tic:<6} no qualified alerts")
            continue

        print(f"{tic}:")
        for det in res["detector"].unique():
            sub = res[res["detector"] == det].sort_values("ts_et")
            r = sub.iloc[0]
            rv = "" if pd.isna(r.get("rvol_used")) else f" rvol={r['rvol_used']:.2f}"
            vol = "" if pd.isna(r.get("bar_volume")) else f" vol={int(r['bar_volume']):,}"
            spk = "" if pd.isna(r.get("vol_spike_mult")) else f" spike×={r['vol_spike_mult']:.1f}"
            print(f"  {det:<22} {r['time_et']}  gate={r['gate']:<14}"
                  f"{rv}{vol}{spk}  gap={r.get('gap_up',np.nan):+.1%}"
                  f"  roomNDH={r.get('room_to_ndh',np.nan):+.1%}")

        res = res.copy()
        res["ticker"] = tic
        res["ts_utc"] = pd.to_datetime(res["ts_et"]).dt.tz_convert("UTC")
        all_rows.append(res[[
            "ticker","detector","gate","ts_utc","ts_et","time_et",
            "price","ret","ret_vs_pdc","ret_vs_open",
            "rvol","rvol_tod","rvol_used","bar_volume","vol_spike_mult",
            "rvol_ok","abs_vol_ok","vol_spike_ok",
            "gap_up","room_to_ndh","score"
        ]])

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True).sort_values(["ticker","ts_utc"])
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results for {out['ticker'].nunique()} tickers to {OUTPUT_CSV}")
    else:
        print("\nNo qualifying alerts. Consider tweaking thresholds or modes.")

if __name__ == "__main__":
    main()
