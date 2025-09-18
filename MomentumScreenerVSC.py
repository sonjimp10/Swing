"""
Small-Cap Momentum Screener (1–5 day timing bias) — ZERO-ARG RUN + FLOAT SHARES
-------------------------------------------------------------------------------
Runs with built‑in config so you can just press ▶ Run.
Writes three CSVs into OUT_DIR:
  • screener_candidates_YYYYMMDD.csv – tickers passing the score threshold
  • screener_full_YYYYMMDD.csv       – all computed features (for auditing)
  • screener_debug_YYYYMMDD.csv      – diagnostics for every symbol (even failures)

Includes:
  • Reads `FloatShares` from your universe CSV and adds it to all outputs.
  • Improved **squeeze**: percentile + Keltner-channel test + ATR% cap.
  • Volume pressure: 5‑day up/down volume ratio, OBV trend, Chaikin Money Flow.
  • All‑time‑low flags and distance to ATL.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import sys

# =========================
# RUN CONFIG (edit these)
# =========================
RUN = {
    # Paths
    "UNIVERSE": r"/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/VerySmallCap.csv",
    "DATA_DIR": r"/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ALPACA_VSC_DAILY_DATA",
    "OUT_DIR":  r"/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading",

    # Core thresholds (permissive for small caps / short history)
    "MIN_PRICE": 0.5,
    "MIN_AVG_VOL": 50_000,           # 20D average shares traded
    "MIN_AVG_DOLLAR_VOL": 100_000,   # price*SMA20(volume)
    "RVOL_TRIGGER": 1.3,             # breakout day rVol threshold
    "ADX_MIN": 15.0,                 # ADX rising through ~15–20
    "SCORE_THRESHOLD": 3.0,

    # History requirement
    "MIN_ROWS": 30,                  # accept 30–50 row histories

    # Squeeze + indicators
    "BB_LOOKBACK": 20,
    "BB_STD": 2.0,
    "SQUEEZE_WINDOW": 45,            # auto-clamped to len(data)
    "SQUEEZE_PERCENTILE": 35.0,      # wider threshold to catch more coils
    "ATR_LEN": 14,
    "DMI_LEN": 14,
    "EMA_FAST": 12,
    "EMA_SLOW": 26,
    "EMA_SIGNAL": 9,
    "RSI_LEN": 14,
    "RSI2_LEN": 2,
    "ATR_COMPRESSION_MAX": 0.07,     # <=7% ATR% counts as compressed
    "KELTNER_MULT": 1.5,             # BB inside Keltner squeeze test

    # Modes
    "VERBOSE": True,                 # print skip reasons
    "SKIP_LIQUIDITY": False,         # set True to see features without liquidity filters
    "DIAGNOSTICS": True,             # write screener_debug_*.csv
}

# =========================
# Indicator weights (tune as you like)
# =========================
W_SQUEEZE = 1.0
W_BREAKOUT = 2.0
W_RVOL = 1.0
W_DMI = 1.0
W_ADX = 1.0
W_MACD = 1.0
W_RSI = 1.0

# =========================
# Helpers: indicators
# =========================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=max(2, span//2)).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


def macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=max(2, length//2)).mean()


def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = true_range(high, low, close)
    atr_sm = tr.ewm(alpha=1/length, adjust=False, min_periods=max(2, length//2)).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_sm)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_sm)
    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=max(2, length//2)).mean()
    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)

@dataclass
class TickerResult:
    symbol: str
    last_date: pd.Timestamp
    last_close: float
    avg_vol20: float
    rvol: float
    bb_width: float
    squeeze: bool
    breakout20: bool
    nr7: bool
    inside_day: bool
    di_plus: float
    di_minus: float
    adx: float
    adx_rising: bool
    macd_hist: float
    macd_up: bool
    rsi14: float
    rsi2: float
    score: float
    reason: str

# =========================
# Core logic
# =========================

def read_universe(universe_path: Path) -> pd.DataFrame:
    df = pd.read_csv(universe_path)
    return df


def normalize_ticker_column(df_u: pd.DataFrame) -> str:
    tick_col = None
    for col in ["symbol", "ticker", "Symbol", "Ticker", "SYM", "TICKER"]:
        if col in df_u.columns:
            tick_col = col
            break
    if tick_col is None:
        tick_col = df_u.columns[0]
    df_u["__ticker"] = df_u[tick_col].astype(str).str.upper().str.strip()
    return "__ticker"


def find_ticker_file(data_dir: Path, symbol: str) -> Optional[Path]:
    exact = data_dir / f"{symbol}.csv"
    if exact.exists():
        return exact
    alt = data_dir / f"{symbol}_daily.csv"
    if alt.exists():
        return alt
    cands = list(data_dir.glob(f"*{symbol}*.csv"))
    if not cands:
        return None
    cands.sort(key=lambda p: (0 if p.stem.upper() == symbol else (1 if p.stem.upper().startswith(symbol) else 2), len(p.stem)))
    return cands[0]


def load_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    date_col = None
    for candidate in ["date", "timestamp", "time", "datetime"]:
        if candidate in cols:
            date_col = cols[candidate]
            break
    if date_col is None:
        return None
    def pick_case(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    o = pick_case("open")
    h = pick_case("high")
    l = pick_case("low")
    c = pick_case("close", "adj_close", "adjclose")
    v = pick_case("volume", "vol")
    if not all([o, h, l, c, v]):
        return None
    out = df[[date_col, o, h, l, c, v]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out.sort_values("date", inplace=True)
    out.dropna(subset=["date", "open", "high", "low", "close", "volume"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MAs & volumes
    df["sma20_vol"] = df["volume"].rolling(20).mean()
    df["dollar_vol20"] = df["close"] * df["sma20_vol"]
    # RVOL
    df["rvol20"] = df["volume"] / df["sma20_vol"]
    # ATR and ATR%
    df["atr14"] = atr(df["high"], df["low"], df["close"], RUN["ATR_LEN"])
    df["atrpct"] = df["atr14"] / df["close"]
    # Bollinger
    mid = df["close"].rolling(RUN["BB_LOOKBACK"]).mean()
    std = df["close"].rolling(RUN["BB_LOOKBACK"]).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_up"] = mid + RUN["BB_STD"] * std
    df["bb_low"] = mid - RUN["BB_STD"] * std
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / mid
    # Keltner Channels
    ema20 = _ema(df["close"], 20)
    kc_up = ema20 + RUN["KELTNER_MULT"] * df["atr14"]
    kc_low = ema20 - RUN["KELTNER_MULT"] * df["atr14"]
    df["kc_up"], df["kc_low"] = kc_up, kc_low

    # Squeeze percentile with adaptive window (works on short histories)
    eff_win = max(20, min(RUN["SQUEEZE_WINDOW"], max(len(df) - 5, 20)))
    def rolling_percentile(s: pd.Series, window: int, pct: float) -> pd.Series:
        return s.rolling(window).apply(lambda x: np.nanpercentile(x, pct), raw=False)
    df["bb_width_pctl"] = rolling_percentile(df["bb_width"], eff_win, RUN["SQUEEZE_PERCENTILE"])

    # Squeeze conditions
    cond_pctile = df["bb_width"] <= df["bb_width_pctl"]
    cond_keltner = (df["bb_up"] <= df["kc_up"]) & (df["bb_low"] >= df["kc_low"])  # BB inside KC
    cond_atr = df["atrpct"] <= RUN["ATR_COMPRESSION_MAX"]

    df["squeeze_on"] = (cond_pctile | cond_keltner | cond_atr).astype(int)
    df["squeeze_score"] = cond_pctile.astype(int) + cond_keltner.astype(int) + cond_atr.astype(int)

    # Breakout vs 20D high (prev window to avoid same-day bias)
    df["rollmax20_prev"] = df["high"].shift(1).rolling(20).max()
    df["breakout20"] = (df["close"] >= df["rollmax20_prev"]).astype(int)
    df["dist_to_20h"] = ((df["rollmax20_prev"] - df["close"]) / df["close"]).clip(lower=-1, upper=1)

    # NR7 & Inside-day
    df["range"] = df["high"] - df["low"]
    df["nr7"] = (df["range"] <= df["range"].shift(1).rolling(7).min()).astype(int)
    df["inside"] = ((df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))).astype(int)

    # DMI / ADX
    plus_di, minus_di, adx_v = dmi_adx(df["high"], df["low"], df["close"], RUN["DMI_LEN"])
    df["di_plus"] = plus_di
    df["di_minus"] = minus_di
    df["adx"] = adx_v
    df["adx_rising"] = (df["adx"].diff() > 0).astype(int)

    # MACD
    _, _, hist = macd_hist(df["close"], RUN["EMA_FAST"], RUN["EMA_SLOW"], RUN["EMA_SIGNAL"])
    df["macd_hist"] = hist
    df["macd_up"] = ((df["macd_hist"] > 0) | (df["macd_hist"].diff() > 0)).astype(int)

    # RSI
    df["rsi14"] = rsi(df["close"], RUN["RSI_LEN"])
    df["rsi2"] = rsi(df["close"], RUN["RSI2_LEN"])

    # 8 EMA for bounce context
    df["ema8"] = _ema(df["close"], 8)

    # Volume pressure metrics
    prev_close = df["close"].shift(1)
    up_day = (df["close"] > prev_close).astype(int)
    df["upvol5"] = (df["volume"] * (up_day == 1)).rolling(5).sum()
    df["downvol5"] = (df["volume"] * (up_day == 0)).rolling(5).sum()
    df["vol_balance5"] = (df["upvol5"] - df["downvol5"]) / (df["upvol5"] + df["downvol5"]).replace(0, np.nan)

    # OBV & trend over 20 days
    df["obv"] = (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()
    df["obv_trend20"] = df["obv"] - df["obv"].shift(20)

    # Chaikin Money Flow (20)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = mfm * df["volume"]
    df["cmf20"] = mfv.rolling(20).sum() / df["volume"].rolling(20).sum()

    # All-time low info (based on provided data)
    df["atl_price"] = df["low"].cummin()
    df["is_atl"] = (df["close"] <= df["atl_price"] * 1.01).astype(int)  # within 1% of dataset ATL
    df["dist_to_atl"] = (df["close"] / df["atl_price"]) - 1.0

    return df


def score_row(row: pd.Series, rvol_trigger: float, adx_min: float) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0
    # Use new squeeze_on instead of old squeeze
    if row.get("squeeze_on", 0) == 1:
        score += W_SQUEEZE; reasons.append("squeeze")
    if row.get("breakout20", 0) == 1:
        score += W_BREAKOUT; reasons.append("20D breakout")
    if row.get("rvol20", 0) >= rvol_trigger:
        score += W_RVOL; reasons.append(f"RVOL≥{rvol_trigger}")
    if row.get("di_plus", 0) > row.get("di_minus", 0):
        score += W_DMI; reasons.append("+DI>-DI")
    if row.get("adx", 0) >= adx_min and row.get("adx_rising", 0) == 1:
        score += W_ADX; reasons.append(f"ADX↑≥{adx_min}")
    if row.get("macd_up", 0) == 1:
        score += W_MACD; reasons.append("MACD up")
    if row.get("rsi14", 0) >= 50 or (row.get("rsi2", 100) <= 5 and row.get("close", 0) >= row.get("ema8", 0)):
        score += W_RSI; reasons.append("RSI trend/bounce")
    return score, reasons


def process_symbol(symbol: str, data_dir: Path) -> Tuple[Optional[TickerResult], dict]:
    dbg = {"symbol": symbol, "status": "", "file": None, "rows": 0,
           "last_date": None, "close": np.nan, "sma20_vol": np.nan, "dollar_vol20": np.nan,
           "rvol": np.nan, "bb_width": np.nan, "squeeze": np.nan, "squeeze_on": np.nan, "squeeze_score": np.nan,
           "breakout20": np.nan, "dist_to_20h": np.nan,
           "nr7": np.nan, "inside_day": np.nan, "+DI": np.nan, "-DI": np.nan, "ADX": np.nan,
           "ADX_rising": np.nan, "MACD_hist": np.nan, "MACD_up": np.nan, "RSI14": np.nan, "RSI2": np.nan,
           "upvol5": np.nan, "downvol5": np.nan, "vol_balance5": np.nan, "cmf20": np.nan, "obv_trend20": np.nan,
           "is_atl": np.nan, "dist_to_atl": np.nan, "atl_price": np.nan,
           "pass_price": np.nan, "pass_avg_vol": np.nan, "pass_dollar_vol": np.nan, "score": np.nan, "reason": ""}

    f = find_ticker_file(Path(data_dir), symbol)
    if f is None:
        dbg["status"] = "no_file"; return None, dbg
    dbg["file"] = str(f)
    df = load_ohlcv(f)
    if df is None:
        dbg["status"] = "bad_schema"; return None, dbg
    dbg["rows"] = len(df)
    if len(df) < RUN["MIN_ROWS"]:
        dbg["status"] = "too_short"
        if RUN["VERBOSE"]:
            print(f"[SKIP] {symbol}: only {len(df)} rows (<{RUN['MIN_ROWS']})")
        return None, dbg

    feats = compute_features(df)
    last = feats.iloc[-1]
    dbg.update({
        "status": "ok_pre_filter",
        "last_date": pd.to_datetime(last["date"]).strftime("%Y-%m-%d"),
        "close": float(last["close"]),
        "sma20_vol": float(last["sma20_vol"]),
        "dollar_vol20": float(last["dollar_vol20"]),
        "rvol": float(last["rvol20"]),
        "bb_width": float(last.get("bb_width", np.nan)) if pd.notna(last.get("bb_width", np.nan)) else np.nan,
        "squeeze": int(last.get("squeeze_on", 0)),
        "squeeze_on": int(last.get("squeeze_on", 0)),
        "squeeze_score": int(last.get("squeeze_score", 0)),
        "breakout20": int(last.get("breakout20", 0)),
        "dist_to_20h": float(last.get("dist_to_20h", np.nan)),
        "nr7": int(last.get("nr7", 0)),
        "inside_day": int(last.get("inside", 0)),
        "+DI": float(last.get("di_plus", np.nan)),
        "-DI": float(last.get("di_minus", np.nan)),
        "ADX": float(last.get("adx", np.nan)),
        "ADX_rising": int(last.get("adx_rising", 0)),
        "MACD_hist": float(last.get("macd_hist", np.nan)),
        "MACD_up": int(last.get("macd_up", 0)),
        "RSI14": float(last.get("rsi14", np.nan)),
        "RSI2": float(last.get("rsi2", np.nan)),
        "upvol5": float(last.get("upvol5", np.nan)),
        "downvol5": float(last.get("downvol5", np.nan)),
        "vol_balance5": float(last.get("vol_balance5", np.nan)),
        "cmf20": float(last.get("cmf20", np.nan)),
        "obv_trend20": float(last.get("obv_trend20", np.nan)),
        "is_atl": int(last.get("is_atl", 0)),
        "dist_to_atl": float(last.get("dist_to_atl", np.nan)),
        "atl_price": float(last.get("atl_price", np.nan)),
    })

    # Liquidity filters (optional)
    price_pass = last["close"] >= RUN["MIN_PRICE"]
    vol_pass = last["sma20_vol"] >= RUN["MIN_AVG_VOL"]
    dvol_pass = last["dollar_vol20"] >= RUN["MIN_AVG_DOLLAR_VOL"]
    dbg["pass_price"] = bool(price_pass)
    dbg["pass_avg_vol"] = bool(vol_pass)
    dbg["pass_dollar_vol"] = bool(dvol_pass)
    if not RUN["SKIP_LIQUIDITY"] and not (price_pass and vol_pass and dvol_pass):
        dbg["status"] = "liquidity_fail"
        if RUN["VERBOSE"]:
            print(f"[SKIP] {symbol}: price={last['close']:.2f} (min {RUN['MIN_PRICE']}), "
                  f"avgVol20={last['sma20_vol']:.0f} (min {RUN['MIN_AVG_VOL']}), $Vol20={last['dollar_vol20']:.0f} (min {RUN['MIN_AVG_DOLLAR_VOL']})")
        return None, dbg

    score, reasons = score_row(last, rvol_trigger=RUN["RVOL_TRIGGER"], adx_min=RUN["ADX_MIN"])
    dbg["status"] = "ok"
    dbg["score"] = float(score)
    dbg["reason"] = ", ".join(reasons)

    res = TickerResult(
        symbol=symbol,
        last_date=pd.to_datetime(last["date"]),
        last_close=float(last["close"]),
        avg_vol20=float(last["sma20_vol"]),
        rvol=float(last["rvol20"]),
        bb_width=float(last.get("bb_width", np.nan)) if pd.notna(last.get("bb_width", np.nan)) else np.nan,
        squeeze=bool(int(last.get("squeeze_on", 0))),
        breakout20=bool(int(last.get("breakout20", 0))),
        nr7=bool(int(last.get("nr7", 0))),
        inside_day=bool(int(last.get("inside", 0))),
        di_plus=float(last.get("di_plus", np.nan)),
        di_minus=float(last.get("di_minus", np.nan)),
        adx=float(last.get("adx", np.nan)),
        adx_rising=bool(int(last.get("adx_rising", 0))),
        macd_hist=float(last.get("macd_hist", np.nan)),
        macd_up=bool(int(last.get("macd_up", 0))),
        rsi14=float(last.get("rsi14", np.nan)),
        rsi2=float(last.get("rsi2", np.nan)),
        score=float(score),
        reason=", ".join(reasons)
    )
    return res, dbg


def main():
    uni_path = Path(RUN["UNIVERSE"]).expanduser()
    data_dir = Path(RUN["DATA_DIR"]).expanduser()
    out_dir = Path(RUN["OUT_DIR"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not uni_path.exists():
        print(f"Universe file not found: {uni_path}", file=sys.stderr); return
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr); return

    df_u = read_universe(uni_path)
    tick_col = normalize_ticker_column(df_u)

    # Build float map from universe if available
    if "FloatShares" in df_u.columns:
        float_map = dict(zip(df_u[tick_col], pd.to_numeric(df_u["FloatShares"], errors="coerce")))
    else:
        float_map = {}

    # Symbols list
    symbols = sorted(set(x for x in df_u[tick_col].astype(str) if x))
    if not symbols:
        print("No symbols found in universe.", file=sys.stderr); return

    results: List[TickerResult] = []
    full_rows = []
    debug_rows = []

    for sym in symbols:
        res, dbg = process_symbol(sym, data_dir)
        # attach float shares to debug
        fs = float_map.get(sym, np.nan)
        dbg["float_shares"] = fs
        debug_rows.append(dbg)
        if res is None:
            continue
        results.append(res)
        full_rows.append({
            "symbol": res.symbol,
            "date": res.last_date,
            "close": res.last_close,
            "avg_vol20": res.avg_vol20,
            "rvol": res.rvol,
            "bb_width": res.bb_width,
            "squeeze_on": dbg["squeeze_on"],
            "squeeze_score": dbg["squeeze_score"],
            "breakout20": dbg["breakout20"],
            "dist_to_20h": dbg["dist_to_20h"],
            "nr7": dbg["nr7"],
            "inside_day": dbg["inside_day"],
            "+DI": dbg["+DI"],
            "-DI": dbg["-DI"],
            "ADX": dbg["ADX"],
            "ADX_rising": dbg["ADX_rising"],
            "MACD_hist": dbg["MACD_hist"],
            "MACD_up": dbg["MACD_up"],
            "RSI14": dbg["RSI14"],
            "RSI2": dbg["RSI2"],
            "upvol5": dbg["upvol5"],
            "downvol5": dbg["downvol5"],
            "vol_balance5": dbg["vol_balance5"],
            "cmf20": dbg["cmf20"],
            "obv_trend20": dbg["obv_trend20"],
            "is_atl": dbg["is_atl"],
            "dist_to_atl": dbg["dist_to_atl"],
            "atl_price": dbg["atl_price"],
            "score": res.score,
            "reason": res.reason,
            "float_shares": fs,
        })

    stamp = pd.Timestamp.today().strftime("%Y%m%d")
    out_full = out_dir / f"screener_full_VSC_{stamp}.csv"
    out_cand = out_dir / f"screener_candidates_VSC_{stamp}.csv"
    out_dbg = out_dir / f"screener_debug_VSC_{stamp}.csv"

    # Write diagnostics always if enabled
    if RUN["DIAGNOSTICS"]:
        pd.DataFrame(debug_rows).to_csv(out_dbg, index=False)

    full_df = pd.DataFrame(full_rows)
    if full_df.empty or "score" not in full_df.columns:
        full_df.to_csv(out_full, index=False)
        print("No valid symbols after feature build / filters.\n"
              "Tips: set SKIP_LIQUIDITY=True in RUN, lower MIN_* thresholds, or confirm OHLCV headers.\n"
              f"Full features saved (may be empty): {out_full}")
        if RUN["DIAGNOSTICS"]:
            print(f"Diagnostics saved: {out_dbg}")
        return

    # Candidates
    candidates_df = full_df[full_df["score"] >= RUN["SCORE_THRESHOLD"]].copy()
    if not candidates_df.empty:
        candidates_df.sort_values(["score", "rvol", "ADX"], ascending=[False, False, False], inplace=True)
    full_df.to_csv(out_full, index=False)
    candidates_df.to_csv(out_cand, index=False)

    print(f"Processed {len(symbols)} symbols. Candidates: {len(candidates_df)}")
    if not candidates_df.empty:
        print(candidates_df[["symbol","score","rvol","breakout20","squeeze_on","squeeze_score","ADX","ADX_rising","MACD_up","RSI14","vol_balance5","cmf20","float_shares","is_atl","dist_to_atl"]].head(20).to_string(index=False))
        print(f"\nSaved: {out_cand}")
    print(f"Full features saved: {out_full}")
    if RUN["DIAGNOSTICS"]:
        print(f"Diagnostics saved: {out_dbg}")


if __name__ == "__main__":
    main()
