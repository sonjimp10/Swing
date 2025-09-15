#!/usr/bin/env python3
"""
Small-Cap Momentum Screener (1–5 day timing bias) — with DIAGNOSTICS
--------------------------------------------------------------------
Scans a universe of tickers, computes key signals:
  • Liquidity filters (price, volume, dollar-volume)
  • Volatility squeeze (BB Width & ATR%)
  • Breakout triggers (20‑day high, NR7 / Inside‑day)
  • RVOL (relative volume)
  • ADX/DMI (trend strength + direction)
  • MACD histogram turn / cross
  • RSI(14) trend context and RSI(2) extreme bounce

Now includes a **--diagnostics** mode that writes a per-symbol audit CSV
with the *actual numbers seen* and the pass/fail reason for each filter.

Outputs:
  • screener_candidates_YYYYMMDD.csv – tickers passing the score threshold
  • screener_full_YYYYMMDD.csv       – all computed features (for auditing)
  • screener_debug_YYYYMMDD.csv      – diagnostics for every symbol (even failures)

USAGE (edit PATHS or pass via CLI):
  python small_cap_momentum_screener.py \
      --universe "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/VerySmallCap.csv" \
      --data-dir "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ALPACA_VSC_DAILY_DATA" \
      --out-dir  "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading" \
      --min-price 0.5 --min-avg-vol 50000 --min-dollar-vol 100000 \
      --rvol-trigger 1.3 --adx-min 15 --min-rows 100 --score-threshold 3 \
      --diagnostics --verbose

DATA expectations:
  • One CSV per ticker in --data-dir with daily OHLCV. Common schemas handled:
      date,timestamp,time  | open,high,low,close | volume
    Example file names tried in this order: TICKER.csv, TICKER_daily.csv, *TICKER*.csv

This script avoids external TA libraries; indicators are implemented in pandas.
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# =========================
# Defaults (can be overridden with CLI)
# =========================
DEFAULT_UNIVERSE = \
    "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/VerySmallCap.csv"
DEFAULT_DATA_DIR = \
    "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ALPACA_VSC_DAILY_DATA"
DEFAULT_OUT_DIR = \
    "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading"

# Screener knobs (permissive defaults for small caps)
MIN_PRICE = 0.5
MIN_AVG_VOL = 50_000              # 20D average shares traded
MIN_AVG_DOLLAR_VOL = 100_000      # price*SMA20(volume) minimum
RVOL_TRIGGER = 1.3                # breakout day rVol threshold
ADX_MIN = 15.0                    # ADX rising through ~15–20 is good
BB_LOOKBACK = 20                  # Bollinger base period
BB_STD = 2.0
SQUEEZE_WINDOW = 90               # ~4–5 months
SQUEEZE_PERCENTILE = 30.0         # BB width in lowest 30% of window
ATR_LEN = 14
DMI_LEN = 14
EMA_FAST = 12                     # MACD params
EMA_SLOW = 26
EMA_SIGNAL = 9
RSI_LEN = 14
RSI2_LEN = 2
MIN_ROWS = 100                    # minimum rows required per symbol

# Score weights (tune as you like)
W_SQUEEZE = 1.0
W_BREAKOUT = 2.0
W_RVOL = 1.0
W_DMI = 1.0
W_ADX = 1.0
W_MACD = 1.0
W_RSI = 1.0
SCORE_THRESHOLD = 3.0

# =========================
# Helpers: indicators
# =========================

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
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


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()


def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Directional movements
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = true_range(high, low, close)
    atr_sm = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_sm)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_sm)

    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

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

def read_universe(universe_path: Path) -> List[str]:
    df = pd.read_csv(universe_path)
    # Try common column names
    for col in ["symbol", "ticker", "Symbol", "Ticker", "SYM", "TICKER"]:
        if col in df.columns:
            syms = df[col].astype(str).str.upper().str.strip()
            return sorted(s for s in syms.unique() if s)
    # If no known columns, assume first column
    first = df.columns[0]
    syms = df[first].astype(str).str.upper().str.strip()
    return sorted(s for s in syms.unique() if s)


def find_ticker_file(data_dir: Path, symbol: str) -> Optional[Path]:
    exact = data_dir / f"{symbol}.csv"
    if exact.exists():
        return exact
    alt = data_dir / f"{symbol}_daily.csv"
    if alt.exists():
        return alt
    # Fallback: any CSV that contains the ticker (avoid substrings by path stem exact match if possible)
    cands = list(data_dir.glob(f"*{symbol}*.csv"))
    if not cands:
        return None
    # Prefer files whose stem equals or starts with symbol
    cands.sort(key=lambda p: (0 if p.stem.upper() == symbol else (1 if p.stem.upper().startswith(symbol) else 2), len(p.stem)))
    return cands[0]


def load_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # Normalize columns
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

    # Basic MAs
    df["sma20_vol"] = df["volume"].rolling(20).mean()
    df["dollar_vol20"] = df["close"] * df["sma20_vol"]

    # RVOL
    df["rvol20"] = df["volume"] / df["sma20_vol"]

    # ATR and ATR%
    df["atr14"] = atr(df["high"], df["low"], df["close"], ATR_LEN)
    df["atrpct"] = df["atr14"] / df["close"]

    # Bollinger Bands and width
    mid = df["close"].rolling(BB_LOOKBACK).mean()
    std = df["close"].rolling(BB_LOOKBACK).std(ddof=0)
    df["bb_mid"] = mid
    df["bb_up"] = mid + BB_STD * std
    df["bb_low"] = mid - BB_STD * std
    df["bb_width"] = (df["bb_up"] - df["bb_low"]) / mid

    # Squeeze flag: width in lowest X% of last N
    def rolling_percentile(s: pd.Series, window: int, pct: float) -> pd.Series:
        # compute rolling percentile threshold
        return s.rolling(window).apply(lambda x: np.nanpercentile(x, pct), raw=False)

    df["bb_width_p30"] = rolling_percentile(df["bb_width"], SQUEEZE_WINDOW, SQUEEZE_PERCENTILE)
    df["squeeze"] = (df["bb_width"] <= df["bb_width_p30"]).astype(int)

    # Breakout vs 20D high (use prior 20 days to avoid same-day bias)
    df["rollmax20_prev"] = df["high"].shift(1).rolling(20).max()
    df["breakout20"] = (df["close"] >= df["rollmax20_prev"]).astype(int)

    # NR7 & Inside-day
    df["range"] = df["high"] - df["low"]
    df["nr7"] = (df["range"] <= df["range"].shift(1).rolling(7).min()).astype(int)
    df["inside"] = ((df["high"] <= df["high"].shift(1)) & (df["low"] >= df["low"].shift(1))).astype(int)

    # DMI / ADX
    plus_di, minus_di, adx_v = dmi_adx(df["high"], df["low"], df["close"], DMI_LEN)
    df["di_plus"] = plus_di
    df["di_minus"] = minus_di
    df["adx"] = adx_v
    df["adx_rising"] = (df["adx"].diff() > 0).astype(int)

    # MACD
    macd_line, signal_line, hist = macd_hist(df["close"], EMA_FAST, EMA_SLOW, EMA_SIGNAL)
    df["macd_hist"] = hist
    df["macd_up"] = ((df["macd_hist"] > 0) | (df["macd_hist"].diff() > 0)).astype(int)

    # RSI
    df["rsi14"] = rsi(df["close"], RSI_LEN)
    df["rsi2"] = rsi(df["close"], RSI2_LEN)

    # 8 EMA for minor trend (used for context only)
    df["ema8"] = _ema(df["close"], 8)

    return df


def score_row(row: pd.Series, rvol_trigger: float, adx_min: float) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    # Squeeze
    if row.get("squeeze", 0) == 1:
        score += W_SQUEEZE
        reasons.append("squeeze")

    # Breakout & RVOL
    if row.get("breakout20", 0) == 1:
        score += W_BREAKOUT
        reasons.append("20D breakout")
    if row.get("rvol20", 0) >= rvol_trigger:
        score += W_RVOL
        reasons.append(f"RVOL≥{rvol_trigger}")

    # DMI / ADX
    if row.get("di_plus", 0) > row.get("di_minus", 0):
        score += W_DMI
        reasons.append("+DI>-DI")
    if row.get("adx", 0) >= adx_min and row.get("adx_rising", 0) == 1:
        score += W_ADX
        reasons.append(f"ADX↑≥{adx_min}")

    # MACD
    if row.get("macd_up", 0) == 1:
        score += W_MACD
        reasons.append("MACD up")

    # RSI context: trend or bounce
    if row.get("rsi14", 0) >= 50 or (row.get("rsi2", 100) <= 5 and row.get("close", 0) >= row.get("ema8", 0)):
        score += W_RSI
        reasons.append("RSI trend/bounce")

    return score, reasons


def process_symbol(symbol: str, data_dir: Path, *, skip_liquidity: bool, verbose: bool,
                   min_rows: int, min_price: float, min_avg_vol: int, min_dollar_vol: int,
                   rvol_trigger: float, adx_min: float) -> Tuple[Optional[TickerResult], dict]:
    debug = {
        "symbol": symbol,
        "status": "",
        "file": None,
        "rows": 0,
        "last_date": None,
        "close": np.nan,
        "sma20_vol": np.nan,
        "dollar_vol20": np.nan,
        "rvol": np.nan,
        "bb_width": np.nan,
        "squeeze": np.nan,
        "breakout20": np.nan,
        "nr7": np.nan,
        "inside_day": np.nan,
        "+DI": np.nan,
        "-DI": np.nan,
        "ADX": np.nan,
        "ADX_rising": np.nan,
        "MACD_hist": np.nan,
        "MACD_up": np.nan,
        "RSI14": np.nan,
        "RSI2": np.nan,
        "pass_price": np.nan,
        "pass_avg_vol": np.nan,
        "pass_dollar_vol": np.nan,
        "score": np.nan,
        "reason": "",
    }

    f = find_ticker_file(data_dir, symbol)
    if f is None:
        debug["status"] = "no_file"
        return None, debug

    debug["file"] = str(f)
    df = load_ohlcv(f)
    if df is None:
        debug["status"] = "bad_schema"
        return None, debug

    debug["rows"] = len(df)
    if len(df) < min_rows:
        debug["status"] = "too_short"
        if verbose:
            print(f"[SKIP] {symbol}: only {len(df)} rows (<{min_rows})")
        return None, debug

    feats = compute_features(df)
    last = feats.iloc[-1]

    # fill debug metrics
    debug.update({
        "status": "ok_pre_filter",
        "last_date": pd.to_datetime(last["date"]).strftime("%Y-%m-%d"),
        "close": float(last["close"]),
        "sma20_vol": float(last["sma20_vol"]),
        "dollar_vol20": float(last["dollar_vol20"]),
        "rvol": float(last["rvol20"]),
        "bb_width": float(last["bb_width"]) if np.isfinite(last["bb_width"]) else np.nan,
        "squeeze": int(last["squeeze"]),
        "breakout20": int(last["breakout20"]),
        "nr7": int(last["nr7"]),
        "inside_day": int(last["inside"]),
        "+DI": float(last["di_plus"]),
        "-DI": float(last["di_minus"]),
        "ADX": float(last["adx"]),
        "ADX_rising": int(last["adx_rising"]),
        "MACD_hist": float(last["macd_hist"]),
        "MACD_up": int(last["macd_up"]),
        "RSI14": float(last["rsi14"]),
        "RSI2": float(last["rsi2"]),
    })

    # Liquidity filters (optional)
    price_pass = last["close"] >= min_price
    vol_pass = last["sma20_vol"] >= min_avg_vol
    dvol_pass = last["dollar_vol20"] >= min_dollar_vol

    debug["pass_price"] = bool(price_pass)
    debug["pass_avg_vol"] = bool(vol_pass)
    debug["pass_dollar_vol"] = bool(dvol_pass)

    if not skip_liquidity and not (price_pass and vol_pass and dvol_pass):
        debug["status"] = "liquidity_fail"
        if verbose:
            print(f"[SKIP] {symbol}: price={last['close']:.2f} (min {min_price}), "
                  f"avgVol20={last['sma20_vol']:.0f} (min {min_avg_vol}), $Vol20={last['dollar_vol20']:.0f} (min {min_dollar_vol})")
        return None, debug

    # Score & reasons
    score, reasons = score_row(last, rvol_trigger=rvol_trigger, adx_min=adx_min)
    debug["status"] = "ok"
    debug["score"] = float(score)
    debug["reason"] = ", ".join(reasons)

    res = TickerResult(
        symbol=symbol,
        last_date=pd.to_datetime(last["date"]),
        last_close=float(last["close"]),
        avg_vol20=float(last["sma20_vol"]),
        rvol=float(last["rvol20"]),
        bb_width=float(last["bb_width"]) if np.isfinite(last["bb_width"]) else np.nan,
        squeeze=bool(int(last["squeeze"])),
        breakout20=bool(int(last["breakout20"])),
        nr7=bool(int(last["nr7"])),
        inside_day=bool(int(last["inside"])),
        di_plus=float(last["di_plus"]),
        di_minus=float(last["di_minus"]),
        adx=float(last["adx"]),
        adx_rising=bool(int(last["adx_rising"])),
        macd_hist=float(last["macd_hist"]),
        macd_up=bool(int(last["macd_up"])),
        rsi14=float(last["rsi14"]),
        rsi2=float(last["rsi2"]),
        score=float(score),
        reason=", ".join(reasons)
    )

    return res, debug


def main():
    parser = argparse.ArgumentParser(description="Small-cap momentum timing screener")
    parser.add_argument("--universe", type=str, default=DEFAULT_UNIVERSE, help="CSV with tickers column (symbol/ticker)")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Folder containing per-ticker OHLCV CSVs")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for CSVs")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD, help="Minimum score to include as candidate")
    parser.add_argument("--skip-liquidity", action="store_true", help="Bypass liquidity filters for debugging")
    parser.add_argument("--verbose", action="store_true", help="Print per-symbol skip reasons")
    parser.add_argument("--diagnostics", action="store_true", help="Write screener_debug_YYYYMMDD.csv with per-symbol metrics and reasons")
    # Threshold overrides
    parser.add_argument("--min-price", type=float, default=MIN_PRICE)
    parser.add_argument("--min-avg-vol", type=int, default=MIN_AVG_VOL)
    parser.add_argument("--min-dollar-vol", type=int, default=MIN_AVG_DOLLAR_VOL)
    parser.add_argument("--rvol-trigger", type=float, default=RVOL_TRIGGER)
    parser.add_argument("--adx-min", type=float, default=ADX_MIN)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    args = parser.parse_args()

    uni_path = Path(args.universe).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not uni_path.exists():
        print(f"Universe file not found: {uni_path}", file=sys.stderr)
        sys.exit(1)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    symbols = read_universe(uni_path)
    if not symbols:
        print("No symbols found in universe.", file=sys.stderr)
        sys.exit(1)

    results: List[TickerResult] = []
    full_rows = []
    debug_rows = []

    for sym in symbols:
        try:
            res, dbg = process_symbol(
                sym, data_dir,
                skip_liquidity=args.skip_liquidity,
                verbose=args.verbose,
                min_rows=args.min_rows,
                min_price=args.min_price,
                min_avg_vol=args.min_avg_vol,
                min_dollar_vol=args.min_dollar_vol,
                rvol_trigger=args.rvol_trigger,
                adx_min=args.adx_min,
            )
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
            res, dbg = None, {"symbol": sym, "status": "error", "reason": str(e)}
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
            "squeeze": res.squeeze,
            "breakout20": res.breakout20,
            "nr7": res.nr7,
            "inside_day": res.inside_day,
            "+DI": res.di_plus,
            "-DI": res.di_minus,
            "ADX": res.adx,
            "ADX_rising": res.adx_rising,
            "MACD_hist": res.macd_hist,
            "MACD_up": res.macd_up,
            "RSI14": res.rsi14,
            "RSI2": res.rsi2,
            "score": res.score,
            "reason": res.reason,
        })

    # Build DataFrames (may be empty!)
    full_df = pd.DataFrame(full_rows)
    stamp = pd.Timestamp.today().strftime("%Y%m%d")
    out_full = out_dir / f"screener_full_{stamp}.csv"
    out_cand = out_dir / f"screener_candidates_{stamp}.csv"
    out_dbg = out_dir / f"screener_debug_{stamp}.csv"

    if args.diagnostics:
        pd.DataFrame(debug_rows).to_csv(out_dbg, index=False)

    if full_df.empty or "score" not in full_df.columns:
        full_df.to_csv(out_full, index=False)
        msg = (
            "No valid symbols after feature build / filters.\
"
            "Tips: run with --diagnostics (check screener_debug_YYYYMMDD.csv), use --skip-liquidity,\
"
            "or relax thresholds: --min-price 0.5 --min-avg-vol 50000 --min-dollar-vol 100000\
"
        )
        print(msg + f"Full features saved (may be empty): {out_full}")
        if args.diagnostics:
            print(f"Diagnostics saved: {out_dbg}")
        return

    # Candidates selection guarded by presence of 'score'
    candidates_df = full_df[full_df["score"] >= args.score_threshold].copy()
    if not candidates_df.empty:
        candidates_df.sort_values(["score", "rvol", "ADX"], ascending=[False, False, False], inplace=True)

    # Save
    full_df.to_csv(out_full, index=False)
    candidates_df.to_csv(out_cand, index=False)

    # Console summary
    print(f"Processed {len(symbols)} symbols. Candidates: {len(candidates_df)}")
    if not candidates_df.empty:
        print(candidates_df[["symbol","score","rvol","breakout20","squeeze","ADX","ADX_rising","MACD_up","RSI14"]].head(20).to_string(index=False))
        print(f"Saved: {out_cand}")
    print(f"Full features saved: {out_full}")
    if args.diagnostics:
        print(f"Diagnostics saved: {out_dbg}")


if __name__ == "__main__":
    main()
