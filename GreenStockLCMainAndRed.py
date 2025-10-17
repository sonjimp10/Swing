# -*- coding: utf-8 -*-
"""
Keep only tickers where the last N days are GREEN and the streak is preceded by RED.
GREEN can be defined vs previous close (default) or vs open.
Outputs last_close and Nd-vs-10d volume ratio.

If you get no matches, try:
- GREEN_METHOD="open" (instead of "prev_close")
- PRECEDING_RED_LOOKBACK=2 or 3 (instead of exactly the prior day)
- REQUIRE_BASE_FOR_OUTPUT=False
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# ----------------- CONFIG -----------------
UNIVERSE = Path(r"C:\Users\JMuk\Py Scripts Old\Swing\LargeCap.csv")
DATA_DIR = Path(r"C:\Users\JMuk\Py Scripts Old\Swing\ALPACA_DAILY_DATA")
OUTPUT_CSV = Path(r"C:\Users\JMuk\Py Scripts Old\Swing\outputs\lastN_green_after_red_volume_screen.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DAYS = 2                        # set to 2, 3, ...
BASE_DAYS_FOR_AVG = 10          # comparison baseline
GREEN_STRICT = True             # True: > ; False: >=
GREEN_METHOD = "prev_close"     # "prev_close" or "open"
PRECEDING_RED_LOOKBACK = 1      # 1 = exactly prior day must be red; try 2 or 3 if too strict
REQUIRE_BASE_FOR_OUTPUT = True  # require valid 10d avg volume
DEBUG = False                   # True prints first few failures and tallies
DEBUG_MAX = 25
# ------------------------------------------


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
    date_col = next((cols[c] for c in ["date", "timestamp", "time", "datetime"] if c in cols), None)
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

    # Robust numeric conversion (handles strings like "1,234")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out.sort_values("date", inplace=True)
    out.dropna(subset=["date", "open", "high", "low", "close", "volume"], inplace=True)
    out = out[~out["date"].duplicated(keep="last")].reset_index(drop=True)
    return out


def green_red_series(df: pd.DataFrame, method: str, strict: bool) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (is_green, is_red) boolean Series of same length as df.
    method="prev_close": green if close - prev_close > 0 (or >= 0)
    method="open":       green if close - open       > 0 (or >= 0)
    """
    if method == "open":
        delta = df["close"] - df["open"]
        if strict:
            green = delta > 0
            red = delta < 0
        else:
            green = delta >= 0
            red = delta <= 0
    else:  # "prev_close"
        chg = df["close"].diff()
        if strict:
            green = chg > 0
            red = chg < 0
        else:
            green = chg >= 0
            red = chg <= 0
    # ensure booleans and fill NaN (first row) as False
    green = green.fillna(False)
    red = red.fillna(False)
    return green.astype(bool), red.astype(bool)


def lastN_green_after_red(df: pd.DataFrame, n: int, method: str, strict: bool, red_lookback: int):
    """
    True if:
      - last n days are all GREEN (per method/strict)
      - and any of the `red_lookback` days immediately *before* the streak is RED.
    Requires at least n + red_lookback + 1 rows to evaluate safely.
    """
    if df.shape[0] < n + red_lookback + 1:
        return (None, None, None)

    green, red = green_red_series(df, method, strict)

    # last n indices
    last_n_green = bool(green.iloc[-n:].all())

    # window *before* the streak: [-(n+red_lookback) : -n)
    start = -n - red_lookback
    stop = -n
    preceded_by_red = bool(red.iloc[start:stop].any())

    passed = last_n_green and preceded_by_red
    return (passed, last_n_green, preceded_by_red)


def compute_vol_metrics(volumes: pd.Series, n_days: int, base_days: int):
    n = volumes.shape[0]
    avg_n = volumes.tail(n_days).mean() if n >= n_days else np.nan
    avg_base = volumes.tail(base_days).mean() if n >= base_days else np.nan
    ratio = (avg_n / avg_base) if (pd.notna(avg_n) and pd.notna(avg_base) and avg_base != 0) else np.nan
    return {
        "avg_vol_Nd": avg_n,
        "avg_vol_base": avg_base,
        "ratio_Nd_to_base": ratio,
        "avg_vol_Nd_as_pct_of_base": (ratio * 100.0) if pd.notna(ratio) else np.nan,
    }


def read_universe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    ticker_col = next((cols[c] for c in ["ticker", "symbol"] if c in cols), None)
    if ticker_col is None:
        raise ValueError("Universe file must have a 'Ticker' or 'Symbol' column.")
    df.rename(columns={ticker_col: "Ticker"}, inplace=True)
    if "FloatShares" not in df.columns and "floatshares" in cols:
        df.rename(columns={cols["floatshares"]: "FloatShares"}, inplace=True)
    keep_cols = ["Ticker"] + (["FloatShares"] if "FloatShares" in df.columns else [])
    return df[keep_cols].copy()


def process_universe(universe_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    rows = []
    dbg_count = 0
    stats = {"no_file": 0, "bad_file": 0, "fail_streak": 0, "fail_base": 0, "passed": 0}

    for _, row in universe_df.iterrows():
        symbol = str(row["Ticker"]).strip().upper()
        float_shares = row.get("FloatShares", np.nan)

        fpath = find_ticker_file(data_dir, symbol)
        if fpath is None:
            stats["no_file"] += 1
            if DEBUG and dbg_count < DEBUG_MAX: print(f"[NOFILE] {symbol}")
            dbg_count += 1
            continue

        df = load_ohlcv(fpath)
        if df is None or df.empty:
            stats["bad_file"] += 1
            if DEBUG and dbg_count < DEBUG_MAX: print(f"[BADFILE] {symbol} -> {fpath}")
            dbg_count += 1
            continue

        passed, lastN_all_green, preceded_by_red = lastN_green_after_red(
            df, DAYS, method=GREEN_METHOD, strict=GREEN_STRICT, red_lookback=PRECEDING_RED_LOOKBACK
        )
        if passed is not True:
            stats["fail_streak"] += 1
            if DEBUG and dbg_count < DEBUG_MAX:
                print(f"[FAIL_STREAK] {symbol} | lastN_all_green={lastN_all_green} preceded_by_red={preceded_by_red}")
            dbg_count += 1
            continue

        vm = compute_vol_metrics(df["volume"], DAYS, BASE_DAYS_FOR_AVG)
        if REQUIRE_BASE_FOR_OUTPUT and (pd.isna(vm["avg_vol_base"]) or vm["avg_vol_base"] == 0):
            stats["fail_base"] += 1
            if DEBUG and dbg_count < DEBUG_MAX: print(f"[FAIL_BASE] {symbol} (no valid {BASE_DAYS_FOR_AVG}d avg)")
            dbg_count += 1
            continue

        stats["passed"] += 1
        rows.append({
            "Ticker": symbol,
            "FloatShares": float_shares,
            "last_date": df["date"].iloc[-1],
            "last_close": float(df["close"].iloc[-1]),
            "days_available": int(df.shape[0]),
            "lastN_all_green": lastN_all_green,
            "preceded_by_red": preceded_by_red,
            "avg_vol_Nd": vm["avg_vol_Nd"],
            "avg_vol_base": vm["avg_vol_base"],
            "ratio_Nd_to_base": vm["ratio_Nd_to_base"],
            "avg_vol_Nd_as_pct_of_base": vm["avg_vol_Nd_as_pct_of_base"],
        })

    out = pd.DataFrame(rows)
    if DEBUG:
        print(f"\nSTATS: {stats}")

    if out.empty:
        print(f"No tickers matched: last {DAYS} GREEN ({GREEN_METHOD}, strict={GREEN_STRICT}) "
              f"after RED (lookback={PRECEDING_RED_LOOKBACK}) + valid {BASE_DAYS_FOR_AVG}-day volume.")
        return out

    out.sort_values(["ratio_Nd_to_base", "avg_vol_Nd"], ascending=[False, False], inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Pretty console formatting; raw saved to CSV
    printable = out.copy()
    for col in ["avg_vol_Nd", "avg_vol_base"]:
        printable[col] = printable[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    printable["avg_vol_Nd_as_pct_of_base"] = printable["avg_vol_Nd_as_pct_of_base"].map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else ""
    )
    printable["last_close"] = printable["last_close"].map(lambda x: f"{x:.2f}")

    print(printable.head(100).to_string(index=False))
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} tickers to: {OUTPUT_CSV}")
    return out


def main():
    universe = read_universe(UNIVERSE)
    process_universe(universe, DATA_DIR)

if __name__ == "__main__":
    main()
