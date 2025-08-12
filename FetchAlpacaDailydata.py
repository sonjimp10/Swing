import os
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import alpaca_trade_api as tradeapi
import config

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

def log(msg):
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)

api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_API_SECRET,
    config.BASE_URL,
    api_version="v2",
)

def fetch_daily_bars(ticker: str, days_back: int, max_retries: int = 3, delay: int = 3) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc)
    # small pad for weekends/holidays
    start_dt = end_dt - timedelta(days=days_back + 7)

    start_str = start_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end_str   = end_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    for attempt in range(1, max_retries + 1):
        try:
            log(f"Fetching 1Day SIP bars for {ticker} ({days_back}d lookback)")
            bars = api.get_bars(
                ticker,
                timeframe="1Day",
                start=start_str,
                end=end_str,
                feed="sip",
                # adjustment="all",
            )
            df = bars.df
            if df is None or df.empty:
                return pd.DataFrame()
            if "symbol" in df.columns:
                df = df[df["symbol"] == ticker].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
            return df[keep].copy()
        except Exception as e:
            log(f"Attempt {attempt} failed for {ticker}: {e}")
            time.sleep(delay * attempt)
    log(f"SKIP {ticker} after {max_retries} failed attempts")
    return pd.DataFrame()

def normalize_to_ny_session_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.index = df.index.tz_convert("America/New_York")
    df["Date"] = df.index.date.astype(str)
    df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    df = df.groupby("Date", as_index=False).agg(
        {"Open":"last","High":"last","Low":"last","Close":"last","Volume":"last"}
    )
    try:
        df["Volume"] = df["Volume"].astype("int64")
    except Exception:
        pass
    return df[["Date","Open","High","Low","Close","Volume"]]

def detect_ticker_column(df: pd.DataFrame) -> str:
    for c in ["Ticker","ticker","Symbol","symbol","SYM"]:
        if c in df.columns:
            return c
    return ""

def main():
    parser = argparse.ArgumentParser(description="Fetch Alpaca DAILY bars -> ALPACA_DAILY_DATA/")
    parser.add_argument("--input", default="LargeCap.csv", help="CSV/Excel with tickers")
    parser.add_argument("--sheet", default=None, help="Sheet name if Excel")
    parser.add_argument("--outdir", default="ALPACA_DAILY_DATA", help="Output folder")

    # Choose either months or days (months default = 2)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--months-back", type=int, default=2, help="Calendar months to fetch (default: 2)")
    group.add_argument("--days-back", type=int, help="Exact calendar days to fetch (overrides months)")

    args = parser.parse_args()

    # Determine effective days_back
    if args.days_back is not None:
        days_back = int(args.days_back)
    else:
        # rough conversion: months * 31
        days_back = int(args.months_back) * 31

    if not os.path.exists(args.input):
        log(f"ERROR: input not found: {args.input}")
        sys.exit(1)

    try:
        if args.input.lower().endswith((".xlsx",".xls")):
            tickers_df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        else:
            tickers_df = pd.read_csv(args.input)
    except Exception as e:
        log(f"ERROR reading input: {e}")
        sys.exit(1)

    col = detect_ticker_column(tickers_df)
    if not col:
        log(f"ERROR: no ticker column in {args.input}. Columns: {list(tickers_df.columns)}")
        sys.exit(1)

    tickers = (
        tickers_df[col].astype(str).str.strip()
        .replace("", pd.NA).dropna().unique().tolist()
    )
    if not tickers:
        log("ERROR: no tickers after cleaning.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    log(f"Output: {args.outdir} | Tickers: {len(tickers)} | Lookback: {days_back} days (~{round(days_back/30,1)} months)")

    saved = 0
    for i, tk in enumerate(tickers, 1):
        if i % 150 == 0:
            log(f"Progress: {i}/{len(tickers)}")
        try:
            df = fetch_daily_bars(tk, days_back=days_back)
            if df.empty:
                continue
            out = normalize_to_ny_session_dates(df)
            if out.empty:
                continue
            fp = os.path.join(args.outdir, f"{tk}.csv")
            out.to_csv(fp, index=False, encoding="utf-8")
            saved += 1
        except Exception as e:
            log(f"Failed {tk}: {e}")
            continue

    log(f"Done. Saved {saved}/{len(tickers)} files.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
