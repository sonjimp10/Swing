#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore

# --------------- CONFIG ---------------
INTRADAY_DIR = Path("ALPACA_DATA")   # CET/CEST timestamps in your CSVs
TICKERS_CSV  = Path("LargeCap.csv")  # columns: Ticker, FloatShares (case-insensitive ok)
target_date  = "2025-08-29"          # ET market date (YYYY-MM-DD)

# 👉 Set the time window in ET (HH:MM). Examples:
# WINDOW_START_ET = "09:30"; WINDOW_END_ET = "10:15"   # market open window
WINDOW_START_ET = "14:00"; WINDOW_END_ET = "15:00"     # closing hour window
STEP_MIN = 5                                           # 5-minute bars
# --------------------------------------


def infer_column(df, candidates, required=True, ctx=""):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Missing column among {candidates} {ctx}")
    return None


def load_tickers_and_floats(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Tickers CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    tcol = infer_column(df, ["ticker", "symbol"], ctx="(tickers file)")
    fcol = infer_column(df, ["floatshares", "float", "shares_float", "free_float"], ctx="(tickers file)")
    df = df[[tcol, fcol]].dropna()
    df[tcol] = df[tcol].astype(str).str.upper().str.strip()
    df[fcol] = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
    return set(df[tcol].tolist()), dict(zip(df[tcol], df[fcol]))


def extract_ticker_from_path(p: Path) -> str:
    stem = p.stem
    tokens = stem.split("_")
    for cand in reversed(tokens):
        if 1 <= len(cand) <= 10 and cand.replace("-", "").isalnum():
            return cand.upper()
    return stem.upper()


def read_single_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = infer_column(df, ["timestamp", "time", "datetime", "date", "datetime_utc", "dt"])
    ocol = infer_column(df, ["open", "o"])
    ccol = infer_column(df, ["close", "c"])
    vcol = infer_column(df, ["volume", "vol", "v"])

    dt = pd.to_datetime(df[tcol], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(ZoneInfo("Europe/Amsterdam"))  # CET/CEST
    dt_et = dt.dt.tz_convert(ZoneInfo("America/New_York"))

    out = pd.DataFrame({
        "time_et": dt_et,
        "open":   pd.to_numeric(df[ocol], errors="coerce"),
        "close":  pd.to_numeric(df[ccol], errors="coerce"),
        "volume": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["time_et", "open", "close", "volume"])

    return out.sort_values("time_et").reset_index(drop=True)


def expected_times(et_date_ts, start_hhmm: str, end_hhmm: str, step_min: int):
    tz_et = ZoneInfo("America/New_York")
    base = pd.Timestamp(et_date_ts.date(), tz=tz_et)
    sh, sm = map(int, start_hhmm.split(":"))
    eh, em = map(int, end_hhmm.split(":"))
    start = base + pd.Timedelta(hours=sh, minutes=sm)
    end   = base + pd.Timedelta(hours=eh, minutes=em)
    # inclusive stepping: start, start+5, ..., end
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += pd.Timedelta(minutes=step_min)
    labels = [t.strftime("%H:%M") for t in times]
    return times, labels, start, end


def compute_cumulative_bar_metrics(df_et: pd.DataFrame, et_date: pd.Timestamp,
                                   float_shares: float,
                                   start_hhmm: str, end_hhmm: str, step_min: int):
    times_dt, time_labels, start, end = expected_times(et_date, start_hhmm, end_hhmm, step_min)

    # Keep only the window
    morning = df_et[(df_et["time_et"] >= start) & (df_et["time_et"] <= end)].copy()
    if morning.empty:
        return pd.DataFrame(columns=["time_et","time_label","cum_green_vol","cum_red_vol",
                                     "cum_green_red_ratio","cum_green_float_ratio"])

    # Per-bar green/red
    green = (morning["close"] >= morning["open"]) * morning["volume"]
    red   = (morning["close"]  < morning["open"]) * morning["volume"]
    morning["green_vol_bar"] = green
    morning["red_vol_bar"]   = red

    # Align to exact stamps; fill missing with zeros
    morning["time_label"] = morning["time_et"].dt.strftime("%H:%M")
    bar_by_label = (morning.groupby("time_label")[["green_vol_bar","red_vol_bar"]]
                    .sum()
                    .reindex(time_labels, fill_value=0)
                    .reset_index())

    # Cumulative across the timeline
    bar_by_label["cum_green_vol"] = bar_by_label["green_vol_bar"].cumsum()
    bar_by_label["cum_red_vol"]   = bar_by_label["red_vol_bar"].cumsum()
    denom = bar_by_label["cum_green_vol"] + bar_by_label["cum_red_vol"]
    bar_by_label["cum_green_red_ratio"] = (bar_by_label["cum_green_vol"] / denom).where(denom > 0, 0.0)

    if float_shares and float_shares > 0:
        bar_by_label["cum_green_float_ratio"] = bar_by_label["cum_green_vol"] / float_shares
    else:
        bar_by_label["cum_green_float_ratio"] = pd.NA

    label_to_ts = {t.strftime("%H:%M"): t for t in times_dt}
    bar_by_label["time_et"] = bar_by_label["time_label"].map(label_to_ts)

    return bar_by_label[["time_et","time_label","cum_green_vol","cum_red_vol",
                         "cum_green_red_ratio","cum_green_float_ratio"]]


def main():
    # Parse ET date
    try:
        et_date = pd.Timestamp(target_date).tz_localize(ZoneInfo("America/New_York"))
    except Exception:
        print("Invalid target_date. Use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)

    # Load tickers + floats
    tickers_set, floats_map = load_tickers_and_floats(TICKERS_CSV)
    if not tickers_set:
        print("No tickers in LargeCap.csv", file=sys.stderr); sys.exit(1)

    rows = []
    for p in INTRADAY_DIR.glob("*.csv"):
        ticker = extract_ticker_from_path(p)
        if ticker not in tickers_set:
            continue
        try:
            df_et = read_single_csv(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}", file=sys.stderr)
            continue

        cum = compute_cumulative_bar_metrics(
            df_et, et_date, floats_map.get(ticker, 0),
            WINDOW_START_ET, WINDOW_END_ET, STEP_MIN
        )
        if cum.empty:
            continue
        cum.insert(0, "ticker", ticker)
        rows.append(cum)

    if not rows:
        print("No data found for the chosen window.", file=sys.stderr)
        return

    # Long: cumulative KPIs per timestamp
    long_df = pd.concat(rows, ignore_index=True)
    long_df["time_et"] = long_df["time_et"].dt.strftime("%Y-%m-%d %H:%M")
    long_outfile = Path(f"window_cum_long_{WINDOW_START_ET.replace(':','')}-{WINDOW_END_ET.replace(':','')}_{et_date.date()}.csv")
    long_df.to_csv(long_outfile, index=False)
    print(f"Wrote {long_outfile.resolve()}")

    # Wide: 4 cumulative columns per time
    def pivot_metric(df, value_col, suffix):
        sub = df.pivot(index="ticker", columns="time_label", values=value_col)
        order = sorted(sub.columns, key=lambda t: (int(t[:2])*60 + int(t[3:])))
        sub = sub.reindex(columns=order)
        sub.columns = [f"{t}_{suffix}" for t in sub.columns]
        return sub

    w_green   = pivot_metric(long_df, "cum_green_vol",        "green_vol")
    w_red     = pivot_metric(long_df, "cum_red_vol",          "red_vol")
    w_grshare = pivot_metric(long_df, "cum_green_red_ratio",  "green_red_ratio")
    w_gfloat  = pivot_metric(long_df, "cum_green_float_ratio","green_float_ratio")

    wide = pd.concat([w_green, w_red, w_grshare, w_gfloat], axis=1).reset_index()

    # Nice ordering
    _, time_labels, _, _ = expected_times(et_date, WINDOW_START_ET, WINDOW_END_ET, STEP_MIN)
    ordered = ["ticker"]
    for t in time_labels:
        ordered += [f"{t}_green_vol", f"{t}_red_vol", f"{t}_green_red_ratio", f"{t}_green_float_ratio"]
    ordered = [c for c in ordered if c in wide.columns or c == "ticker"]
    wide = wide[ordered]

    wide_outfile = Path(f"window_cum_wide_{WINDOW_START_ET.replace(':','')}-{WINDOW_END_ET.replace(':','')}_{et_date.date()}.csv")
    wide.to_csv(wide_outfile, index=False)
    print(f"Wrote {wide_outfile.resolve()}")


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    main()
