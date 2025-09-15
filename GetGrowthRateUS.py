# growth_builder_from_fmp_v3_only_with_dates.py
# Uses ONLY FMP v3 endpoints:
#   - /api/v3/income-statement
#   - /api/v3/financial-growth
#   - /api/v3/analyst-estimates
#
# What this version does:
# - Builds a reconciled analyst path from latest actual to farthest analyst year, fills gaps geometrically.
# - Computes YoY along that path.
# - DCF glide: 5 explicit years (Y1..Y5) starting the first full FY after "today".
#     * First displayed year is toned per your thresholds ( >50% → ×0.75 ; 35–50% → ×2/3 ; 15–35% → ×0.5 )
#     * Monotone non-increasing glide to Y5 with Y5 ≥ terminal; terminal growth applies only AFTER Y5.
#     * If exact product is infeasible with those constraints, choose best feasible fallback and report ConsensusGap_5yr.
#     * If solution would be flat (q≈1), enforce a tiny decay (FORCE_DECAY_EPS) so it slopes down.
# - Keeps Short/Mid totals vs latest actual (Mid capped at Short+MAX_MID_PREMIUM).
# - Keeps/exports all your date fields, plan scaffold, and audits.

import time
from datetime import date, datetime
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# USER CONFIG (REQUIRED)
# =========================
WACC_XLSX_PATH = "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/WACC/WACC.xlsx"
WACC_SHEET     = "WACC"

FMP_API_KEY    = "BUy1zjTqw4dpREy1p96iqGvG4npO9qJg"     # <-- REQUIRED
GDP_RATE_US    = 0.025      # terminal growth (e.g., 2.5%)

# Mid cannot exceed short by more than this premium
MAX_MID_PREMIUM = 0.03

# Optional: tame outliers for industry medians
WINSORIZE_INDUSTRY = True
WINSOR_Q_LOW  = 0.05
WINSOR_Q_HIGH = 0.95

# Polite API pacing
FMP_SLEEP_SEC = 0.25

SUMMARY_SHEET_NAME = "Growth_Summary"

# Plan horizon
TARGET_MAX_YEARS = 5  # << you asked to make this 5

# “Today” anchor (per your context)
TODAY = date(2025, 9, 11)

# For fabricating a date when we only have a 'year'
FYE_MONTH_DAY = "09-28"

# Smoothing / tone-down config
SMOOTH_TO_TERMINAL = True
FORCE_DECAY_EPS = 0.01  # enforce small slope (≈1% decay per step) when q≈1 so years aren’t flat

# Tone-down thresholds (applied to the first displayed DCF year’s raw YoY)
# (strict > thresholds; boundaries fall into the lower bucket)
def tone_down_first_year(g):
    if g > 0.50:
        return g * 0.25, ">50%×0.75"
    elif g > 0.35:
        return g * (1.0/3.0), "35–50%×2/3"
    elif g > 0.15:
        return g * 0.50, "15–35%×0.5"
    else:
        return g, "≤15% (no tone)"

# =========================
# Robust HTTP session
# =========================
_session = requests.Session()
_retries = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.mount("http://",  HTTPAdapter(max_retries=_retries))

def fmp_get(url, params=None):
    params = {} if params is None else dict(params)
    params["apikey"] = FMP_API_KEY
    try:
        r = _session.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except:
        return []

# =========================
# FMP v3 helpers
# =========================
def get_income_statement(symbol, limit=10):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
    return fmp_get(url, {"period": "annual", "limit": limit})

def get_financial_growth(symbol, limit=8):
    url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}"
    return fmp_get(url, {"period": "annual", "limit": limit})

def get_analyst_estimates_v3(symbol, limit=50):
    url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
    return fmp_get(url, {"limit": limit})

# =========================
# Numeric helpers
# =========================
def to_real_float(x):
    try:
        if isinstance(x, complex):
            return float(np.real(x))
        return float(x)
    except:
        return np.nan

def safe_float(x):
    try:
        if isinstance(x, complex):
            return float(np.real(x))
        return float(x)
    except:
        return np.nan

def cagr(series_old_to_new):
    arr = [safe_float(x) for x in series_old_to_new if pd.notna(x)]
    if len(arr) < 2:
        return np.nan
    start, end = arr[0], arr[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.nan
    years = len(arr) - 1
    try:
        return (end / start) ** (1/years) - 1.0
    except:
        return np.nan

def pct_growth(new, base):
    new = to_real_float(new); base = to_real_float(base)
    if not np.isfinite(new) or not np.isfinite(base) or base == 0:
        return np.nan
    return (new / base) - 1.0

def make_numeric_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if np.issubdtype(s_num.dtype, np.complexfloating):
        s_num = s_num.apply(lambda v: np.nan if pd.isna(v) else float(np.real(v)))
    return s_num

def winsorize_series(s: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    s_num = make_numeric_series(s)
    a = s_num.to_numpy(dtype="float64")
    a = a[np.isfinite(a)]
    if a.size < 3:
        return s_num
    lo = np.nanpercentile(a, q_low * 100.0)
    hi = np.nanpercentile(a, q_high * 100.0)
    return s_num.clip(lower=lo, upper=hi)

def cap_midterm(short_g, mid_g, cap=0.03):
    short_g = to_real_float(short_g); mid_g = to_real_float(mid_g); cap = to_real_float(cap)
    if not np.isfinite(short_g) or not np.isfinite(mid_g):
        return mid_g
    return min(mid_g, short_g + cap)

# =========================
# Analyst parsing & path building
# =========================
def parse_estimates_with_dates(symbol):
    """
    Returns:
        rev_by_year: dict[int->estimatedRevenueAvg]
        eps_by_year: dict[int->estimatedEpsAvg]
        analyst_dates: list[datetime] (all 'date' fields found)
    """
    est = get_analyst_estimates_v3(symbol, limit=50) or []
    time.sleep(FMP_SLEEP_SEC)
    rev_by_year, eps_by_year, dates = {}, {}, []
    for r in est:
        d = r.get("date")
        if not d:
            continue
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue
        y = dt.year
        rev = to_real_float(r.get("estimatedRevenueAvg"))
        eps = to_real_float(r.get("estimatedEpsAvg"))
        if np.isfinite(rev):
            rev_by_year[y] = rev
        if np.isfinite(eps):
            eps_by_year[y] = eps
        dates.append(dt)
    return rev_by_year, eps_by_year, dates

def build_complete_revenue_path(latest_year, latest_rev, rev_by_year):
    """
    Build a reconciled revenue path from latest_year to far_year, honoring all analyst anchors.
    Between anchors, fill missing years via geometric progression.
    Returns dict: year -> revenue for years [latest_year .. far_year]
    """
    if latest_year is None or not np.isfinite(latest_rev) or latest_rev <= 0 or not rev_by_year:
        return {}

    far_year = max(rev_by_year.keys())
    path = {latest_year: latest_rev}

    anchors = [latest_year] + sorted([y for y in rev_by_year.keys() if y > latest_year])
    anchors = sorted(set(anchors + [far_year]))

    for i in range(len(anchors) - 1):
        y_a = anchors[i]
        y_b = anchors[i+1]
        R_a = path.get(y_a, rev_by_year.get(y_a, np.nan))
        if y_a == latest_year:
            R_a = latest_rev
        R_b = rev_by_year.get(y_b, np.nan)
        if not (np.isfinite(R_a) and R_a > 0 and np.isfinite(R_b) and R_b > 0):
            continue
        gap = y_b - y_a
        if gap == 1:
            path[y_b] = R_b
            continue
        g = (R_b / R_a) ** (1.0 / gap)
        R = R_a
        for step in range(1, gap+1):
            y_fill = y_a + step
            if y_fill < y_b:
                R = R * g
                path[y_fill] = R
            else:
                path[y_fill] = R_b
    return path

# =========================
# Short/Mid and EPS helpers
# =========================
def latest_actuals(symbol):
    """
    Returns:
        latest_rev (float),
        latest_year (int),
        latest_eps (float),
        latest_actual_date (str 'YYYY-MM-DD')
    """
    inc = get_income_statement(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    if not inc:
        return np.nan, None, np.nan, None

    latest_rev = safe_float(inc[0].get("revenue"))
    latest_date = inc[0].get("date")
    latest_year = pd.to_datetime(latest_date).year if latest_date else None

    fg = get_financial_growth(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    latest_eps = safe_float(fg[0].get("eps")) if (fg and fg[0].get("eps") is not None) else np.nan
    return latest_rev, latest_year, latest_eps, latest_date

def hist_5y_rev_cagr(symbol):
    """Compute ~5-year revenue CAGR from annual income statements (last 6 points)."""
    try:
        inc = get_income_statement(symbol, limit=8) or []
        time.sleep(FMP_SLEEP_SEC)
        rev_series = [safe_float(row.get("revenue")) for row in reversed(inc)]
        if len(rev_series) < 6:
            return np.nan
        return cagr(rev_series[-6:])
    except Exception:
        return np.nan

def compute_short_mid_from_estimates_v3_dateaware(symbol, latest_rev, latest_year):
    """
    Short = total change to earliest future year > latest_year
    Mid   = total change to second future year > latest_year
    """
    if latest_year is None or not np.isfinite(latest_rev) or latest_rev <= 0:
        return np.nan, np.nan, None, None

    rev_by_year, _, _ = parse_estimates_with_dates(symbol)
    fut = sorted([y for y in rev_by_year.keys() if y > latest_year])
    if not fut:
        return np.nan, np.nan, None, None

    y1 = fut[0]
    short_total = (rev_by_year[y1] / latest_rev) - 1.0 if np.isfinite(rev_by_year[y1]) else np.nan

    if len(fut) >= 2:
        y2 = fut[1]
        mid_total = (rev_by_year[y2] / latest_rev) - 1.0 if np.isfinite(rev_by_year[y2]) else np.nan
    else:
        y2, mid_total = None, short_total

    return short_total, mid_total, y1, y2

def short_mid_from_estimates(symbol, latest_rev, latest_eps, latest_year):
    """
    EPS sanity + backup revenue path using v3 (date-aware).
    Returns: (short_rev_total, mid_rev_total, short_eps, mid_eps, flags)
    """
    flags = {
        "used_est_short_rev": False, "used_est_mid_rev": False,
        "used_est_short_eps": False, "used_est_mid_eps": False,
        "eps_mid_used_yoy_due_to_sign": False
    }

    rev_by_year, eps_by_year, _ = parse_estimates_with_dates(symbol)
    fut = sorted([y for y in rev_by_year.keys() if y > (latest_year if latest_year is not None else -10)])
    if not fut:
        return np.nan, np.nan, np.nan, np.nan, flags

    y1 = fut[0]
    y2 = fut[1] if len(fut) >= 2 else None

    short_rev = np.nan
    mid_rev   = np.nan
    short_eps = np.nan
    mid_eps   = np.nan

    r1 = rev_by_year.get(y1, np.nan)
    if np.isfinite(r1) and np.isfinite(latest_rev) and latest_rev > 0:
        short_rev = (r1 / latest_rev) - 1.0
        flags["used_est_short_rev"] = True

    e1 = eps_by_year.get(y1, np.nan)
    if np.isfinite(e1) and np.isfinite(latest_eps) and latest_eps != 0:
        short_eps = (e1 / latest_eps) - 1.0
        flags["used_est_short_eps"] = True

    if y2 is not None:
        r2 = rev_by_year.get(y2, np.nan)
        if np.isfinite(r2) and np.isfinite(latest_rev) and latest_rev > 0:
            mid_rev = (r2 / latest_rev) - 1.0
            flags["used_est_mid_rev"] = True

        e2 = eps_by_year.get(y2, np.nan)
        if np.isfinite(latest_eps) and latest_eps > 0 and np.isfinite(e1) and e1 > 0 and np.isfinite(e2) and e2 > 0:
            try:
                mid_eps = (e2 / latest_eps) - 1.0
                flags["used_est_mid_eps"] = True
            except:
                pass
        if not np.isfinite(mid_eps) and np.isfinite(e1) and e1 != 0 and np.isfinite(e2):
            mid_eps = (e2 / e1) - 1.0
            flags["used_est_mid_eps"] = True
            flags["eps_mid_used_yoy_due_to_sign"] = True

    return short_rev, mid_rev, short_eps, mid_eps, flags

def year_to_fye_date(y, month_day="09-28"):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return None
    try:
        y = int(y)
        return f"{y}-{month_day}"
    except Exception:
        return None

def get_next_two_estimate_years_and_dates_v3(symbol, base_year):
    """
    Return (y1, y2, d1, d2) where:
      - y1,y2 are the smallest two estimate years strictly > base_year
      - d1,d2 are the corresponding FMP 'date' strings
    """
    rev_by_year, _, dates = parse_estimates_with_dates(symbol)
    if not rev_by_year:
        return None, None, None, None
    fut = sorted([y for y in rev_by_year.keys() if y > base_year])
    if not fut:
        return None, None, None, None
    y1 = fut[0]
    y2 = fut[1] if len(fut) >= 2 else None

    y_to_date = {}
    for dt in dates:
        y = dt.year
        if y not in y_to_date:
            y_to_date[y] = dt.strftime("%Y-%m-%d")
    d1 = y_to_date.get(y1)
    d2 = y_to_date.get(y2) if y2 is not None else None
    return y1, y2, d1, d2

# =========================
# 5-year smoothing with tone-down on year 1
# =========================
def smooth_monotone5_to_terminal_after_with_tone_down(
    g1_raw: float,
    P_target_5yr: float,     # product over the 5 displayed DCF years
    g_terminal: float,       # GDP_RATE_US
    force_decay_eps: float = FORCE_DECAY_EPS
):
    """
    Returns:
      g1,g2,g3,g4,g5, gap5, g1_raw_report, g1_after_tone, tone_bucket

    Model:
      factors = [k, k q, k q^2, k q^3, k q^4]
      product constraint: k^5 q^10 = P_target_5yr
      monotone: q <= 1
      terminal after Y5: k q^4 >= t, where t = 1 + g_terminal  => q >= (t/k)^(1/4)
    """
    import math

    # Tone-down the first displayed year per your rule
    g1_after, tone_bucket = tone_down_first_year(g1_raw)

    k = 1.0 + g1_after
    t = 1.0 + g_terminal

    if not (math.isfinite(P_target_5yr) and P_target_5yr > 0 and k > 0 and t > 0):
        return (float('nan'),)*5 + (float('nan'), g1_raw, g1_after, tone_bucket)

    # Solve q from product
    q_pow10 = P_target_5yr / (k**5)
    q = (q_pow10 ** (1.0/10.0)) if q_pow10 > 0 else float('inf')

    q_min = (t / k) ** 0.25 if k > 0 else float('inf')
    feasible = (q <= 1.0) and (q >= q_min)

    def to_yoys(kf, qf):
        return (kf-1.0, kf*qf - 1.0, kf*(qf**2) - 1.0, kf*(qf**3) - 1.0, kf*(qf**4) - 1.0)

    # If feasible but q≈1, gently force a slight decay so it’s not flat
    if feasible:
        if q > 1.0 - 1e-9:
            q_use = max(q_min, 1.0 - force_decay_eps)
            g1, g2, g3, g4, g5 = to_yoys(k, q_use)
            P_capped = (k**5) * (q_use**10)
            gap = (P_target_5yr / P_capped) - 1.0 if P_capped > 0 else float('inf')
            return g1, g2, g3, g4, g5, gap, g1_raw, g1_after, tone_bucket
        else:
            g1, g2, g3, g4, g5 = to_yoys(k, q)
            return g1, g2, g3, g4, g5, 0.0, g1_raw, g1_after, tone_bucket

    # Infeasible → best feasible monotone fallback
    if q > 1.0:
        # target product too high; use max feasible product under monotone
        q_use = 1.0 - force_decay_eps
        q_use = max(q_min, q_use)
    elif q < q_min:
        # target product too low; set boundary where Y5 hits terminal
        q_use = q_min
    else:
        q_use = min(1.0 - force_decay_eps, max(q_min, q))

    g1, g2, g3, g4, g5 = to_yoys(k, q_use)
    P_capped = (k**5) * (q_use**10)
    gap = (P_target_5yr / P_capped) - 1.0 if (P_capped > 0) else float('inf')
    return g1, g2, g3, g4, g5, gap, g1_raw, g1_after, tone_bucket

# =========================
# Main
# =========================
def main():
    if not FMP_API_KEY or FMP_API_KEY in {"PUT_YOUR_FMP_API_KEY_HERE", "YOUR KEY"}:
        raise ValueError("FMP_API_KEY is required.")
    if GDP_RATE_US is None:
        raise ValueError("GDP_RATE_US (terminal growth) is required (e.g., 0.025).")

    # Load WACC workbook
    xls = pd.ExcelFile(WACC_XLSX_PATH)
    if WACC_SHEET not in xls.sheet_names:
        raise ValueError(f"Sheet '{WACC_SHEET}' not found in {WACC_XLSX_PATH}")
    base = pd.read_excel(WACC_XLSX_PATH, sheet_name=WACC_SHEET)

    # Identify key columns
    tk_col = next((c for c in base.columns if str(c).strip().lower() == "ticker"), None)
    ind_col = next((c for c in base.columns if str(c).strip().lower() in
                    ("industry group","industry","industry_name","industryname")), None)
    if tk_col is None:
        raise ValueError(f"Ticker column not found. Columns: {list(base.columns)}")
    if ind_col is None:
        base["_Industry_"] = "UNKNOWN"
    else:
        base["_Industry_"] = base[ind_col].astype(str).str.strip()

    base["_Ticker_"] = base[tk_col].astype(str).str.upper().str.strip()
    tickers = base["_Ticker_"].dropna().unique().tolist()

    # 1) Historical 5Y revenue CAGR per ticker
    hist_rows = []
    for sym in tickers:
        rev_cagr_5y = hist_5y_rev_cagr(sym)
        hist_rows.append({"_Ticker_": sym, "rev_cagr_5y": rev_cagr_5y})
    hist_df = pd.DataFrame(hist_rows)
    df = base.merge(hist_df, on="_Ticker_", how="left")

    # 2) Industry medians
    grouped = df.groupby("_Industry_", dropna=False)
    ind_rows = []
    for gname, g in grouped:
        r = make_numeric_series(g["rev_cagr_5y"])
        r_use = winsorize_series(r, WINSOR_Q_LOW, WINSOR_Q_HIGH) if WINSORIZE_INDUSTRY else r
        ind_rows.append({"_Industry_": gname, "ind_rev_cagr_med": r_use.median(skipna=True)})
    ind_df = pd.DataFrame(ind_rows)
    df = df.merge(ind_df, on="_Industry_", how="left")

    # 3) Estimates + path + outputs
    out_rows = []
    sum_rows = []

    for _, r in df.iterrows():
        sym = r["_Ticker_"]
        ind = r["_Industry_"]

        try:
            latest_rev, latest_year, latest_eps, latest_actual_date = latest_actuals(sym)
            base_year_for_fy12 = max(latest_year if latest_year is not None else -10, TODAY.year)

            # Analyst estimates and dates
            rev_by_year, eps_by_year, analyst_dates = parse_estimates_with_dates(sym)
            analyst_proj_date_dt = max(analyst_dates) if analyst_dates else None
            analyst_proj_date = analyst_proj_date_dt.strftime("%Y-%m-%d") if analyst_proj_date_dt else None

            # Short/Mid (totals vs latest actual) + cap
            s_rev_da, m_rev_da, _, _ = compute_short_mid_from_estimates_v3_dateaware(sym, latest_rev, latest_year)
            s_rev_v3, m_rev_v3, s_eps, m_eps, flags = short_mid_from_estimates(sym, latest_rev, latest_eps, latest_year)
            s_rev = s_rev_da if np.isfinite(s_rev_da) else s_rev_v3
            m_rev_raw = m_rev_da if np.isfinite(m_rev_da) else m_rev_v3
            m_rev = cap_midterm(s_rev, m_rev_raw, MAX_MID_PREMIUM)

            # FY1/FY2 audit dates
            y1_sel, y2_sel, y1_fmp_date, y2_fmp_date = get_next_two_estimate_years_and_dates_v3(sym, base_year_for_fy12)
            sel_fy1_date = year_to_fye_date(y1_sel, FYE_MONTH_DAY)
            sel_fy2_date = year_to_fye_date(y2_sel, FYE_MONTH_DAY)

            # ======= Build reconciled path latest->far and compute YoY =======
            adjY = [np.nan]*5
            hidden_first_yoy = np.nan
            recon_display_product = np.nan
            recon_total_product = np.nan
            recon_target_ratio = np.nan
            far_year = None

            raw_product_4yr = np.nan
            smoothed_product_4yr = np.nan
            consensus_gap_4yr = np.nan

            raw_product_5yr = np.nan
            smoothed_product_5yr = np.nan
            consensus_gap_5yr = np.nan

            tone_raw = np.nan
            tone_after = np.nan
            tone_bucket = ""

            if np.isfinite(latest_rev) and latest_year is not None and rev_by_year:
                path = build_complete_revenue_path(latest_year, latest_rev, rev_by_year)
                if path:
                    years_sorted = sorted(path.keys())
                    far_year = years_sorted[-1]
                    # YoY for every year after latest
                    yoy = {}
                    for i in range(1, len(years_sorted)):
                        y = years_sorted[i]
                        prev = years_sorted[i-1]
                        R_prev, R_cur = path[prev], path[y]
                        yoy[y] = (R_cur / R_prev) - 1.0 if (np.isfinite(R_prev) and R_prev > 0 and np.isfinite(R_cur)) else np.nan

                    # Hidden YoY for first year after latest (e.g., 2025 if latest=2024)
                    first_after_latest = latest_year + 1
                    hidden_first_yoy = yoy.get(first_after_latest, np.nan)

                    # Recon target: endpoint ratio latest->far
                    recon_target_ratio = (path[far_year] / path[latest_year]) if (far_year in path and latest_year in path and path[latest_year] > 0) else np.nan

                    # DCF displayed years: 5 explicit years starting after TODAY
                    start_fy = max(latest_year, TODAY.year) + 1  # e.g., 2026
                    display_years_5 = [start_fy + i for i in range(5)]
                    display_years_4 = display_years_5[:4]

                    # Target products (extend with terminal if analyst stops early)
                    start_prev = start_fy - 1
                    def product_target(end_fy):
                        if start_prev not in path or path[start_prev] <= 0:
                            return np.nan
                        if far_year >= end_fy:
                            return path[end_fy] / path[start_prev]
                        factor = path[far_year] / path[start_prev]
                        n_missing = end_fy - far_year
                        return factor * ((1.0 + GDP_RATE_US) ** n_missing)

                    P_target_4yr = product_target(display_years_4[-1])
                    P_target_5yr = product_target(display_years_5[-1])

                    # Raw products from exact path (extend with terminal where missing)
                    def raw_product(years):
                        prod = 1.0
                        for yy in years:
                            g = yoy.get(yy, np.nan)
                            if np.isfinite(g):
                                prod *= (1.0 + g)
                            else:
                                prod *= (1.0 + GDP_RATE_US)
                        return prod
                    raw_product_4yr = raw_product(display_years_4)
                    raw_product_5yr = raw_product(display_years_5)

                    # First displayed raw YoY (the year we tone)
                    g1_raw = yoy.get(display_years_5[0], GDP_RATE_US)

                    if SMOOTH_TO_TERMINAL and np.isfinite(P_target_5yr):
                        g1, g2, g3, g4, g5, gap5, g1_raw_rep, g1_after, tbucket = smooth_monotone5_to_terminal_after_with_tone_down(
                            g1_raw=g1_raw,
                            P_target_5yr=P_target_5yr,
                            g_terminal=GDP_RATE_US,
                            force_decay_eps=FORCE_DECAY_EPS
                        )
                        adjY = [g1, g2, g3, g4, g5]
                        smoothed_product_5yr = (1.0+g1)*(1.0+g2)*(1.0+g3)*(1.0+g4)*(1.0+g5)
                        consensus_gap_5yr = gap5

                        smoothed_product_4yr = (1.0+g1)*(1.0+g2)*(1.0+g3)*(1.0+g4)
                        if np.isfinite(P_target_4yr) and P_target_4yr > 0:
                            consensus_gap_4yr = (P_target_4yr / smoothed_product_4yr) - 1.0

                        tone_raw = g1_raw_rep
                        tone_after = g1_after
                        tone_bucket = tbucket
                    else:
                        # Fallback: use raw YoYs
                        seq = []
                        for yy in display_years_5:
                            g = yoy.get(yy, np.nan)
                            if not np.isfinite(g):
                                g = GDP_RATE_US
                            seq.append(g)
                        while len(seq) < 5:
                            seq.append(GDP_RATE_US)
                        adjY = seq[:5]
                        smoothed_product_5yr = raw_product_5yr
                        consensus_gap_5yr = 0.0
                        smoothed_product_4yr = np.prod([1.0 + x for x in adjY[:4]])
                        if np.isfinite(P_target_4yr) and P_target_4yr > 0:
                            consensus_gap_4yr = (P_target_4yr / smoothed_product_4yr) - 1.0
                        tone_raw = g1_raw
                        tone_after = g1_raw
                        tone_bucket = "raw"

                    recon_display_product = np.prod([1.0 + (g if np.isfinite(g) else 0.0) for g in adjY])

                    # Product from first year after latest through far_year (exact)
                    recon_total_product = 1.0
                    ok = True
                    for yy in range(latest_year + 1, far_year + 1):
                        g = yoy.get(yy, np.nan)
                        if np.isfinite(g):
                            recon_total_product *= (1.0 + g)
                        else:
                            ok = False
                            break
                    if not ok:
                        recon_total_product = np.nan

            else:
                # No analyst: fallback smooth from historical/industry
                ind_med = to_real_float(r.get("ind_rev_cagr_med", np.nan))
                fallback = ind_med if np.isfinite(ind_med) else to_real_float(r.get("rev_cagr_5y", np.nan))
                if np.isfinite(fallback):
                    y1 = fallback
                    y2 = max(fallback*0.8, GDP_RATE_US + 0.005)
                    y3 = max(fallback*0.6, GDP_RATE_US + 0.003)
                    y4 = max(fallback*0.45, GDP_RATE_US + 0.001)
                    y5 = max(fallback*0.35, GDP_RATE_US)
                    adjY = [y1, y2, y3, y4, y5]

            # ======= Plan (informational flat scaffold) =======
            PlanY1 = PlanY2 = PlanY3 = PlanY4 = PlanY5 = np.nan
            PlanAnnualRate = np.nan
            PlanYears = np.nan
            PlanTargetDate = None
            if np.isfinite(latest_rev) and latest_year is not None and rev_by_year:
                fut_pairs = [(y, rev_by_year[y]) for y in sorted(rev_by_year.keys())
                             if y > latest_year and np.isfinite(rev_by_year[y])]
                if fut_pairs:
                    within = [(y, rev) for (y, rev) in fut_pairs if (y - latest_year) <= TARGET_MAX_YEARS]
                    target = within[-1] if within else fut_pairs[0]
                    target_year, target_rev = target
                    g_ann = (target_rev / latest_rev) ** (1.0 / max(1, (target_year - latest_year))) - 1.0
                    yrs = max(1, int(target_year - latest_year))
                    plan = [g_ann] * min(yrs, 5)
                    while len(plan) < 5:
                        plan.append(g_ann)
                    PlanY1, PlanY2, PlanY3, PlanY4, PlanY5 = plan[:5]
                    PlanAnnualRate = g_ann
                    PlanYears = yrs
                    PlanTargetDate = pd.to_datetime(f"{target_year}-{FYE_MONTH_DAY}").date()

            notes = []
            if np.isfinite(m_rev_raw) and m_rev != m_rev_raw:
                notes.append("Mid REV capped to short+premium")
            if 'flags' in locals() and flags.get("eps_mid_used_yoy_due_to_sign"):
                notes.append("EPS mid sanity used YoY due to sign/flip")

            out_rows.append({
                "Ticker": sym,
                "Industry Group": ind,

                "ShortTermRevenueGrowth": s_rev,
                "MidTermRevenueGrowth": m_rev,
                "TerminalGrowth": GDP_RATE_US,

                "ShortTermEPSGrowth_Estimate": locals().get('s_eps', np.nan),
                "MidTermEPSGrowth_Estimate": locals().get('m_eps', np.nan),

                "IndustryMedian_RevCAGR": to_real_float(r.get("ind_rev_cagr_med", np.nan)),

                # Smoothed 5Y DCF path (YoY)
                "AdjGrowthY1": adjY[0],
                "AdjGrowthY2": adjY[1],
                "AdjGrowthY3": adjY[2],
                "AdjGrowthY4": adjY[3],
                "AdjGrowthY5": adjY[4],
                "AdjTerminalGrowth": GDP_RATE_US,

                # Plan scaffold
                "PlanAnnualRate": PlanAnnualRate,
                "PlanYears": PlanYears,
                "PlanTargetDate": PlanTargetDate,
                "PlanY1": PlanY1,
                "PlanY2": PlanY2,
                "PlanY3": PlanY3,
                "PlanY4": PlanY4,
                "PlanY5": PlanY5,

                # Dates / audit
                "LatestActualDate": locals().get('latest_actual_date', None),
                "AnalystProjDate": locals().get('analyst_proj_date', None),
                "SelectedFY1FMPDate": locals().get('y1_fmp_date', None),
                "SelectedFY2FMPDate": locals().get('y2_fmp_date', None),
                "SelectedFY1Year": locals().get('y1_sel', None),
                "SelectedFY2Year": locals().get('y2_sel', None),
                "SelectedFY1Date": locals().get('sel_fy1_date', None),
                "SelectedFY2Date": locals().get('sel_fy2_date', None),

                # Reconciliation audits
                "HiddenYoY_FirstAfterLatest": locals().get('hidden_first_yoy', np.nan),
                "Recon_Product_Displayed": locals().get('recon_display_product', np.nan),
                "Recon_Product_Total": locals().get('recon_total_product', np.nan),
                "Recon_Target_Ratio": locals().get('recon_target_ratio', np.nan),
                "Recon_Error": (locals().get('recon_total_product', np.nan) - locals().get('recon_target_ratio', np.nan))
                               if (np.isfinite(locals().get('recon_total_product', np.nan)) and np.isfinite(locals().get('recon_target_ratio', np.nan))) else np.nan,

                # DCF smoothing audits
                "Raw_Product_2026_2029": locals().get('raw_product_4yr', np.nan),
                "Smoothed_Product_2026_2029": locals().get('smoothed_product_4yr', np.nan),
                "ConsensusGap_2026_2029": locals().get('consensus_gap_4yr', np.nan),
                "Raw_Product_5yr": locals().get('raw_product_5yr', np.nan),
                "Smoothed_Product_5yr": locals().get('smoothed_product_5yr', np.nan),
                "ConsensusGap_5yr": locals().get('consensus_gap_5yr', np.nan),

                # Tone audit (quiet)
                "Tone_FirstYear_Raw": locals().get('tone_raw', np.nan),
                "Tone_FirstYear_After": locals().get('tone_after', np.nan),
                "Tone_Bucket": locals().get('tone_bucket', ""),
            })

            sum_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "UsedEstimates_ShortRev": bool(np.isfinite(s_rev)),
                "UsedEstimates_MidRev": bool(np.isfinite(m_rev)),
                "AppliedWinsorization": WINSORIZE_INDUSTRY,
                "AppliedMidCapRule": (np.isfinite(m_rev_raw) and m_rev != m_rev_raw),
                "EPSMidUsedYoYDueToSign": ('flags' in locals()) and flags.get("eps_mid_used_yoy_due_to_sign", False),
                "Notes": "; ".join(notes) if notes else ""
            })

        except Exception as e:
            ind_med = to_real_float(r.get("ind_rev_cagr_med", np.nan))
            out_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "ShortTermRevenueGrowth": np.nan,
                "MidTermRevenueGrowth": np.nan,
                "TerminalGrowth": GDP_RATE_US,
                "ShortTermEPSGrowth_Estimate": np.nan,
                "MidTermEPSGrowth_Estimate": np.nan,
                "IndustryMedian_RevCAGR": ind_med,
                "AdjGrowthY1": np.nan, "AdjGrowthY2": np.nan, "AdjGrowthY3": np.nan,
                "AdjGrowthY4": np.nan, "AdjGrowthY5": np.nan, "AdjTerminalGrowth": GDP_RATE_US,
                "PlanAnnualRate": np.nan, "PlanYears": np.nan, "PlanTargetDate": None,
                "PlanY1": np.nan, "PlanY2": np.nan, "PlanY3": np.nan, "PlanY4": np.nan, "PlanY5": np.nan,
                "LatestActualDate": None, "AnalystProjDate": None,
                "SelectedFY1FMPDate": None, "SelectedFY2FMPDate": None,
                "SelectedFY1Year": None, "SelectedFY2Year": None,
                "SelectedFY1Date": None, "SelectedFY2Date": None,
                "HiddenYoY_FirstAfterLatest": np.nan,
                "Recon_Product_Displayed": np.nan,
                "Recon_Product_Total": np.nan,
                "Recon_Target_Ratio": np.nan,
                "Recon_Error": np.nan,
                "Raw_Product_2026_2029": np.nan,
                "Smoothed_Product_2026_2029": np.nan,
                "ConsensusGap_2026_2029": np.nan,
                "Raw_Product_5yr": np.nan,
                "Smoothed_Product_5yr": np.nan,
                "ConsensusGap_5yr": np.nan,
                "Tone_FirstYear_Raw": np.nan,
                "Tone_FirstYear_After": np.nan,
                "Tone_Bucket": "error",
            })
            sum_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "UsedEstimates_ShortRev": False,
                "UsedEstimates_MidRev": False,
                "AppliedWinsorization": WINSORIZE_INDUSTRY,
                "AppliedMidCapRule": False,
                "EPSMidUsedYoYDueToSign": False,
                "Notes": f"Error: {e}"
            })

    out_df = pd.DataFrame(out_rows)
    summary_df = pd.DataFrame(sum_rows)

    # 4) Write back: replace WACC sheet & add summary sheet
    base2 = base.copy()
    base2["_Ticker_"] = base2[tk_col].astype(str).str.upper().str.strip()
    merged = base2.merge(out_df, left_on="_Ticker_", right_on="Ticker", how="left")
    merged.drop(columns=["_Ticker_","Ticker"], inplace=True, errors="ignore")

    with pd.ExcelWriter(WACC_XLSX_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        merged.to_excel(w, sheet_name=WACC_SHEET, index=False)
        summary_df.to_excel(w, sheet_name=SUMMARY_SHEET_NAME, index=False)

    print("\nDone. Columns added to WACC sheet:")
    print("  - ShortTermRevenueGrowth (total-change vs latest actual)")
    print("  - MidTermRevenueGrowth   (total-change vs latest actual, capped by Short + MAX_MID_PREMIUM)")
    print("  - ShortTerm/MidTerm EPS sanity")
    print("  - IndustryMedian_RevCAGR")
    print("  - AdjGrowthY1..Y5 (5-year monotone YoY; Y5 ≥ terminal; terminal applies only AFTER Y5)")
    print("  - PlanAnnualRate, PlanYears, PlanTargetDate, PlanY1..Y5 (informational)")
    print("  - Dates: LatestActualDate, AnalystProjDate, SelectedFY1/2 FMPDate + Year + fabricated Date")
    print("  - Reconciliation + smoothing audits (incl. 5-year product gap)")
    print("  - Tone audits: Tone_FirstYear_Raw, Tone_FirstYear_After, Tone_Bucket")

if __name__ == "__main__":
    main()
