# growth_builder_from_fmp_v3_only_with_dates.py
# Uses ONLY FMP v3 endpoints:
#   - /api/v3/income-statement
#   - /api/v3/financial-growth
#   - /api/v3/analyst-estimates
# Keeps your outputs and adds SelectedFY1Date/SelectedFY2Date (fabricated from v3 'year').

import time
from datetime import date
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
GDP_RATE_US    = 0.025          # terminal growth (e.g., 2.5%)

# Rule: mid-term cannot exceed short-term by more than this premium
MAX_MID_PREMIUM = 0.03

# Optional: tame outliers when computing industry medians of historical CAGRs
WINSORIZE_INDUSTRY = True
WINSOR_Q_LOW  = 0.05
WINSOR_Q_HIGH = 0.95

# Polite API pacing
FMP_SLEEP_SEC = 0.25

SUMMARY_SHEET_NAME = "Growth_Summary"

# Fade rule cap for very high short-term growth (set 0.30 for 30%, 0.40 for 40%, etc.)
FADE_SHORT_CAP = 0.30

# Plan horizon (inverse CAGR plan: pick a target year within this many years ahead)
TARGET_MAX_YEARS = 4

# Fabricate an FYE date from a v3 'year' (change month/day if you prefer)
FYE_MONTH_DAY = "09-28"  # e.g., "01-31" or "12-31" etc.

# =========================
# Robust HTTP session with retries
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
    except requests.HTTPError as e:
        # If plan doesn't include endpoint, return empty instead of crashing
        if r is not None and r.status_code in (401, 402, 403):
            return []
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
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
# Numeric helpers (safe)
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

# ===== Fade rule function (adds Y1..Y5 + terminal) =====
def apply_fade_rule(short_term, industry_median, terminal_growth, cap=FADE_SHORT_CAP):
    short_term = to_real_float(short_term)
    industry_median = to_real_float(industry_median)
    terminal_growth = to_real_float(terminal_growth)
    cap = to_real_float(cap)

    if pd.isna(short_term):
        return {
            "AdjGrowthY1": np.nan, "AdjGrowthY2": np.nan, "AdjGrowthY3": np.nan,
            "AdjGrowthY4": np.nan, "AdjGrowthY5": np.nan, "AdjTerminalGrowth": terminal_growth,
            "FadeApplied": False, "FadeNote": "Missing short-term"
        }

    if not np.isfinite(industry_median):
        industry_median = cap  # fallback anchor

    if short_term <= cap:
        return {
            "AdjGrowthY1": short_term, "AdjGrowthY2": short_term, "AdjGrowthY3": short_term,
            "AdjGrowthY4": short_term, "AdjGrowthY5": short_term, "AdjTerminalGrowth": terminal_growth,
            "FadeApplied": False, "FadeNote": "No fade applied"
        }

    # Fade case: ceiling target so Y5 cannot exceed FADE_SHORT_CAP
    y1 = short_term
    y5_target = min(industry_median, FADE_SHORT_CAP)
    fade_years = 4  # Y2..Y5
    step = (y5_target - y1) / fade_years

    return {
        "AdjGrowthY1": y1,
        "AdjGrowthY2": y1 + step,
        "AdjGrowthY3": y1 + 2*step,
        "AdjGrowthY4": y1 + 3*step,
        "AdjGrowthY5": y5_target,
        "AdjTerminalGrowth": terminal_growth,
        "FadeApplied": True,
        "FadeNote": f"Fade applied from {y1:.1%} to {y5_target:.1%}"
    }

# =========================
# Core data extractors (v3)
# =========================
def latest_actuals(symbol):
    """
    Returns: (latest_annual_revenue, latest_actual_year, latest_eps_guess)
    """
    inc = get_income_statement(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    if not inc:
        return np.nan, None, np.nan

    latest_rev = safe_float(inc[0].get("revenue"))
    # derive fiscal year from 'date'
    latest_date = inc[0].get("date")
    latest_year = pd.to_datetime(latest_date).year if latest_date else None

    fg = get_financial_growth(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    latest_eps = safe_float(fg[0].get("eps")) if (fg and fg[0].get("eps") is not None) else np.nan
    return latest_rev, latest_year, latest_eps

def hist_5y_rev_cagr(symbol):
    """
    Compute ~5-year revenue CAGR from annual income statements.
    Uses the last 6 annual points (≈5 intervals), old->new order.
    """
    try:
        inc = get_income_statement(symbol, limit=8) or []
        time.sleep(FMP_SLEEP_SEC)
        rev_series = [safe_float(row.get("revenue")) for row in reversed(inc)]
        if len(rev_series) < 6:
            return np.nan
        return cagr(rev_series[-6:])
    except Exception as e:
        print(f"[{symbol}] hist error: {e}")
        return np.nan

def compute_short_mid_from_estimates_v3_dateaware(symbol, latest_rev, latest_year):
    """
    v3 analyst-estimates has `year` + estimatedRevenueAvg.
    Pick the smallest two years strictly > latest_year.
    If the earliest estimate is >1 year ahead, SHORT & MID are still returned (no annualization here),
    because user will do math in Excel. We also return the selected years.
    Returns: (short_total_change, mid_total_change, y1_sel, y2_sel)
    NOTE: short_total_change = (rev_y1 / latest_rev) - 1.0  (multi-year if gap>1)
          mid_total_change   = (rev_y2 / latest_rev) - 1.0  (multi-year if gap>1)
    """
    if latest_year is None or not np.isfinite(latest_rev) or latest_rev <= 0:
        return np.nan, np.nan, None, None

    est = get_analyst_estimates_v3(symbol, limit=50) or []
    time.sleep(FMP_SLEEP_SEC)
    if not est:
        return np.nan, np.nan, None, None

    pairs = []
    for r in est:
        y = r.get("year")
        rev = r.get("estimatedRevenueAvg")
        if y is None or rev is None:
            continue
        try:
            y = int(y)
            rev = to_real_float(rev)
        except:
            continue
        if np.isfinite(rev):
            pairs.append((y, rev))

    if not pairs:
        return np.nan, np.nan, None, None

    pairs.sort(key=lambda t: t[0])
    fut = [(y, rev) for (y, rev) in pairs if y > latest_year]
    if not fut:
        return np.nan, np.nan, None, None

    # Short: earliest future year (total change vs latest actual)
    y1, rev1 = fut[0]
    short_total = (rev1 / latest_rev) - 1.0

    # Mid: second future year if available (total change vs latest actual)
    if len(fut) >= 2:
        y2, rev2 = fut[1]
        mid_total = (rev2 / latest_rev) - 1.0
    else:
        y2, mid_total = None, short_total

    return short_total, mid_total, y1, y2

def short_mid_from_estimates(symbol, latest_rev, latest_eps, latest_year):
    """
    EPS sanity + backup revenue path using v3.
    Returns: (short_rev_total, mid_rev_total, short_eps, mid_eps, flags)
    (Revenue values here are total-change vs latest actual, not annualized.)
    """
    flags = {
        "used_est_short_rev": False, "used_est_mid_rev": False,
        "used_est_short_eps": False, "used_est_mid_eps": False,
        "eps_mid_used_yoy_due_to_sign": False
    }

    est = get_analyst_estimates_v3(symbol, limit=50) or []
    time.sleep(FMP_SLEEP_SEC)
    if not est:
        return np.nan, np.nan, np.nan, np.nan, flags

    by_year = {}
    for r in est:
        y = r.get("year")
        if y is None:
            continue
        try:
            y = int(y)
        except:
            continue
        by_year[y] = {
            "rev": to_real_float(r.get("estimatedRevenueAvg")),
            "eps": to_real_float(r.get("estimatedEpsAvg")),
        }
    if not by_year:
        return np.nan, np.nan, np.nan, np.nan, flags

    years_sorted = sorted([y for y in by_year.keys() if y > (latest_year if latest_year is not None else -10)])
    if not years_sorted:
        return np.nan, np.nan, np.nan, np.nan, flags

    y1 = years_sorted[0]
    y2 = years_sorted[1] if len(years_sorted) >= 2 else None

    short_rev = np.nan
    mid_rev   = np.nan
    short_eps = np.nan
    mid_eps   = np.nan

    r1 = by_year[y1]["rev"]; e1 = by_year[y1]["eps"]

    if np.isfinite(r1) and np.isfinite(latest_rev) and latest_rev > 0:
        short_rev = (r1 / latest_rev) - 1.0
        flags["used_est_short_rev"] = True

    if np.isfinite(e1) and np.isfinite(latest_eps) and latest_eps != 0:
        short_eps = (e1 / latest_eps) - 1.0
        flags["used_est_short_eps"] = True

    if y2 is not None:
        r2 = by_year[y2]["rev"]; e2 = by_year[y2]["eps"]
        if np.isfinite(r2) and np.isfinite(latest_rev) and latest_rev > 0:
            try:
                mid_rev = (r2 / latest_rev) - 1.0
                flags["used_est_mid_rev"] = True
            except:
                mid_rev = np.nan

        # EPS mid-term sanity (prefer CAGR if all positive; else YoY FY2/FY1); but you can ignore if only cross-check
        if np.isfinite(latest_eps) and latest_eps > 0 and np.isfinite(e1) and e1 > 0 and np.isfinite(e2) and e2 > 0:
            try:
                # not annualized; just total change sanity if you want to use it
                mid_eps = (e2 / latest_eps) - 1.0
                flags["used_est_mid_eps"] = True
            except:
                pass
        if not np.isfinite(mid_eps) and np.isfinite(e1) and e1 != 0 and np.isfinite(e2):
            mid_eps = (e2 / e1) - 1.0
            flags["used_est_mid_eps"] = True
            flags["eps_mid_used_yoy_due_to_sign"] = True

    return short_rev, mid_rev, short_eps, mid_eps, flags

# =========================
# Plan (inverse CAGR) based on v3 estimate years
# =========================
def annualized_rate_to_target_year(rev_base: float, rev_target: float, latest_year: int, target_year: int) -> float:
    if not (np.isfinite(rev_base) and np.isfinite(rev_target)) or rev_base <= 0:
        return np.nan
    n = max(1, int(target_year - latest_year))
    try:
        return (rev_target / rev_base) ** (1.0 / n) - 1.0
    except:
        return np.nan

def plan_yearly_growths(g_annual: float, years: int, max_years_out: int = 5):
    if not np.isfinite(g_annual):
        return [np.nan] * max_years_out
    k = max(1, min(int(years), max_years_out))
    base = [g_annual] * k
    if k < max_years_out:
        base.extend([g_annual] * (max_years_out - k))
    return base

def year_to_fye_date(y, month_day="09-28"):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return None
    try:
        y = int(y)
        return f"{y}-{month_day}"
    except Exception:
        return None
def get_next_two_estimate_years_v3(symbol, latest_year):
    """
    Return (y1, y2) = the smallest two estimate years strictly > latest_year
    using /api/v3/analyst-estimates. Returns (None, None) if not found.
    """
    if latest_year is None:
        return None, None

    est = get_analyst_estimates_v3(symbol, limit=50) or []
    time.sleep(FMP_SLEEP_SEC)
    if not est:
        return None, None

    years = []
    for r in est:
        y = r.get("year")
        if y is None:
            continue
        try:
            y = int(y)
        except:
            continue
        if y > latest_year:
            years.append(y)

    if not years:
        return None, None
    years = sorted(set(years))
    y1 = years[0]
    y2 = years[1] if len(years) >= 2 else None
    return y1, y2

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
        print("Warning: 'Industry Group' column not found; industry medians will be limited.")
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

    # 2) Industry medians (winsorized if chosen)
    grouped = df.groupby("_Industry_", dropna=False)
    ind_rows = []
    for gname, g in grouped:
        r = make_numeric_series(g["rev_cagr_5y"])
        r_use = winsorize_series(r, WINSOR_Q_LOW, WINSOR_Q_HIGH) if WINSORIZE_INDUSTRY else r
        if r_use.dropna().size < 3 and WINSORIZE_INDUSTRY:
            print(f"[Info] Industry '{gname}' has <3 numeric rev_cagr_5y points; winsorization skipped effectively.")
        ind_rows.append({"_Industry_": gname, "ind_rev_cagr_med": r_use.median(skipna=True)})
    ind_df = pd.DataFrame(ind_rows)

    df = df.merge(ind_df, on="_Industry_", how="left")

    # 3) Estimates + fallbacks + caps + EPS sanity + Fade path + Plan + NEW dates
    out_rows = []
    sum_rows = []

    for _, r in df.iterrows():
        sym = r["_Ticker_"]
        ind = r["_Industry_"]

        try:
            latest_rev, latest_year, latest_eps = latest_actuals(sym)
            # Always fetch the next two estimate years for audit/date columns
            y1_sel, y2_sel = get_next_two_estimate_years_v3(sym, latest_year)
            sel_fy1_date = year_to_fye_date(y1_sel, FYE_MONTH_DAY)
            sel_fy2_date = year_to_fye_date(y2_sel, FYE_MONTH_DAY)

            # v3 date-aware (by year) short/mid (total-change vs latest actual)
            s_rev_da, m_rev_da, y1_sel, y2_sel = compute_short_mid_from_estimates_v3_dateaware(
                sym, latest_rev, latest_year
            )

            # EPS sanity via v3; also backup revenue path if needed (also total-change vs latest actual)
            s_rev_v3, m_rev_v3, s_eps, m_eps, flags = short_mid_from_estimates(
                sym, latest_rev, latest_eps, latest_year
            )

            # Prefer date-aware values; fallback to v3 if missing
            s_rev = s_rev_da if np.isfinite(s_rev_da) else s_rev_v3
            m_rev = m_rev_da if np.isfinite(m_rev_da) else m_rev_v3

            used_est_short = np.isfinite(s_rev)
            used_est_mid   = np.isfinite(m_rev)

            # Fallbacks if still missing
            if not used_est_short:
                s_rev = r["rev_cagr_5y"]
            if not used_est_mid:
                fallback = np.nanmax([r["rev_cagr_5y"], r["ind_rev_cagr_med"]])
                m_rev = fallback if np.isfinite(fallback) else s_rev

            # Cap mid-term vs short-term
            m_rev_capped = cap_midterm(s_rev, m_rev, MAX_MID_PREMIUM)

            # Fade rule (adjusted growth path)
            ind_med = to_real_float(r.get("ind_rev_cagr_med", np.nan))
            fade_dict = apply_fade_rule(s_rev, ind_med, GDP_RATE_US, cap=FADE_SHORT_CAP)

            # Build inverse-CAGR plan to a target future YEAR within horizon (v3 years)
            PlanY1 = PlanY2 = PlanY3 = PlanY4 = PlanY5 = np.nan
            PlanAnnualRate = np.nan
            PlanYears = np.nan
            PlanTargetDate = None

            if latest_year is not None and np.isfinite(latest_rev) and latest_rev > 0:
                est_list = get_analyst_estimates_v3(sym, limit=50) or []
                time.sleep(FMP_SLEEP_SEC)
                pairs = []
                for rr in est_list:
                    y = rr.get("year"); rev = rr.get("estimatedRevenueAvg")
                    if y is None or rev is None: continue
                    try:
                        y = int(y); rev = to_real_float(rev)
                    except: continue
                    if np.isfinite(rev) and y > latest_year:
                        pairs.append((y, rev))
                pairs.sort(key=lambda t: t[0])

                if pairs:
                    within = [(y, rev) for (y, rev) in pairs if (y - latest_year) <= TARGET_MAX_YEARS]
                    target = within[-1] if within else pairs[0]
                    target_year, target_rev = target
                    g_ann = annualized_rate_to_target_year(latest_rev, target_rev, latest_year, target_year)
                    yrs = max(1, int(target_year - latest_year))
                    plan = plan_yearly_growths(g_ann, yrs, max_years_out=5)
                    PlanY1, PlanY2, PlanY3, PlanY4, PlanY5 = plan
                    PlanAnnualRate = g_ann
                    PlanYears = yrs
                    PlanTargetDate = pd.to_datetime(f"{target_year}-{FYE_MONTH_DAY}").date()

            # NEW: fabricate dates from selected years (you’ll use these in Excel)
            sel_fy1_date = year_to_fye_date(y1_sel, FYE_MONTH_DAY)
            sel_fy2_date = year_to_fye_date(y2_sel, FYE_MONTH_DAY)

            notes = []
            if m_rev_capped != m_rev:
                notes.append("Mid REV capped to short+premium")
            if flags.get("eps_mid_used_yoy_due_to_sign"):
                notes.append("EPS mid sanity used YoY due to sign/flip")
            if fade_dict.get("FadeApplied", False):
                notes.append("Fade path Y1..Y5 applied")
            if not used_est_short or not used_est_mid:
                notes.append("Used fallback for missing estimates")

            out_rows.append({
                "Ticker": sym,
                "Industry Group": ind,

                # original outputs (unchanged)
                "ShortTermRevenueGrowth": s_rev,           # NOTE: total-change vs latest actual
                "MidTermRevenueGrowth": m_rev_capped,      # NOTE: total-change vs latest actual (capped)
                "TerminalGrowth": GDP_RATE_US,
                "ShortTermEPSGrowth_Estimate": s_eps,
                "MidTermEPSGrowth_Estimate": m_eps,

                # adjusted growth path (fade)
                "IndustryMedian_RevCAGR": ind_med,
                "AdjGrowthY1": fade_dict["AdjGrowthY1"],
                "AdjGrowthY2": fade_dict["AdjGrowthY2"],
                "AdjGrowthY3": fade_dict["AdjGrowthY3"],
                "AdjGrowthY4": fade_dict["AdjGrowthY4"],
                "AdjGrowthY5": fade_dict["AdjGrowthY5"],
                "AdjTerminalGrowth": fade_dict["AdjTerminalGrowth"],
                "FadeApplied": fade_dict["FadeApplied"],
                "FadeNote": fade_dict["FadeNote"],

                # plan to a target future year
                "PlanAnnualRate": PlanAnnualRate,
                "PlanYears": PlanYears,
                "PlanTargetDate": PlanTargetDate,
                "PlanY1": PlanY1,
                "PlanY2": PlanY2,
                "PlanY3": PlanY3,
                "PlanY4": PlanY4,
                "PlanY5": PlanY5,

                # NEW: explicit dates derived from v3 'year' for your Excel math
                "SelectedFY1Year": y1_sel,
                "SelectedFY2Year": y2_sel,
                "SelectedFY1Date": sel_fy1_date,  # e.g., "2029-09-28"
                "SelectedFY2Date": sel_fy2_date,  # e.g., "2030-09-28"
            })

            sum_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "UsedEstimates_ShortRev": bool(used_est_short),
                "UsedEstimates_MidRev": bool(used_est_mid),
                "AppliedWinsorization": WINSORIZE_INDUSTRY,
                "AppliedMidCapRule": (m_rev_capped != m_rev),
                "EPSMidUsedYoYDueToSign": flags.get("eps_mid_used_yoy_due_to_sign", False),
                "FadeApplied": fade_dict["FadeApplied"],
                "Notes": "; ".join(notes) if notes else ""
            })

        except Exception as e:
            print(f"[{sym}] error: {e}")
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
                "FadeApplied": False, "FadeNote": f"Error: {e}",
                "PlanAnnualRate": np.nan, "PlanYears": np.nan, "PlanTargetDate": None,
                "PlanY1": np.nan, "PlanY2": np.nan, "PlanY3": np.nan, "PlanY4": np.nan, "PlanY5": np.nan,
                "SelectedFY1Year": None, "SelectedFY2Year": None,
                "SelectedFY1Date": None, "SelectedFY2Date": None,
            })
            sum_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "UsedEstimates_ShortRev": False,
                "UsedEstimates_MidRev": False,
                "AppliedWinsorization": WINSORIZE_INDUSTRY,
                "AppliedMidCapRule": False,
                "EPSMidUsedYoYDueToSign": False,
                "FadeApplied": False,
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
    print("  - ShortTermRevenueGrowth (v3 total-change vs latest actual)")
    print("  - MidTermRevenueGrowth   (v3 total-change vs latest actual, capped)")
    print("  - TerminalGrowth")
    print("  - ShortTermEPSGrowth_Estimate  (sanity only)")
    print("  - MidTermEPSGrowth_Estimate    (sanity only)")
    print("  - IndustryMedian_RevCAGR")
    print("  - AdjGrowthY1..Y5 (fade) + AdjTerminalGrowth")
    print("  - PlanAnnualRate, PlanYears, PlanTargetDate, PlanY1..Y5")
    print("  - SelectedFY1Year/SelectedFY2Year + SelectedFY1Date/SelectedFY2Date")
    print(f"Audit written to '{SUMMARY_SHEET_NAME}'.")
    print("Reminder: You asked to do annualization/gap math in Excel; dates are provided for that.")

if __name__ == "__main__":
    main()
