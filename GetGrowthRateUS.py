# growth_builder_from_fmp.py
# Purpose: For each ticker in your WACC workbook, compute short-, mid-, and terminal revenue growth
#          using FMP analyst estimates as the driver, with robust fallbacks and audit trail.

import time
import requests
import numpy as np
import pandas as pd

# =========================
# USER CONFIG (REQUIRED)
# =========================
WACC_XLSX_PATH = r"C:\Users\JMuk\Py Scripts Old\Swing\WACC\WACC.xlsx"
WACC_SHEET     = "By broad region"

FMP_API_KEY    = "BUy1zjTqw4dpREy1p96iqGvG4npO9qJg"  # <-- REQUIRED
GDP_RATE_US    = 0.025  # <-- REQUIRED (e.g., 0.025 for 2.5% nominal long-run)

# Rule: mid-term cannot exceed short-term by more than this premium
MAX_MID_PREMIUM = 0.03

# Optional: tame outliers when computing industry medians of historical CAGRs
WINSORIZE_INDUSTRY = True
WINSOR_Q_LOW  = 0.05
WINSOR_Q_HIGH = 0.95

# Polite API pacing
FMP_SLEEP_SEC = 0.25

SUMMARY_SHEET_NAME = "Growth_Summary"

# =========================
# FMP HELPERS
# =========================
def fmp_get(url, params=None):
    params = {} if params is None else dict(params)
    params["apikey"] = FMP_API_KEY
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_income_statement(symbol, limit=10):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}"
    return fmp_get(url, {"period": "annual", "limit": limit})

def get_analyst_estimates(symbol, limit=8):
    url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
    return fmp_get(url, {"limit": limit})

def get_financial_growth(symbol, limit=8):
    url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}"
    return fmp_get(url, {"period": "annual", "limit": limit})

# =========================
# CALC HELPERS
# =========================
def safe_float(x):
    try: return float(x)
    except: return np.nan

def cagr(series_old_to_new):
    arr = [safe_float(x) for x in series_old_to_new if pd.notna(x)]
    if len(arr) < 2: return np.nan
    start, end = arr[0], arr[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0: return np.nan
    years = len(arr) - 1
    try: return (end / start) ** (1/years) - 1.0
    except: return np.nan

def pct_growth(new, base):
    if not np.isfinite(new) or not np.isfinite(base) or base == 0: return np.nan
    return (new / base) - 1.0

def winsorize_series(s, q_low=0.05, q_high=0.95):
    s = s.copy()
    if s.dropna().empty: return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)

def cap_midterm(short_g, mid_g, cap=0.03):
    if not np.isfinite(short_g) or not np.isfinite(mid_g): return mid_g
    return min(mid_g, short_g + cap)

# =========================
# DATA EXTRACTORS
# =========================
def latest_actuals(symbol):
    """
    Returns: latest_annual_revenue, latest_eps (if available from financial-growth)
    """
    inc = get_income_statement(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    latest_rev = safe_float(inc[0].get("revenue")) if inc else np.nan

    fg = get_financial_growth(symbol, limit=2) or []
    time.sleep(FMP_SLEEP_SEC)
    latest_eps = safe_float(fg[0].get("eps")) if (fg and fg[0].get("eps") is not None) else np.nan
    return latest_rev, latest_eps

def hist_5y_rev_cagr(symbol):
    inc = get_income_statement(symbol, limit=8) or []
    time.sleep(FMP_SLEEP_SEC)
    # Build revenue series old->new
    rev_series = [safe_float(row.get("revenue")) for row in reversed(inc)]
    # 5-year CAGR over last 6 points if possible
    return cagr(rev_series[-6:])

def short_mid_from_estimates(symbol, latest_rev, latest_eps):
    """
    From analyst estimates compute:
      short_rev = growth to next FY revenue estimate
      mid_rev   = 2y CAGR from latest revenue to FY+2 estimate
      short_eps / mid_eps = for sanity only (with sign handling; mid=YoY y2/y1 if sign issues)
    Returns: (short_rev, mid_rev, short_eps, mid_eps, flags)
    """
    flags = {
        "used_est_short_rev": False, "used_est_mid_rev": False,
        "used_est_short_eps": False, "used_est_mid_eps": False,
        "eps_mid_used_yoy_due_to_sign": False
    }

    est = get_analyst_estimates(symbol, limit=8) or []
    time.sleep(FMP_SLEEP_SEC)
    if not est: return np.nan, np.nan, np.nan, np.nan, flags

    by_year = {}
    for r in est:
        y = r.get("year")
        if y is None: continue
        by_year[int(y)] = {
            "rev": safe_float(r.get("estimatedRevenueAvg")),
            "eps": safe_float(r.get("estimatedEpsAvg"))
        }
    if not by_year: return np.nan, np.nan, np.nan, np.nan, flags

    yrs = sorted(by_year.keys())
    y1, y2 = (yrs[0], yrs[1]) if len(yrs) >= 2 else (yrs[0], None)

    short_rev = np.nan
    mid_rev   = np.nan
    short_eps = np.nan
    mid_eps   = np.nan

    if y1:
        r1 = by_year[y1]["rev"]
        e1 = by_year[y1]["eps"]
        if np.isfinite(r1) and np.isfinite(latest_rev) and latest_rev > 0:
            short_rev = pct_growth(r1, latest_rev)
            flags["used_est_short_rev"] = True
        if np.isfinite(e1) and np.isfinite(latest_eps) and latest_eps != 0:
            short_eps = pct_growth(e1, latest_eps)
            flags["used_est_short_eps"] = True

    if y2:
        r2 = by_year[y2]["rev"]
        e2 = by_year[y2]["eps"]

        if np.isfinite(r2) and np.isfinite(latest_rev) and latest_rev > 0:
            try:
                mid_rev = (r2 / latest_rev) ** (1/2) - 1.0
                flags["used_est_mid_rev"] = True
            except:
                mid_rev = np.nan

        # EPS mid-term sanity (not driver)
        sign_issue = True
        if np.isfinite(latest_eps) and np.isfinite(by_year[y1]["eps"]) and np.isfinite(e2):
            if latest_eps > 0 and by_year[y1]["eps"] > 0 and e2 > 0:
                sign_issue = False
        if not sign_issue:
            try:
                mid_eps = (e2 / latest_eps) ** (1/2) - 1.0
                flags["used_est_mid_eps"] = True
            except:
                sign_issue = True
        if sign_issue and np.isfinite(by_year[y1]["eps"]) and by_year[y1]["eps"] != 0 and np.isfinite(e2):
            mid_eps = pct_growth(e2, by_year[y1]["eps"])
            flags["used_est_mid_eps"] = True
            flags["eps_mid_used_yoy_due_to_sign"] = True

    return short_rev, mid_rev, short_eps, mid_eps, flags

# =========================
# MAIN
# =========================
def main():
    if not FMP_API_KEY or FMP_API_KEY == "PUT_YOUR_FMP_API_KEY_HERE":
        raise ValueError("FMP_API_KEY is required.")
    if GDP_RATE_US is None:
        raise ValueError("GDP_RATE_US (terminal growth) is required (e.g., 0.025).")

    # Load WACC workbook
    xls = pd.ExcelFile(WACC_XLSX_PATH)
    if WACC_SHEET not in xls.sheet_names:
        raise ValueError(f"Sheet '{WACC_SHEET}' not found in {WACC_XLSX_PATH}")
    base = pd.read_excel(WACC_XLSX_PATH, sheet_name=WACC_SHEET)

    # Identify columns
    tk_col = next((c for c in base.columns if str(c).strip().lower() == "ticker"), None)
    ind_col = next((c for c in base.columns if str(c).strip().lower() in
                    ("industry group","industry","industry_name","industryname")), None)
    if tk_col is None: raise ValueError("Ticker column not found.")
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
        try:
            rev_cagr_5y = hist_5y_rev_cagr(sym)
            hist_rows.append({"_Ticker_": sym, "rev_cagr_5y": rev_cagr_5y})
        except Exception as e:
            print(f"[{sym}] hist error: {e}")
            hist_rows.append({"_Ticker_": sym, "rev_cagr_5y": np.nan})
    hist_df = pd.DataFrame(hist_rows)
    df = base.merge(hist_df, on="_Ticker_", how="left")

    # 2) Industry medians (winsorized if chosen)
    grouped = df.groupby("_Industry_", dropna=False)
    ind_rows = []
    for gname, g in grouped:
        s = g["rev_cagr_5y"]
        s_use = winsorize_series(s, WINSOR_Q_LOW, WINSOR_Q_HIGH) if WINSORIZE_INDUSTRY else s
        ind_rows.append({"_Industry_": gname, "ind_rev_cagr_med": s_use.median(skipna=True)})
    ind_df = pd.DataFrame(ind_rows)
    df = df.merge(ind_df, on="_Industry_", how="left")

    # 3) Estimates + fallbacks + caps + EPS sanity
    out_rows = []
    sum_rows = []
    for _, r in df.iterrows():
        sym = r["_Ticker_"]
        ind = r["_Industry_"]
        try:
            latest_rev, latest_eps = latest_actuals(sym)
            s_rev, m_rev, s_eps, m_eps, flags = short_mid_from_estimates(sym, latest_rev, latest_eps)

            used_est_short = flags["used_est_short_rev"]
            used_est_mid   = flags["used_est_mid_rev"]

            # Fallbacks
            if not np.isfinite(s_rev):
                s_rev = r["rev_cagr_5y"]
            if not np.isfinite(m_rev):
                fallback = np.nanmax([r["rev_cagr_5y"], r["ind_rev_cagr_med"]])
                m_rev = fallback if np.isfinite(fallback) else s_rev

            # Cap mid-term vs short-term
            m_rev_capped = cap_midterm(s_rev, m_rev, MAX_MID_PREMIUM)

            notes = []
            if m_rev_capped != m_rev:
                notes.append("Mid REV capped to short+premium")
            if flags.get("eps_mid_used_yoy_due_to_sign"):
                notes.append("EPS mid sanity used YoY due to sign/flip")

            out_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "ShortTermRevenueGrowth": s_rev,
                "MidTermRevenueGrowth": m_rev_capped,
                "TerminalGrowth": GDP_RATE_US,
                # EPS sanity only (not used as driver)
                "ShortTermEPSGrowth_Estimate": s_eps,
                "MidTermEPSGrowth_Estimate": m_eps
            })

            sum_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "UsedEstimates_ShortRev": used_est_short,
                "UsedEstimates_MidRev": used_est_mid,
                "AppliedWinsorization": WINSORIZE_INDUSTRY,
                "AppliedMidCapRule": (m_rev_capped != m_rev),
                "EPSMidUsedYoYDueToSign": flags.get("eps_mid_used_yoy_due_to_sign", False),
                "Notes": "; ".join(notes)
            })

        except Exception as e:
            print(f"[{sym}] estimate error: {e}")
            out_rows.append({
                "Ticker": sym,
                "Industry Group": ind,
                "ShortTermRevenueGrowth": np.nan,
                "MidTermRevenueGrowth": np.nan,
                "TerminalGrowth": GDP_RATE_US,
                "ShortTermEPSGrowth_Estimate": np.nan,
                "MidTermEPSGrowth_Estimate": np.nan
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
    print("  - ShortTermRevenueGrowth")
    print("  - MidTermRevenueGrowth")
    print("  - TerminalGrowth")
    print("  - ShortTermEPSGrowth_Estimate  (sanity only)")
    print("  - MidTermEPSGrowth_Estimate    (sanity only)")
    print(f"Audit written to '{SUMMARY_SHEET_NAME}'.")
    print("Remember: these growths are for DCF driver (revenue). EPS fields are just a cross-check.")

if __name__ == "__main__":
    main()
