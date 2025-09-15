# dcf_lean_deltaNWC_option3.py  (updated: robust shares/price sourcing)
# -----------------------------------------------------------------------------
# Lean DCF engine that:
#  1) Projects EBIT margins with "Option 3": growth-linked + mean-reversion.
#  2) Uses ΔNWC from CF 'changeInWorkingCapital' (median theta).
#  3) Uses historical medians for Capex/Sales and D&A/Sales.
#  4) Reads tickers + WACC + Tax from WACC.xlsx.
#  5) Reads already-downloaded per-ticker workbooks from FinancialStatement folder:
#       {TICKER}_FMP_financials.xlsx (IS_Annual, CF_Annual, BS_Annual, Company_Profile,
#       Enterprise_Values, Key_Metrics if present).
#  6) Writes outputs to a NEW Excel with multiple worksheets, including per-share value.
# -----------------------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# =========================
# USER CONFIG
# =========================
WACC_XLSX_PATH = r"/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/WACC/WACC.xlsx"
WACC_SHEET     = "WACC"

STATEMENTS_DIR = Path("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/FinancialStatement/")
OUTPUT_XLSX    = Path("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/DCF/DCF_Lean_DeltaNWC_Option3.xlsx")

TERMINAL_G_DEFAULT = 0.025
PROJ_YEARS = 5

FIRST_YOY_RULES = [
    (0.50, 0.75),
    (0.35, 2/3),
    (0.15, 0.50),
]

# Option-3 EBIT margin params
BETA_UP   = 0.30
BETA_DOWN = 0.55
G_REF     = 0.05
GAMMA     = 0.18
STEP_CAP  = 0.02
M_BOUNDS  = (0.00, 0.35)

# Medians / clamps
DEFAULT_CAPEX_PCT = 0.04
DEFAULT_DA_PCT    = 0.04
DEFAULT_THETA_NWC = 0.08
THETA_CLAMP       = (-0.20, 0.40)
RATIO_CLAMP       = (0.0, 0.20)

# WACC/TAX column names
POSSIBLE_TAX_COLS  = ["Tax Rate", "Taks Rate", "Tax", "Effective Tax", "TaxRate"]
POSSIBLE_WACC_COLS = ["WACC", "Cost Of Capital", "CostOfCapital"]

# =========================
# Helpers
# =========================

def winsorize_series(s: pd.Series, p_low=0.05, p_high=0.95) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    v = s[np.isfinite(s)]
    if v.empty:
        return s
    lo = np.nanpercentile(v, p_low*100)
    hi = np.nanpercentile(v, p_high*100)
    return s.clip(lower=lo, upper=hi)

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    canon = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in canon:
            return canon[key]
    return None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def read_wacc_sheet(path: str, sheet: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    df  = pd.read_excel(path, sheet_name=sheet)
    # ticker
    tk_col = None
    for c in df.columns:
        if str(c).strip().lower() == "ticker":
            tk_col = c
            break
    if tk_col is None:
        raise ValueError("Ticker column not found in WACC sheet.")
    df["_Ticker_"] = df[tk_col].astype(str).str.upper().str.strip()
    # wacc
    wacc_col = find_col(df, POSSIBLE_WACC_COLS)
    if wacc_col is None:
        raise ValueError("WACC column not found in WACC sheet.")
    df["_WACC_"] = pd.to_numeric(df[wacc_col], errors="coerce")
    # tax
    tax_col = find_col(df, POSSIBLE_TAX_COLS)
    df["_Tax_"] = pd.to_numeric(df[tax_col], errors="coerce").fillna(0.21) if tax_col else 0.21
    # terminal g
    df["_Tg_"] = pd.to_numeric(df.get("Terminal_g", TERMINAL_G_DEFAULT), errors="coerce").fillna(TERMINAL_G_DEFAULT)
    return df

def load_financials_workbook(symbol: str) -> Dict[str, pd.DataFrame]:
    p = STATEMENTS_DIR / f"{symbol}_FMP_financials.xlsx"
    if not p.exists():
        return {}
    frames = {}
    with pd.ExcelFile(p) as x:
        for sheet in x.sheet_names:
            frames[sheet] = pd.read_excel(p, sheet_name=sheet)
    return frames

def get_last_annual_sales(is_df: pd.DataFrame) -> Tuple[Optional[int], float]:
    if is_df.empty:
        return None, np.nan
    df = is_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return None, np.nan
        last = df.iloc[-1]
        year = int(last["date"].year)
    else:
        year = None
        last = df.iloc[-1]
    sales_col = None
    for c in ["revenue", "sales", "totalRevenue"]:
        if c in df.columns:
            sales_col = c
            break
    if sales_col is None:
        return year, np.nan
    return year, float(last[sales_col])

def build_sales_growth_path(base_sales: float, first_yoy_raw: float, terminal_g: float,
                            years: int = PROJ_YEARS) -> Tuple[List[float], List[float]]:
    g1 = first_yoy_raw
    for thr, mult in FIRST_YOY_RULES:
        if g1 > thr:
            g1 = g1 * mult
            break
    start_g = max(g1, terminal_g)
    yoy = []
    if years == 1:
        yoy = [start_g]
    else:
        step = (terminal_g - start_g) / (years - 1)
        yoy = [start_g + i*step for i in range(years)]
        for i in range(1, years):
            yoy[i] = min(yoy[i-1], yoy[i])
    sales = []
    level = base_sales
    for g in yoy:
        level *= (1.0 + g)
        sales.append(level)
    return sales, yoy

def option3_margin_path(m0: float, m_lr: float, yoy: List[float],
                        beta_up=BETA_UP, beta_down=BETA_DOWN,
                        g_ref=G_REF, gamma=GAMMA, step_cap=STEP_CAP,
                        bounds=M_BOUNDS) -> Tuple[List[float], List[float], List[float]]:
    margins, ge_list, rev_list = [], [], []
    m_prev = float(m0)
    lo, hi = bounds
    for g in yoy:
        beta = beta_up if g >= g_ref else beta_down
        growth_effect = beta * (g - g_ref)
        reversion     = gamma * (m_lr - m_prev)
        delta_pre     = growth_effect + reversion
        delta         = clamp(delta_pre, -step_cap, step_cap)
        m_t           = clamp(m_prev + delta, lo, hi)
        margins.append(m_t)
        ge_list.append(growth_effect)
        rev_list.append(reversion)
        m_prev = m_t
    return margins, ge_list, rev_list

def median_ratio(series: Optional[pd.Series], denom: Optional[pd.Series],
                 clamp_to: Tuple[float,float], winsor=(0.05,0.95)) -> Optional[float]:
    if series is None or denom is None:
        return None
    a = pd.to_numeric(series, errors="coerce")
    b = pd.to_numeric(denom, errors="coerce")
    mask = (b.abs() > 1e-9) & a.notna() & b.notna()
    if mask.sum() < 2:
        return None
    r = (a[mask] / b[mask]).astype(float)
    r = winsorize_series(r, winsor[0], winsor[1])
    med = np.nanmedian(r.values)
    if np.isnan(med):
        return None
    return float(clamp(med, clamp_to[0], clamp_to[1]))

def median_theta_from_cf(cf_df: pd.DataFrame, is_df: pd.DataFrame) -> Optional[float]:
    if cf_df.empty or is_df.empty:
        return None
    cf = cf_df.copy()
    is_ = is_df.copy()
    for d in ["date", "fillingDate", "filingDate", "acceptedDate"]:
        if d in cf.columns:
            cf[d] = pd.to_datetime(cf[d], errors="coerce")
        if d in is_.columns:
            is_[d] = pd.to_datetime(is_[d], errors="coerce")
    if "date" in cf.columns:
        cf = cf.dropna(subset=["date"]).sort_values("date")
    if "date" in is_.columns:
        is_ = is_.dropna(subset=["date"]).sort_values("date")
    # sales col
    rev_col = None
    for c in ["revenue", "sales", "totalRevenue"]:
        if c in is_.columns:
            rev_col = c
            break
    if rev_col is None:
        return None
    is_["year"] = is_["date"].dt.year if "date" in is_.columns else None
    cf["year"]  = cf["date"].dt.year if "date" in cf.columns else None
    sales_by_year = is_.groupby("year", dropna=True)[rev_col].last().sort_index()
    d_sales = sales_by_year.diff()
    wc_col = None
    for c in ["changeInWorkingCapital", "changeinWorkingCapital", "netWorkingCapitalChange"]:
        if c in cf.columns:
            wc_col = c
            break
    if wc_col is None:
        return None
    wc_by_year = cf.groupby("year", dropna=True)[wc_col].sum().sort_index()
    idx = sorted(set(d_sales.index).intersection(set(wc_by_year.index)))
    pairs = []
    for y in idx:
        ds = d_sales.get(y, np.nan)
        wc_flow = wc_by_year.get(y, np.nan)
        if pd.isna(ds) or abs(ds) < 1e-6 or pd.isna(wc_flow):
            continue
        d_nwc = -float(wc_flow)  # ΔNWC ≈ - changeInWorkingCapital
        theta = d_nwc / float(ds)
        if np.isfinite(theta):
            pairs.append(theta)
    if len(pairs) < 2:
        return None
    s = winsorize_series(pd.Series(pairs), 0.05, 0.95)
    med = float(np.nanmedian(s.values))
    return clamp(med, THETA_CLAMP[0], THETA_CLAMP[1])

def derive_shares_price(wb: Dict[str, pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
    """Return (shares, price) using multiple fallbacks across sheets."""
    shares = None
    price  = None

    prof = wb.get("Company_Profile", pd.DataFrame())
    ev   = wb.get("Enterprise_Values", pd.DataFrame())
    km   = wb.get("Key_Metrics", pd.DataFrame())
    is_a = wb.get("IS_Annual", pd.DataFrame())
    bs_a = wb.get("BS_Annual", pd.DataFrame())

    # Try Enterprise_Values.numberOfShares
    if not ev.empty:
        if "numberOfShares" in ev.columns:
            s = pd.to_numeric(ev["numberOfShares"], errors="coerce").dropna()
            if not s.empty:
                shares = float(s.iloc[-1])

    # Key_Metrics.sharesOutstanding or weightedAverageShsOutDil
    if shares is None and not km.empty:
        for c in ["sharesOutstanding", "weightedAverageShsOutDil", "weightedAverageShsOut"]:
            if c in km.columns:
                s = pd.to_numeric(km[c], errors="coerce").dropna()
                if not s.empty:
                    shares = float(s.iloc[-1])
                    break

    # IS_Annual weighted averages
    if shares is None and not is_a.empty:
        for c in ["weightedAverageShsOutDil", "weightedAverageShsOut"]:
            if c in is_a.columns:
                s = pd.to_numeric(is_a[c], errors="coerce").dropna()
                if not s.empty:
                    shares = float(s.iloc[-1])
                    break

    # BS_Annual commonStockSharesOutstanding
    if shares is None and not bs_a.empty:
        for c in ["commonStockSharesOutstanding", "shareIssued", "commonSharesOutstanding"]:
            if c in bs_a.columns:
                s = pd.to_numeric(bs_a[c], errors="coerce").dropna()
                if not s.empty:
                    shares = float(s.iloc[-1])
                    break

    # Price from Profile or Key_Metrics
    if not prof.empty:
        for c in ["price", "lastClose", "close"]:
            if c in prof.columns:
                p = pd.to_numeric(prof[c], errors="coerce").dropna()
                if not p.empty:
                    price = float(p.iloc[0])
                    break
        # If we have market cap + price, compute shares
        if shares is None:
            for mc_col in ["mktCap", "marketCap"]:
                if mc_col in prof.columns:
                    mc = pd.to_numeric(prof[mc_col], errors="coerce").dropna()
                    if not mc.empty and price and price > 0:
                        shares = float(mc.iloc[0]) / price
                        break

    if price is None and not km.empty:
        for c in ["price", "close"]:
            if c in km.columns:
                p = pd.to_numeric(km[c], errors="coerce").dropna()
                if not p.empty:
                    price = float(p.iloc[-1])
                    break
        if shares is None:
            for mc_col in ["marketCap", "mktCap"]:
                if mc_col in km.columns:
                    mc = pd.to_numeric(km[mc_col], errors="coerce").dropna()
                    if not mc.empty and price and price > 0:
                        shares = float(mc.iloc[-1]) / price
                        break

    return shares, price

def derive_net_debt(wb: Dict[str, pd.DataFrame]) -> Optional[float]:
    """NetDebt from Profile if present; else BS_Annual: debt - cash."""
    prof = wb.get("Company_Profile", pd.DataFrame())
    if not prof.empty:
        for c in ["netDebt", "netDebtPenultimate"]:
            if c in prof.columns:
                v = pd.to_numeric(prof[c], errors="coerce").dropna()
                if not v.empty:
                    return float(v.iloc[0])
    bs_a = wb.get("BS_Annual", pd.DataFrame())
    if not bs_a.empty:
        debt = None
        cash = None
        for c in ["totalDebt", "netDebt", "shortLongTermDebtTotal", "longTermDebt", "shortTermDebt"]:
            if c in bs_a.columns:
                v = pd.to_numeric(bs_a[c], errors="coerce").dropna()
                if not v.empty:
                    debt = float(v.iloc[-1]); break
        for c in ["cashAndShortTermInvestments", "cashAndCashEquivalents", "cashAndCashEquivalentsIncludingRestrictedCash"]:
            if c in bs_a.columns:
                v = pd.to_numeric(bs_a[c], errors="coerce").dropna()
                if not v.empty:
                    cash = float(v.iloc[-1]); break
        if debt is not None and cash is not None:
            return debt - cash
    return None

# =========================
# Core compute
# =========================

def compute_dcf_for_ticker(sym: str, wacc: float, tax_rate: float, terminal_g: float,
                           params: Dict[str, Any]) -> Dict[str, Any]:
    wb = load_financials_workbook(sym)
    if not wb:
        return {"error": f"Workbook not found for {sym}"}

    is_ann = wb.get("IS_Annual", pd.DataFrame())
    cf_ann = wb.get("CF_Annual", pd.DataFrame())
    bs_ann = wb.get("BS_Annual", pd.DataFrame())

    base_year, sales_base = get_last_annual_sales(is_ann)
    if base_year is None or not np.isfinite(sales_base):
        return {"error": f"Missing sales base for {sym}"}

    # Capex% and D&A%
    capex_col = None
    for c in ["capitalExpenditure", "capitalExpenditures"]:
        if c in cf_ann.columns:
            capex_col = c
            break
    capex_pct_med = median_ratio(cf_ann.get(capex_col), is_ann.get("revenue"), RATIO_CLAMP) if capex_col else None
    if capex_pct_med is None: capex_pct_med = DEFAULT_CAPEX_PCT

    da_col = None; da_base = None
    for c in ["depreciationAndAmortization", "depreciation", "amortization"]:
        if c in cf_ann.columns:
            da_col = c; da_base = cf_ann[c]; break
    if da_col is None:
        for c in ["depreciationAndAmortization", "depreciation", "amortization"]:
            if c in is_ann.columns:
                da_col = c; da_base = is_ann[c]; break
    da_pct_med = median_ratio(da_base if da_col else None, is_ann.get("revenue"), RATIO_CLAMP)
    if da_pct_med is None: da_pct_med = DEFAULT_DA_PCT

    # theta (ΔNWC/ΔSales)
    theta_med = median_theta_from_cf(cf_ann, is_ann)
    if theta_med is None: theta_med = DEFAULT_THETA_NWC

    # first YoY from trailing revenue CAGR (<=3-4 points)
    is_sorted = is_ann.copy()
    rev_col = None
    if "date" in is_sorted.columns:
        is_sorted["date"] = pd.to_datetime(is_sorted["date"], errors="coerce")
        is_sorted = is_sorted.dropna(subset=["date"]).sort_values("date")
    for c in ["revenue", "sales", "totalRevenue"]:
        if c in is_sorted.columns:
            rev_col = c; break
    first_yoy_raw = 0.06
    if rev_col:
        vals = is_sorted[rev_col].dropna().astype(float).tail(4).values
        if len(vals) >= 2:
            old, new = vals[0], vals[-1]
            n = len(vals)-1
            if old > 0 and n > 0:
                first_yoy_raw = (new/old)**(1.0/n) - 1.0

    sales_list, yoy_list = build_sales_growth_path(sales_base, first_yoy_raw, terminal_g, years=PROJ_YEARS)

    # EBIT margin start & LR
    ebit_col = None
    for c in ["ebit", "operatingIncome", "operatingIncomeLoss", "operatingincome"]:
        if c in is_ann.columns:
            ebit_col = c; break
    if ebit_col and rev_col:
        tmp = is_sorted[[rev_col, ebit_col]].dropna()
        tmp = tmp[tmp[rev_col].abs() > 1e-9]
        if not tmp.empty:
            margins_hist = (tmp[ebit_col].astype(float) / tmp[rev_col].astype(float)).clip(-1, 1)
            m0 = float(np.nanmedian(winsorize_series(margins_hist, 0.05, 0.95)))
        else:
            m0 = 0.15
    else:
        m0 = 0.15
    m_lr = clamp(m0, M_BOUNDS[0], M_BOUNDS[1])

    margins, geff, reff = option3_margin_path(
        m0=m0, m_lr=m_lr, yoy=yoy_list,
        beta_up=BETA_UP, beta_down=BETA_DOWN,
        g_ref=G_REF, gamma=GAMMA, step_cap=STEP_CAP, bounds=M_BOUNDS
    )

    # Per-year projection
    years = list(range(base_year+1, base_year+1+PROJ_YEARS))
    rows = []
    last_fcff = None
    pv_fcffs = []
    for i, y in enumerate(years):
        sales = sales_list[i]; yoy = yoy_list[i]; m = margins[i]
        ebit = sales * m
        tax_cash = tax_rate * ebit if ebit > 0 else 0.0
        nopat = ebit - tax_cash
        da    = da_pct_med * sales
        capex = capex_pct_med * sales
        dsales = sales - (sales_base if i == 0 else sales_list[i-1])
        delta_nwc = theta_med * dsales
        fcff = nopat + da - capex - delta_nwc

        t = i+1
        df = (1.0 + float(wacc)) ** t
        pv_fcff = fcff / df

        rows.append({
            "Year": y, "Sales": sales, "YoY": yoy,
            "EBIT_Margin": m, "EBIT": ebit, "Taxes_Cash": tax_cash,
            "NOPAT": nopat, "DandA": da, "Capex": capex, "DeltaNWC": delta_nwc,
            "FCFF": fcff, "DiscountFactor": 1.0/df, "PV_FCFF": pv_fcff
        })
        pv_fcffs.append(pv_fcff)
        last_fcff = fcff

    per_year_df = pd.DataFrame(rows)

    # Terminal
    g = float(terminal_g); w = float(wacc)
    fcff_next = last_fcff * (1.0 + g) if last_fcff is not None else np.nan
    tv = fcff_next / max(1e-9, (w - g)) if (w - g) > 0 and np.isfinite(fcff_next) else np.nan
    df_N = (1.0 + w)**PROJ_YEARS
    pv_tv = tv / df_N if np.isfinite(tv) else np.nan

    ev = float(per_year_df["PV_FCFF"].sum()) + (pv_tv if np.isfinite(pv_tv) else 0.0)

    # Robust shares/price & net debt
    shares, close_px = derive_shares_price(wb)
    net_debt = derive_net_debt(wb)
    if net_debt is None: net_debt = 0.0

    equity = ev - float(net_debt) if np.isfinite(ev) else np.nan
    per_share = (equity / shares) if (shares and shares > 0 and np.isfinite(equity)) else np.nan

    summary = {
        "Ticker": sym,
        "BaseYear": base_year,
        "Sales_Base": sales_base,
        "EBIT_Margin_Start": m0,
        "EBIT_Margin_LRTarget": m_lr,
        "Tax": tax_rate,
        "CapexPct_Median": capex_pct_med,
        "DA_Pct_Median": da_pct_med,
        "ThetaNWC_Median": theta_med,
        "WACC": wacc,
        "Terminal_g": terminal_g,
        "EV_from_DCF": ev,
        "NetDebt": net_debt,
        "Shares": shares,
        "EquityValue": equity,
        "ValuePerShare": per_share,
        "ClosePrice": close_px
    }

    ebit_path_df = pd.DataFrame({
        "Ticker": sym, "Year": years, "Sales": sales_list, "YoY": yoy_list,
        "EBIT_Margin": margins, "ΔM_GrowthEffect": geff, "ΔM_Reversion": reff
    })

    terminal_df = pd.DataFrame([{
        "Ticker": sym, "FCFF_Last": last_fcff, "FCFF_Next": fcff_next,
        "TerminalValue": tv, "PV_TerminalValue": pv_tv
    }])

    ratios_df = pd.DataFrame([{
        "Ticker": sym,
        "CapexPct_Median": capex_pct_med,
        "DA_Pct_Median": da_pct_med,
        "ThetaNWC_Median": theta_med,
        "Theta_Clamp": str(THETA_CLAMP),
        "Capex_DA_Clamp": str(RATIO_CLAMP),
        "BETA_UP": BETA_UP, "BETA_DOWN": BETA_DOWN,
        "G_REF": G_REF, "GAMMA": GAMMA, "STEP_CAP": STEP_CAP,
        "M_BOUNDS": str(M_BOUNDS),
    }])

    return {
        "summary": pd.DataFrame([summary]),
        "per_year": per_year_df.assign(Ticker=sym),
        "ebit_margin_path": ebit_path_df,
        "ratios": ratios_df,
        "terminal": terminal_df,
    }

# =========================
# Batch
# =========================

def main():
    os.makedirs(OUTPUT_XLSX.parent, exist_ok=True)
    wacc_df = read_wacc_sheet(WACC_XLSX_PATH, WACC_SHEET)
    tickers = wacc_df["_Ticker_"].dropna().astype(str).str.upper().str.strip().tolist()

    summaries = []; per_year_all = []; ebit_paths = []; ratios_all = []; terminals = []; errors = []

    for sym in tickers:
        row = wacc_df.loc[wacc_df["_Ticker_"] == sym].iloc[0]
        wacc = float(row["_WACC_"]) if pd.notna(row["_WACC_"]) else 0.10
        tax  = float(row["_Tax_"])  if pd.notna(row["_Tax_"])  else 0.21
        tg   = float(row["_Tg_"])   if pd.notna(row["_Tg_"])   else TERMINAL_G_DEFAULT

        try:
            res = compute_dcf_for_ticker(sym, wacc, tax, tg, params={})
            if "error" in res:
                errors.append({"Ticker": sym, "Error": res["error"]})
                continue
            summaries.append(res["summary"])
            per_year_all.append(res["per_year"])
            ebit_paths.append(res["ebit_margin_path"])
            ratios_all.append(res["ratios"])
            terminals.append(res["terminal"])
        except Exception as e:
            errors.append({"Ticker": sym, "Error": str(e)})

    summary_df  = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    per_year_df = pd.concat(per_year_all, ignore_index=True) if per_year_all else pd.DataFrame()
    ebit_path_df= pd.concat(ebit_paths, ignore_index=True) if ebit_paths else pd.DataFrame()
    ratios_df   = pd.concat(ratios_all, ignore_index=True) if ratios_all else pd.DataFrame()
    terminal_df = pd.concat(terminals, ignore_index=True) if terminals else pd.DataFrame()
    errors_df   = pd.DataFrame(errors)

    config_used = pd.DataFrame([{
        "FIRST_YOY_RULES": str(FIRST_YOY_RULES),
        "BETA_UP": BETA_UP, "BETA_DOWN": BETA_DOWN,
        "G_REF": G_REF, "GAMMA": GAMMA, "STEP_CAP": STEP_CAP, "M_BOUNDS": str(M_BOUNDS),
        "DEFAULT_CAPEX_PCT": DEFAULT_CAPEX_PCT, "DEFAULT_DA_PCT": DEFAULT_DA_PCT,
        "DEFAULT_THETA_NWC": DEFAULT_THETA_NWC, "THETA_CLAMP": str(THETA_CLAMP),
        "RATIO_CLAMP": str(RATIO_CLAMP), "PROJ_YEARS": PROJ_YEARS,
        "TERMINAL_G_DEFAULT": TERMINAL_G_DEFAULT, "WACC_SHEET": WACC_SHEET,
        "STATEMENTS_DIR": str(STATEMENTS_DIR),
    }])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        per_year_df.to_excel(w, sheet_name="PerYear", index=False)
        ebit_path_df.to_excel(w, sheet_name="EBIT_Margin_Path", index=False)
        ratios_df.to_excel(w, sheet_name="Ratios_Diagnostics", index=False)
        terminal_df.to_excel(w, sheet_name="Terminal", index=False)
        if not errors_df.empty:
            errors_df.to_excel(w, sheet_name="Errors", index=False)
        config_used.to_excel(w, sheet_name="Config_Used", index=False)

    print(f"Done. Wrote: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
