"""
DCF_Lean_DeltaNWC_MonotoneGlide.py  (EBIT from EBITDA ratio; cap after Y5)

- Reads tickers/inputs from WACC workbook (sheet "WACC").
- Reads each ticker's local financials: <TICKER>_FMP_financials.xlsx in FIN_DIR.
- Sales YoY: lighter toning slabs, enforce non-increasing glide to terminal g by year N.
- EBIT margin path:
    • Start = latest EBITDA ratio (IS_Annual.ebitdaratio). If missing, fallback to trailing median EBIT margin.
    • Each year, margin is multiplied by a factor derived from Sales YoY bands (gentle).
    • Apply ±3 ppts step cap to yearly change.
    • Hard cap at 50% (configurable).
    • After Year 5, margin is held flat (no further rise) to avoid runaway.
- ΔNWC modeled as k * ΔSales (k from CF_Annual changeInWorkingCapital median).
- D&A% and Capex% from historical medians.
- Shares from Enterprise_Values.numberOfShares (latest), NetDebt from BS_Annual (netDebt or totalDebt - cash - ST inv).
- ClosePrice from FMP (if API key set) else Yahoo live.
- Reverse DCF (implied WACC) and WACC×g sensitivity per ticker.
- WACC policy switch: use sheet, implied, or average(sheet, implied).

Author: (you)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import time, requests

# =========================
# USER CONFIG
# =========================

WACC_XLSX_PATH = r"C:\Users\JMuk\Py Scripts Old\Swing\WACC\WACC.xlsx"
WACC_SHEET     = "WACC"

FIN_DIR = Path(r"C:\Users\JMuk\Py Scripts Old\Swing\FinancialStatement")
OUTPUT_XLSX = str(Path(FIN_DIR).parent / "DCF_Lean_DeltaNWC_MonotoneGlide.xlsx")

# Live price:
# - Set FMP_API_KEY to fetch from FinancialModelingPrep.
# - If blank, Yahoo Finance is used (no key required).
FMP_API_KEY = "BUy1zjTqw4dpREy1p96iqGvG4npO9qJg"  # e.g. "YOUR_FMP_KEY"

BASE_YEAR   = 2025
PROJ_YEARS  = 10
GDP_G_DEFAULT = 0.025
HIST_LOOKBACK_YEARS = 6
CAGR_POINTS = 6

# EBIT margin behavior (EBITDA-based start)
HARD_MARGIN_CAP = 0.50     # never exceed 50% EBIT margin
MAX_STEP_UP     = 0.03     # +3 ppts per year max up
MAX_STEP_DOWN   = 0.03     # -3 ppts per year max down
FREEZE_AFTER_YEARS = 5     # after year 5, hold margin flat

# ΔNWC modeling
MIN_DS_FOR_RATIO = 1e6

# HTTP
TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF = 2.0
USER_AGENT = "dcf-script/ebitda-start/1.0"

# WACC policy: "sheet" | "implied" | "average"
WACC_POLICY = "average"  # change to "average" to blend sheet & implied WACC

# =========================
# small helpers
# =========================

def is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def first_number(*vals) -> Optional[float]:
    for v in vals:
        try:
            f = float(v)
            if np.isfinite(f):
                return f
        except Exception:
            pass
    return None

def df_sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    dcol = None
    for c in ("date","filingDate","fillingDate","acceptedDate"):
        if c in df.columns: dcol = c; break
    if dcol is None: return df
    out = df.copy()
    out[dcol] = pd.to_datetime(out[dcol], errors="coerce")
    return out.sort_values(dcol, ascending=False).reset_index(drop=True)

def sheet_or_none(path: str, name:str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=name)
    except Exception:
        return None

def find_financial_file(symbol: str) -> Optional[str]:
    p = FIN_DIR / f"{symbol}_FMP_financials.xlsx"
    if p.exists(): return str(p)
    cand = list(FIN_DIR.glob(f"*{symbol}*FMP_financials*.xlsx"))
    return str(cand[0]) if cand else None

def read_financials(symbol: str) -> Dict[str, pd.DataFrame]:
    path = find_financial_file(symbol)
    if not path: return {}
    fin = {}
    for sh in ["IS_Annual","IS_Quarterly","BS_Annual","CF_Annual",
               "Company_Profile","Enterprise_Values","Key_Metrics",
               "Analyst_Estimates"]:
        fin[sh] = sheet_or_none(path, sh)
    return fin

# =========================
# WACC reader (percent-aware + optional overrides)
# =========================

def _parse_percent_series(s: pd.Series) -> pd.Series:
    """Accept '9.33%' or 9.33 or 0.0933 -> decimals (0.0933)."""
    s = s.astype(str).str.strip()
    is_pct = s.str.endswith("%")
    s_num = pd.to_numeric(s.str.rstrip("%"), errors="coerce")
    s_num = s_num.where(~is_pct, s_num / 100.0)
    s_num = s_num.where(~(s_num > 1.0), s_num / 100.0)
    return s_num

def _num_col_or_default(df: pd.DataFrame, colname: Optional[str], default: float, as_percent: bool=False) -> pd.Series:
    if colname and colname in df.columns:
        s = df[colname]
        s_num = _parse_percent_series(s) if as_percent else pd.to_numeric(s, errors="coerce")
        return s_num.fillna(default)
    else:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")

def _opt_num_col(df: pd.DataFrame, colname: str, as_percent: bool=False) -> pd.Series:
    if colname in df.columns:
        s = df[colname]
        return _parse_percent_series(s) if as_percent else pd.to_numeric(s, errors="coerce")
    else:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

def read_wacc_sheet(xlsx:str, sheet:str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    tk_col = next((c for c in df.columns if c.lower()=="ticker"), None)
    if tk_col is None: raise ValueError("Ticker column not found.")
    df["_Ticker_"] = df[tk_col].astype(str).str.upper().str.strip()

    ind_col = next((c for c in df.columns if c.lower() in ("industry group","industry","industry_name","industryname")), None)
    df["_Industry_"] = (df[ind_col].astype(str) if ind_col else pd.Series(["UNKNOWN"] * len(df), index=df.index))

    wacc_col = next((c for c in df.columns if c.lower() in ("wacc","cost of capital")), None)
    df["_Wacc_"] = _num_col_or_default(df, wacc_col, np.nan, as_percent=True)

    tg_col = next((c for c in df.columns if "terminal" in c.lower() and "g" in c.lower()), None)
    df["_Tg_"] = _num_col_or_default(df, tg_col, GDP_G_DEFAULT, as_percent=False)

    tax_col = next((c for c in df.columns if c.lower() in ("tax rate","taks rate","tax","taxrate")), None)
    df["_Tax_"] = _num_col_or_default(df, tax_col, 0.21, as_percent=True)

    # Optional overrides (kept for flexibility)
    df["_SharesOverride_"]  = _opt_num_col(df, "SharesOverride")
    df["_NetDebtOverride_"] = _opt_num_col(df, "NetDebtOverride")
    df["_CloseOverride_"]   = _opt_num_col(df, "ClosePrice")

    return df[["_Ticker_","_Industry_","_Wacc_","_Tg_","_Tax_",
               "_SharesOverride_","_NetDebtOverride_","_CloseOverride_"]]

# =========================
# Revenue building
# =========================

def latest_annual_rev_and_year(is_ann: pd.DataFrame) -> Tuple[Optional[float], Optional[int]]:
    if is_ann is None or is_ann.empty: return None, None
    d = df_sort_by_date(is_ann)
    rev = pd.to_numeric(d.get("revenue"), errors="coerce")
    dt  = pd.to_datetime(d.get("date"), errors="coerce")
    if rev is None or dt is None or rev.empty or dt.empty: return None, None
    return float(rev.iloc[0]), int(dt.iloc[0].year) if pd.notna(dt.iloc[0]) else None

def hist_rev_cagr(is_ann: pd.DataFrame, points:int=CAGR_POINTS) -> Optional[float]:
    if is_ann is None or is_ann.empty: return None
    d = df_sort_by_date(is_ann).head(points)
    rev = pd.to_numeric(d.get("revenue"), errors="coerce").dropna()
    if len(rev) < 2: return None
    series = rev[::-1].to_numpy()
    start, end = series[0], series[-1]
    if not (is_finite(start) and is_finite(end) and start > 0): return None
    years = len(series) - 1
    try:
        return float((end / start)**(1.0/years) - 1.0)
    except Exception:
        return None

def analyst_estimates_series(est_df: Optional[pd.DataFrame], base_year:int, latest_annual_year: Optional[int]) -> List[Tuple[int,float]]:
    if est_df is None or est_df.empty: return []
    df = est_df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["year"] = df["date"].dt.year
    if "estimatedRevenueAvg" not in df.columns:
        for alt in ("revenueAvg","estimatedRevenue","estimatedRevenueHigh","revenueHigh","revenueLow"):
            if alt in df.columns:
                df["estimatedRevenueAvg"] = pd.to_numeric(df[alt], errors="coerce")
                break
    df = df.dropna(subset=["date","estimatedRevenueAvg"])
    if df.empty: return []
    cutoff = latest_annual_year if latest_annual_year is not None else (base_year-1)
    fut = df[(df["year"] > cutoff) & (df["year"] <= base_year + PROJ_YEARS)].copy()
    if fut.empty: return []
    fut = fut.sort_values(["year","date"]).groupby("year").tail(1)
    fut = fut.sort_values("year")
    return [(int(y), float(v)) for y, v in zip(fut["year"], fut["estimatedRevenueAvg"])]

# Sales YoY toning slabs (keep fraction of growth), then glide down non-increasing to terminal
def tone_growth_slabs(g: float) -> float:
    if not is_finite(g) or g <= 0: return g
    if g > 0.50:            return g * 0.80
    if 0.40 < g <= 0.50:    return g * 0.75
    if 0.30 < g <= 0.40:    return g * 0.70
    if 0.20 < g <= 0.30:    return g * 0.65
    return g

def geometric_glide(start_g: float, end_g: float, steps: int) -> List[float]:
    if steps <= 1: return [start_g]
    if not (is_finite(start_g) and is_finite(end_g)): return [start_g] * steps
    if start_g <= end_g: return [start_g] * steps
    r = (end_g / start_g) ** (1.0 / (steps - 1))
    out = [start_g * (r ** i) for i in range(steps)]
    for i in range(1, steps):
        out[i] = min(out[i], out[i-1])
    return out

def revenue_path_with_estimates(base_sales: float, base_year:int,
                                est_series: List[Tuple[int,float]], terminal_g: float
                               ) -> Tuple[List[float], List[float]]:
    years_sorted = [y for (y, _) in est_series if y <= base_year + PROJ_YEARS]
    tgt = {y: v for (y, v) in est_series}
    yoy, sales = [], []
    prev_s = base_sales
    for y in years_sorted:
        est_s = tgt[y]
        g_imp = est_s / prev_s - 1.0 if (is_finite(est_s) and is_finite(prev_s) and prev_s > 0) else np.nan
        g_toned = tone_growth_slabs(g_imp) if is_finite(g_imp) else np.nan
        if not is_finite(g_toned): break
        if yoy: g_toned = min(g_toned, yoy[-1])
        yoy.append(g_toned)
        prev_s *= (1.0 + g_toned)
        sales.append(prev_s)
    remaining = PROJ_YEARS - len(yoy)
    if remaining > 0:
        last_g = yoy[-1] if yoy else GDP_G_DEFAULT
        tail = geometric_glide(last_g, terminal_g, remaining)
        yoy.extend(tail)
        s = sales[-1] if sales else base_sales
        for g in tail:
            s *= (1.0 + g)
            sales.append(s)
    return sales, yoy

def revenue_path_without_estimates(base_sales: float, is_ann: pd.DataFrame, terminal_g: float
                                  ) -> Tuple[List[float], List[float]]:
    g_hist = hist_rev_cagr(is_ann)
    g1 = g_hist if is_finite(g_hist) else terminal_g
    g1_toned = tone_growth_slabs(g1)
    yoy = geometric_glide(g1_toned, terminal_g, PROJ_YEARS)
    sales = []
    s = base_sales
    for g in yoy:
        s *= (1.0 + g)
        sales.append(s)
    return sales, yoy

# =========================
# EBIT margin from EBITDA ratio (your request)
# =========================

def latest_ebitda_ratio(is_ann: pd.DataFrame) -> Optional[float]:
    """Return latest IS_Annual.ebitdaratio as decimal (0.1193)."""
    if is_ann is None or is_ann.empty: return None
    d = df_sort_by_date(is_ann)
    col = None
    for c in ("ebitdaratio","ebitdaRatio","EBITDARatio","ebitda_ratio"):
        if c in d.columns: col = c; break
    if col is None: return None
    v = pd.to_numeric(d[col], errors="coerce").iloc[0]
    if not is_finite(v): return None
    # If someone stored as percentage (e.g., 11.93), convert to 0.1193
    return float(v/100.0) if v > 1.0 else float(v)

def trailing_median_ebit_margin(is_ann: pd.DataFrame, lookback:int=HIST_LOOKBACK_YEARS) -> Optional[float]:
    if is_ann is None or is_ann.empty: return None
    df = df_sort_by_date(is_ann).head(lookback).copy()
    if "ebit" not in df.columns and "operatingIncome" in df.columns:
        df["ebit"] = df["operatingIncome"]
    if "ebit" not in df.columns or "revenue" not in df.columns:
        return None
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    ebt = pd.to_numeric(df["ebit"], errors="coerce")
    m = (ebt / rev).replace([np.inf,-np.inf], np.nan).dropna()
    return float(np.median(m)) if not m.empty else None

def margin_multiplier_from_sales_g(gs: float) -> float:
    """Gentle multiplicative drift for margin based on Sales YoY."""
    if gs > 0.50:                  return 1.50   # +50%
    if 0.40 < gs <= 0.50:          return 1.35   # +35%
    if 0.30 < gs <= 0.40:          return 1.35   # +35%
    if 0.20 < gs <= 0.30:          return 1.25   # +25%
    if 0.10 <= gs <= 0.20:         return 1.10   # +10%
    if 0.00 <= gs < 0.10:          return 1.05   # +5%
    return 0.95                              # -5% on contraction

def ebit_margin_curve_from_ebitdaratio(
    is_ann: pd.DataFrame,
    sales_yoy: List[float],
    hard_cap: float = HARD_MARGIN_CAP,
    max_up: float = MAX_STEP_UP,
    max_down: float = MAX_STEP_DOWN,
    freeze_after_years: int = FREEZE_AFTER_YEARS,
) -> List[float]:
    """
    Start margin at latest EBITDA ratio; each year multiply by margin_multiplier_from_sales_g(gs),
    apply ± step caps, cap by hard_cap, and hold flat after 'freeze_after_years'.
    """
    m0 = latest_ebitda_ratio(is_ann)
    if not is_finite(m0):
        m0 = trailing_median_ebit_margin(is_ann) or 0.10
    m_prev = float(np.clip(m0, 0.0, hard_cap))

    margins = []
    for i, gs in enumerate(sales_yoy, start=1):
        if i > freeze_after_years:
            # hold flat after the freeze
            margins.append(m_prev)
            continue
        mult = margin_multiplier_from_sales_g(gs)
        m_des = m_prev * mult
        # step capped change (in absolute margin points)
        delta = float(np.clip(m_des - m_prev, -max_down, max_up))
        m_i = float(np.clip(m_prev + delta, 0.0, hard_cap))
        margins.append(m_i)
        m_prev = m_i
    return margins

# =========================
# Ratios: D&A, Capex, ΔNWC
# =========================

def hist_ratios(is_ann: pd.DataFrame, cf_ann: pd.DataFrame) -> Tuple[float,float,float]:
    da_ratio = 0.05
    capex_ratio = 0.05
    k_dsales = 0.0
    if is_ann is None or is_ann.empty: return da_ratio, capex_ratio, k_dsales

    isa = df_sort_by_date(is_ann).head(HIST_LOOKBACK_YEARS).copy()
    sales = pd.to_numeric(isa.get("revenue"), errors="coerce")

    # D&A
    if cf_ann is not None and not cf_ann.empty and "depreciationAndAmortization" in cf_ann.columns:
        cfa = df_sort_by_date(cf_ann).head(HIST_LOOKBACK_YEARS)
        da = pd.to_numeric(cfa.get("depreciationAndAmortization"), errors="coerce")
    else:
        da = pd.to_numeric(isa.get("depreciationAndAmortization"), errors="coerce")
    r = (da / sales).replace([np.inf,-np.inf], np.nan).dropna()
    if not r.empty: da_ratio = float(np.median(r.clip(lower=0.0)))

    # Capex
    if cf_ann is not None and not cf_ann.empty and "capitalExpenditure" in cf_ann.columns:
        cfa = df_sort_by_date(cf_ann).head(HIST_LOOKBACK_YEARS)
        capex = pd.to_numeric(cfa.get("capitalExpenditure"), errors="coerce")
        capex_pos = capex.apply(lambda v: abs(v) if is_finite(v) else np.nan)
        r = (capex_pos / sales).replace([np.inf,-np.inf], np.nan).dropna()
        if not r.empty: capex_ratio = float(np.median(r.clip(lower=0.0)))

    # k for ΔNWC
    if cf_ann is not None and not cf_ann.empty and "changeInWorkingCapital" in cf_ann.columns:
        cfa = df_sort_by_date(cf_ann).head(HIST_LOOKBACK_YEARS)
        dWC = pd.to_numeric(cfa.get("changeInWorkingCapital"), errors="coerce")
        dNWC = -dWC
        sales_series = sales.reset_index(drop=True)
        dNWC = dNWC.reset_index(drop=True)
        dS = sales_series.diff()
        mask = (dS.abs() >= MIN_DS_FOR_RATIO) & sales_series.notna() & dNWC.notna()
        k = (dNWC[mask] / dS[mask]).replace([np.inf,-np.inf], np.nan).dropna()
        if not k.empty:
            k = k.clip(-0.5, 0.5)
            k_dsales = float(np.median(k))

    da_ratio    = float(np.clip(da_ratio, 0.0, 0.25))
    capex_ratio = float(np.clip(capex_ratio, 0.0, 0.25))
    k_dsales    = float(np.clip(k_dsales, -0.5, 0.5))
    return da_ratio, capex_ratio, k_dsales

# =========================
# Shares / NetDebt / Close
# =========================

def shares_netdebt_from_files(fin: Dict[str,pd.DataFrame]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    shares = None
    netdebt = None
    price_file = None

    # Enterprise_Values: latest numberOfShares and (fallback) price
    ev = fin.get("Enterprise_Values")
    if ev is not None and not ev.empty:
        ev_sorted = df_sort_by_date(ev)
        for sc in ["numberOfShares","shares","Shares"]:
            if sc in ev_sorted.columns:
                s = first_number(ev_sorted[sc].iloc[0])
                if is_finite(s) and s > 0:
                    shares = s
                    break
        for pc in ["stockPrice","price","close","previousClose"]:
            if pc in ev_sorted.columns and not is_finite(price_file):
                price_file = first_number(ev_sorted[pc].iloc[0])

    # Company_Profile: fallback price only
    prof = fin.get("Company_Profile")
    if prof is not None and not prof.empty and not is_finite(price_file):
        for pc in ["price","lastPrice","close","previousClose"]:
            if pc in prof.columns:
                val = first_number(prof[pc].iloc[0])
                if is_finite(val) and val > 0:
                    price_file = val
                    break

    # BS_Annual: netDebt (preferred) else totalDebt - cash - ST inv
    bsa = fin.get("BS_Annual")
    if bsa is not None and not bsa.empty:
        bs = df_sort_by_date(bsa)
        if "netDebt" in bs.columns:
            nd = first_number(bs["netDebt"].iloc[0])
            if is_finite(nd):
                netdebt = nd
        else:
            td   = first_number(bs.get("totalDebt",              pd.Series([np.nan])).iloc[0])
            cash = first_number(bs.get("cashAndCashEquivalents", pd.Series([np.nan])).iloc[0])
            sti  = first_number(bs.get("shortTermInvestments",   pd.Series([0.0])).iloc[0])
            if is_finite(td) and (is_finite(cash) or is_finite(sti)):
                csum = (cash or 0.0) + (sti or 0.0)
                netdebt = td - csum

    shares     = shares if is_finite(shares) and shares > 0 else None
    netdebt    = netdebt if is_finite(netdebt) else None
    price_file = price_file if is_finite(price_file) and price_file > 0 else None
    return shares, netdebt, price_file

def get_live_close_price(symbol: str) -> Optional[float]:
    """Try FMP (if API key) then Yahoo Finance; return latest price."""
    headers = {"User-Agent": USER_AGENT}
    # FMP
    if FMP_API_KEY:
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
        params = {"apikey": FMP_API_KEY}
        for attempt in range(1, MAX_RETRIES+1):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
                if r.status_code in (429,500,502,503,504):
                    time.sleep(BACKOFF**attempt); continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and data:
                    row = data[0]
                    px = first_number(row.get("price"), row.get("previousClose"), row.get("close"))
                    if is_finite(px) and px > 0:
                        return px
                break
            except Exception:
                if attempt == MAX_RETRIES: break
                time.sleep(BACKOFF**attempt)
    # Yahoo
    try:
        yurl = "https://query1.finance.yahoo.com/v7/finance/quote"
        params = {"symbols": symbol}
        r = requests.get(yurl, params=params, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        res = j.get("quoteResponse", {}).get("result", [])
        if res:
            q = res[0]
            px = first_number(q.get("regularMarketPrice"), q.get("postMarketPrice"), q.get("regularMarketPreviousClose"))
            if is_finite(px) and px > 0:
                return px
    except Exception:
        pass
    return None

# =========================
# DCF & Reverse-DCF
# =========================

def discount_factor(wacc: float, t:int) -> float:
    return (1.0 / ((1.0 + wacc)**t)) if (is_finite(wacc) and wacc > 0) else 0.0

def gordon_terminal(fcff_last: float, wacc: float, g: float) -> float:
    if not (is_finite(fcff_last) and is_finite(wacc) and is_finite(g)): return 0.0
    if (wacc - g) <= 0: g = min(g, wacc - 1e-6)
    return fcff_last * (1.0 + g) / (wacc - g)

def ev_from_cashflows(fcff_list, wacc, g_terminal):
    if not (fcff_list and is_finite(wacc)):
        return np.nan
    N = len(fcff_list)
    pvs = []
    for t, fcff in enumerate(fcff_list, start=1):
        pvs.append(fcff / ((1.0 + wacc) ** t))
    tv = gordon_terminal(fcff_list[-1], wacc, g_terminal)
    pv_tv = tv / ((1.0 + wacc) ** N)
    return float(np.nansum(pvs) + pv_tv)

def reverse_wacc_for_price(target_ev, fcff_list, g, wacc_low=0.04, wacc_high=0.14, tol=1e-5, iters=60):
    if not (is_finite(target_ev) and target_ev>0 and fcff_list):
        return np.nan
    lo, hi = wacc_low, wacc_high
    ev_lo = ev_from_cashflows(fcff_list, lo, g)
    ev_hi = ev_from_cashflows(fcff_list, hi, g)
    if not (is_finite(ev_lo) and is_finite(ev_hi)):
        return np.nan
    if not (ev_lo >= target_ev >= ev_hi):
        if ev_hi >= target_ev >= ev_lo:
            lo, hi = hi, lo
        else:
            return np.nan
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        ev_mid = ev_from_cashflows(fcff_list, mid, g)
        if not is_finite(ev_mid): return np.nan
        if abs(ev_mid - target_ev) <= tol*target_ev:
            return mid
        if ev_mid > target_ev:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)

def write_wacc_tg_sensitivity(ws_writer, base_wacc, base_g, fcff_list, sheet_name="Sens_WACC_g"):
    waccs = [max(0.01, base_wacc + d) for d in (-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03)]
    gs    = [base_g + d for d in (-0.01,-0.005,0.00,0.005,0.01)]
    grid = []
    for g in gs:
        row = [ev_from_cashflows(fcff_list, w, g) for w in waccs]
        grid.append(row)
    df = pd.DataFrame(grid, index=[f"g={g:.3%}" for g in gs],
                             columns=[f"WACC={w:.3%}" for w in waccs])
    df.to_excel(ws_writer, sheet_name=sheet_name)

# =========================
# main
# =========================

def main():
    w = read_wacc_sheet(WACC_XLSX_PATH, WACC_SHEET)
    summary_rows = []
    peryear_rows = []

    for _, r in w.iterrows():
        sym = str(r["_Ticker_"]).upper()
        industry = str(r["_Industry_"])
        wacc_sheet = safe_float(r["_Wacc_"], np.nan)
        tg   = safe_float(r["_Tg_"], GDP_G_DEFAULT)
        tax  = safe_float(r["_Tax_"], 0.21)

        fin = read_financials(sym)
        if not fin:
            continue

        is_ann = fin.get("IS_Annual")
        cf_ann = fin.get("CF_Annual")
        est_df = fin.get("Analyst_Estimates")

        latest_rev, latest_year = latest_annual_rev_and_year(is_ann)
        if not is_finite(latest_rev):
            continue

        base_sales = float(latest_rev)
        base_year  = BASE_YEAR

        # --- Sales path ---
        est_series = analyst_estimates_series(est_df, base_year, latest_year)
        if est_series:
            sales_fwd, yoy_fwd = revenue_path_with_estimates(base_sales, base_year, est_series, tg)
        else:
            sales_fwd, yoy_fwd = revenue_path_without_estimates(base_sales, is_ann, tg)

        # --- EBIT margins from EBITDA ratio (bounded, step-capped, freeze after Y5) ---
        ebit_margins = ebit_margin_curve_from_ebitdaratio(is_ann, yoy_fwd)

        # Ratios
        da_ratio, capex_ratio, k_dsales = hist_ratios(is_ann, cf_ann)

        # Shares / NetDebt / Close
        sh_f, nd_f, price_file = shares_netdebt_from_files(fin)
        shares  = sh_f if is_finite(sh_f) else safe_float(r["_SharesOverride_"])
        netdebt = nd_f if is_finite(nd_f) else safe_float(r["_NetDebtOverride_"], 0.0)
        if not is_finite(netdebt): netdebt = 0.0

        px_live = get_live_close_price(sym)
        close_px = (
            px_live if is_finite(px_live) else
            safe_float(r["_CloseOverride_"]) if is_finite(r["_CloseOverride_"]) else
            price_file
        )
        close_px = close_px if is_finite(close_px) else np.nan

        # Per-year FCFF
        years = [base_year + i for i in range(1, PROJ_YEARS + 1)]
        s_prev = base_sales

        fcff_list, pv_list_sheet = [], []
        # For the initial EV (using sheet WACC), we’ll discount here; for implied/average we recompute later.
        for i, (yr, s, g, m) in enumerate(zip(years, sales_fwd, yoy_fwd, ebit_margins), start=1):
            ebit = s * m
            nopat = ebit * (1.0 - tax)
            da = s * da_ratio
            capex = s * capex_ratio
            ds = s - s_prev
            dNWC = k_dsales * ds

            fcff = nopat - capex - dNWC + da
            fcff_list.append(fcff)

            df_t = discount_factor(wacc_sheet, i)
            pv_t = fcff * df_t if is_finite(wacc_sheet) else np.nan

            peryear_rows.append({
                "Ticker": sym,
                "Year": yr,
                "Sales": s,
                "YoY": g,
                "EBIT_Margin": m,
                "EBIT": ebit,
                "NOPAT": nopat,
                "DandA": da,
                "Capex": capex,
                "DeltaNWC": dNWC,
                "FCFF": fcff,
                "DiscountFactor": df_t,
                "PV_FCFF": pv_t,
            })
            s_prev = s
            pv_list_sheet.append(pv_t)

        # Terminal, EV with sheet WACC
        tv_sheet = gordon_terminal(fcff_list[-1], wacc_sheet, tg) if is_finite(wacc_sheet) else np.nan
        dfN_sheet = discount_factor(wacc_sheet, PROJ_YEARS) if is_finite(wacc_sheet) else np.nan
        pv_tv_sheet = (tv_sheet * dfN_sheet) if (is_finite(tv_sheet) and is_finite(dfN_sheet)) else np.nan

        EV_sheet = (float(np.nansum(pv_list_sheet)) + pv_tv_sheet) if is_finite(pv_tv_sheet) else np.nan

        # Market EV & implied WACC
        market_ev = np.nan
        implied_wacc = np.nan
        if is_finite(close_px) and is_finite(shares):
            market_ev = close_px * shares + (netdebt if is_finite(netdebt) else 0.0)
            implied_wacc = reverse_wacc_for_price(market_ev, fcff_list, tg)

        # Decide WACC in use per WACC_POLICY
        chosen_wacc = wacc_sheet
        if WACC_POLICY.lower() == "implied" and is_finite(implied_wacc):
            chosen_wacc = implied_wacc
        elif WACC_POLICY.lower() == "average" and is_finite(implied_wacc) and is_finite(wacc_sheet):
            chosen_wacc = 0.5 * (wacc_sheet + implied_wacc)

        # EV/Equity/Per-share using chosen WACC
        EV = ev_from_cashflows(fcff_list, chosen_wacc, tg) if is_finite(chosen_wacc) else EV_sheet
        equity = EV - (netdebt if is_finite(netdebt) else 0.0)
        vps = equity / shares if (is_finite(shares) and shares > 0) else np.nan

        summary_rows.append({
            "Ticker": sym,
            "Industry Group": industry,
            "BaseYear": base_year,
            "Sales_Base": base_sales,
            "WACC_Sheet": wacc_sheet,
            "WACC_Implied": implied_wacc,
            "WACC_Used": chosen_wacc,
            "WACC_Policy": WACC_POLICY,
            "Terminal_g": tg,
            "Tax": tax,
            "DA_Ratio": da_ratio,
            "Capex_Ratio": capex_ratio,
            "k_DeltaSales_for_DeltaNWC": k_dsales,
            "Analyst_Used": bool(est_series),
            "EBIT_Margin_Y1": ebit_margins[0] if ebit_margins else np.nan,
            "EBIT_Margin_Y5": ebit_margins[min(4, len(ebit_margins)-1)] if ebit_margins else np.nan,
            "EBIT_Margin_Y10": ebit_margins[-1] if ebit_margins else np.nan,
            "Shares": shares,
            "NetDebt": netdebt,
            "ClosePrice": close_px,
            "EV_from_DCF": EV,
            "EquityValue": equity,
            "ValuePerShare": vps,
            "MarketEV_ifPrice": market_ev,
            "DCF_to_MarketEV": (EV / market_ev) if (is_finite(market_ev) and market_ev>0) else np.nan,
            "EV_SheetWACC": EV_sheet,
        })

        # attach terminal values onto last per-year row (for sheet-WACC PV context)
        peryear_rows[-1]["TerminalValue"] = tv_sheet
        peryear_rows[-1]["PV_TerminalValue"] = pv_tv_sheet

    # Build DataFrames & save
    summary_df = pd.DataFrame(summary_rows)
    peryear_df = pd.DataFrame(peryear_rows)

    sum_cols = [
        "Ticker","Industry Group","BaseYear","Sales_Base",
        "WACC_Sheet","WACC_Implied","WACC_Used","WACC_Policy",
        "Terminal_g","Tax",
        "DA_Ratio","Capex_Ratio","k_DeltaSales_for_DeltaNWC",
        "Analyst_Used","EBIT_Margin_Y1","EBIT_Margin_Y5","EBIT_Margin_Y10",
        "Shares","NetDebt","ClosePrice",
        "EV_from_DCF","EquityValue","ValuePerShare",
        "MarketEV_ifPrice","DCF_to_MarketEV",
        "EV_SheetWACC",
    ]
    for c in sum_cols:
        if c not in summary_df.columns:
            summary_df[c] = np.nan
    summary_df = summary_df[sum_cols]

    per_cols = [
        "Ticker","Year","Sales","YoY","EBIT_Margin","EBIT","NOPAT","DandA","Capex","DeltaNWC",
        "FCFF","DiscountFactor","PV_FCFF","TerminalValue","PV_TerminalValue"
    ]
    for c in per_cols:
        if c not in peryear_df.columns:
            peryear_df[c] = np.nan
    peryear_df = peryear_df[per_cols]

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        peryear_df.to_excel(w, sheet_name="PerYear", index=False)

        # Sensitivity around the chosen WACC
        for sym in summary_df["Ticker"].tolist():
            sub = peryear_df[peryear_df["Ticker"]==sym].sort_values("Year")
            fcffs = sub["FCFF"].tolist()
            row = summary_df[summary_df["Ticker"]==sym].iloc[0]
            wacc_used = float(row["WACC_Used"]) if is_finite(row["WACC_Used"]) else float(row["WACC_Sheet"])
            write_wacc_tg_sensitivity(w, wacc_used, float(row["Terminal_g"]), fcffs, sheet_name=f"Sens_{sym}")

    print(f"Done. Saved to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
