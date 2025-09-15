#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DCF_Lean_DeltaNWC_MonotoneGlide.py

- Reads tickers/inputs from WACC workbook.
- Reads each ticker's local financials: <TICKER>_FMP_financials.xlsx in FIN_DIR.
- Revenue YoY: monotonic glide (anchor-aware if Analyst_Estimates exists), with first-year tone rules and exact
  normalization to match the analyst anchor at its year.
- EBIT margin: monotonic glide from TTM to trailing-annual median (not constant), 4ppt/year cap.
- ΔNWC modeled as k*ΔSales using CF_Annual changeInWorkingCapital.
- D&A% and Capex% from historical medians.
- EV, Equity, Value per Share; includes Industry Group, NetDebt, Shares, ClosePrice if available.

Author: (you)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import time, requests, math

# =========================
# USER CONFIG
# =========================

WACC_XLSX_PATH = r"/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/WACC/WACC.xlsx"
WACC_SHEET     = "WACC"

FIN_DIR = Path("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/FinancialStatement/")
OUTPUT_XLSX = str(Path(FIN_DIR).parent / "DCF_Lean_DeltaNWC_MonotoneGlide.xlsx")

FMP_API_KEY = ""   # optional (for live last close), leave "" to skip

BASE_YEAR   = 2025
PROJ_YEARS  = 5
GDP_G_DEFAULT = 0.025
HIST_LOOKBACK_YEARS = 6
CAGR_POINTS = 6

# EBIT margin clamps and glide settings
M_MIN, M_MAX = 0.00, 0.75
W_TTM = 0.70
MAX_MARGIN_STEP = 0.04     # 4ppts/yr max change
EPSILON_NUDGE   = 0.005    # tiny nudge to avoid perfectly flat paths

# ΔNWC modeling
MIN_DS_FOR_RATIO = 1e6

TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF = 2.0

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
        f = float(x)
        return f
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

def read_wacc_sheet(xlsx: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # --- helpers ---
    def find_col(candidates) -> Optional[str]:
        cands = [c.lower() for c in candidates]
        for col in df.columns:
            if col.strip().lower() in cands:
                return col
        return None

    def series_or_default(colname: Optional[str], default_val) -> pd.Series:
        """Return a numeric Series for colname (if present) else a constant Series of default_val."""
        if colname and colname in df.columns:
            return pd.to_numeric(df[colname], errors="coerce")
        # make a constant Series of the same length
        return pd.Series([default_val] * len(df), index=df.index, dtype="float64")

    # --- required: ticker ---
    tk_col = find_col(["ticker"])
    if tk_col is None:
        raise ValueError("Ticker column not found in WACC sheet.")
    df["_Ticker_"] = df[tk_col].astype(str).str.upper().str.strip()

    # --- industry (optional) ---
    ind_col = find_col(["industry group", "industry", "industry_name", "industryname"])
    if ind_col:
        df["_Industry_"] = df[ind_col].astype(str).str.strip()
    else:
        df["_Industry_"] = "UNKNOWN"

    # --- WACC (optional, leave NaN if absent) ---
    wacc_col = find_col(["wacc", "cost of capital"])
    df["_Wacc_"] = series_or_default(wacc_col, np.nan)

    # --- Terminal g (default GDP_G_DEFAULT when absent/NaN) ---
    tg_col = find_col(["terminal_g", "terminal g", "terminal growth", "terminalgrowth"])
    _tg = series_or_default(tg_col, GDP_G_DEFAULT).fillna(GDP_G_DEFAULT)
    df["_Tg_"] = _tg

    # --- Tax rate (use your WACC sheet's value; default 21%) ---
    tax_col = find_col(["taks rate", "tax rate", "tax", "taxrate"])
    _tax = series_or_default(tax_col, 0.21).fillna(0.21)
    # hard clamp to [0,1]
    df["_Tax_"] = _tax.clip(lower=0.0, upper=1.0)

    # --- optional overrides ---
    df["_SharesOverride_"]  = pd.to_numeric(df.get("SharesOverride"), errors="coerce")
    df["_NetDebtOverride_"] = pd.to_numeric(df.get("NetDebtOverride"), errors="coerce")
    df["_CloseOverride_"]   = pd.to_numeric(df.get("ClosePrice"),    errors="coerce")

    return df[[
        "_Ticker_", "_Industry_", "_Wacc_", "_Tg_", "_Tax_",
        "_SharesOverride_", "_NetDebtOverride_", "_CloseOverride_"
    ]]

# =========================
# revenue path w/ monotonic glide
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

def analyst_anchor(est_df: Optional[pd.DataFrame],
                   base_year:int, latest_annual_year: Optional[int]) -> Tuple[Optional[int], Optional[float], Optional[pd.Timestamp]]:
    if est_df is None or est_df.empty: return None, None, None
    df = est_df.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["year"] = df["date"].dt.year
    # standardize revenue field name
    if "estimatedRevenueAvg" not in df.columns:
        for alt in ("revenueAvg","estimatedRevenue","estimatedRevenueHigh","revenueHigh","revenueLow"):
            if alt in df.columns:
                df["estimatedRevenueAvg"] = pd.to_numeric(df[alt], errors="coerce")
                break
    df = df.dropna(subset=["date","estimatedRevenueAvg"])
    if df.empty: return None, None, None

    cutoff = latest_annual_year if latest_annual_year is not None else (base_year-1)
    fut = df[(df["year"] > cutoff) & (df["year"] <= base_year + PROJ_YEARS)].copy()
    if fut.empty: return None, None, None
    fut = fut.sort_values("date")
    r = fut.iloc[-1]
    return int(r["year"]), float(r["estimatedRevenueAvg"]), pd.to_datetime(r["date"])

def tone_first_year(g1: float) -> float:
    if g1 > 0.50:    # >50% → reduce by 1/4
        return g1 * 0.75
    if g1 > 0.35:    # 35–50% → reduce by 1/3
        return g1 * (2.0/3.0)
    if g1 > 0.15:    # 15–35% → reduce by 1/2
        return g1 * 0.50
    return g1

def normalize_to_ratio(g_list: List[float], target_ratio: float) -> List[float]:
    """Scale all (1+g_i) by the same factor so ∏(1+g_i) == target_ratio."""
    P = 1.0
    for g in g_list: P *= (1.0 + g)
    if P <= 0 or target_ratio <= 0:
        return g_list
    n = len(g_list)
    lam = (target_ratio / P) ** (1.0 / n)
    return [lam*(1.0 + g) - 1.0 for g in g_list]

def revenue_monotone_path(
    base_sales: float, base_year:int,
    is_ann: pd.DataFrame, est_df: Optional[pd.DataFrame],
    latest_annual_year: Optional[int],
    terminal_g: float
) -> Tuple[List[float], List[float], Optional[int], Optional[pd.Timestamp], Optional[float]]:
    """
    Build Y1..Y5 sales and yoy with monotonic glides.
    If analyst anchor exists (year A, sales S_A), ensure product to year A matches exactly.
    Beyond anchor, glide down to terminal g by Y5 (never increase).
    If no analyst, use hist CAGR and glide down to terminal.
    """
    # Try analyst anchor
    A_year, S_A, A_date = analyst_anchor(est_df, base_year, latest_annual_year)

    if A_year is not None and is_finite(S_A) and S_A > 0:
        n = max(1, int(A_year - base_year))
        g_imp = (S_A / base_sales)**(1.0/n) - 1.0

        # shape a raw decreasing sequence for the n years
        g1 = tone_first_year(g_imp)
        gN_raw = max(terminal_g, g1 * 0.4)  # keep above terminal, but significantly lower than g1
        if gN_raw > g1: gN_raw = g1  # never increase

        if n == 1:
            g_list = [g1]
        else:
            g_list = list(np.linspace(g1, gN_raw, num=n))

        # normalize to match anchor exactly
        target_ratio = S_A / base_sales
        g_adj = normalize_to_ratio(g_list, target_ratio)

        # build full 5-year YoY
        yoy = []
        yoy.extend(g_adj)  # years to anchor
        # remaining years: glide toward terminal_g without increasing
        remaining = PROJ_YEARS - n
        if remaining > 0:
            start_g = g_adj[-1]
            if start_g <= terminal_g:
                # already at/below terminal; keep flat at start_g (avoid increase)
                yoy.extend([start_g]*remaining)
            else:
                # linear down to terminal over remaining years
                tail = list(np.linspace(start_g, terminal_g, num=remaining+1))[1:]
                # make sure non-increasing strictly
                last = start_g
                tail_mono = []
                for t in tail:
                    t2 = min(last, t)
                    tail_mono.append(t2)
                    last = t2
                yoy.extend(tail_mono)
        # sales path
        sales = []
        s = base_sales
        for g in yoy:
            s *= (1.0 + g)
            sales.append(s)
        return sales, yoy, A_year, A_date, S_A

    # No analyst anchor → hist CAGR then glide to terminal
    g_hist = hist_rev_cagr(is_ann)
    if not is_finite(g_hist): g_hist = terminal_g
    # set decreasing path from g_hist to terminal by Y5
    if PROJ_YEARS == 1:
        yoy = [g_hist]
    else:
        end_g = min(g_hist, terminal_g + 0.5*(g_hist - terminal_g))  # keep above terminal
        if end_g < terminal_g: end_g = terminal_g
        # If g_hist < terminal_g (decline business), avoid upward move; keep flat at g_hist
        if g_hist <= terminal_g:
            yoy = [g_hist]*PROJ_YEARS
        else:
            yoy = list(np.linspace(g_hist, terminal_g, num=PROJ_YEARS))
    # Sales
    sales = []
    s = base_sales
    for g in yoy:
        s *= (1.0 + g)
        sales.append(s)
    return sales, yoy, None, None, None

# =========================
# EBIT margin glide (TTM -> annual median)
# =========================

def ttm_ebit_margin(is_q: pd.DataFrame) -> Optional[float]:
    if is_q is None or is_q.empty: return None
    df = df_sort_by_date(is_q).copy()
    if "ebit" not in df.columns:
        if "operatingIncome" in df.columns:
            df["ebit"] = df["operatingIncome"]
        else:
            return None
    rev = pd.to_numeric(df.get("revenue"), errors="coerce")
    ebt = pd.to_numeric(df.get("ebit"), errors="coerce")
    last4 = pd.DataFrame({"revenue": rev, "ebit": ebt}).head(4).sum(numeric_only=True)
    if is_finite(last4.get("revenue")) and last4["revenue"] > 0:
        return float(last4["ebit"] / last4["revenue"])
    return None

def trailing_median_margin(is_ann: pd.DataFrame, lookback:int=HIST_LOOKBACK_YEARS) -> Optional[float]:
    if is_ann is None or is_ann.empty: return None
    df = df_sort_by_date(is_ann).head(lookback).copy()
    if "ebit" not in df.columns:
        if "operatingIncome" in df.columns:
            df["ebit"] = df["operatingIncome"]
        else:
            return None
    rev = pd.to_numeric(df["revenue"], errors="coerce")
    ebt = pd.to_numeric(df["ebit"], errors="coerce")
    m = (ebt / rev).replace([np.inf,-np.inf], np.nan).dropna()
    if m.empty: return None
    return float(np.median(m))

def ebit_margin_curve(is_ann: pd.DataFrame, is_q: pd.DataFrame) -> List[float]:
    m_ttm = ttm_ebit_margin(is_q)
    m_med = trailing_median_margin(is_ann, lookback=HIST_LOOKBACK_YEARS)

    if m_ttm is None and m_med is None:
        m_start = 0.05
        m_target = 0.05 - EPSILON_NUDGE
    else:
        if m_ttm is None: m_ttm = m_med
        if m_med is None: m_med = m_ttm - EPSILON_NUDGE
        m_start = float(np.clip(m_ttm, M_MIN, M_MAX))
        m_target = float(np.clip(m_med, M_MIN, M_MAX))

    # if start is negative/tiny, lift to a small floor
    if m_start <= 0:
        m_start = max(0.03, m_target)  # keep sensible
    # build linear glide with step cap
    raw = list(np.linspace(m_start, m_target, num=PROJ_YEARS))
    curve = [raw[0]]
    for i in range(1, PROJ_YEARS):
        prev = curve[-1]
        desired = raw[i]
        delta = desired - prev
        delta = np.clip(delta, -MAX_MARGIN_STEP, MAX_MARGIN_STEP)
        curve.append(float(np.clip(prev + delta, M_MIN, M_MAX)))
    # enforce monotonic trend (no zigzags). If m_target < m_start, make non-increasing; else non-decreasing
    if curve[-1] < curve[0]:
        # non-increasing
        for i in range(1, PROJ_YEARS):
            curve[i] = min(curve[i-1], curve[i])
    else:
        # non-decreasing
        for i in range(1, PROJ_YEARS):
            curve[i] = max(curve[i-1], curve[i])
    return curve

# =========================
# Ratios: D&A, Capex, k for ΔNWC
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
        dNWC = -dWC  # CF sign convention
        sales_series = sales.reset_index(drop=True)
        dNWC = dNWC.reset_index(drop=True)
        dS = sales_series.diff()
        mask = (dS.abs() >= MIN_DS_FOR_RATIO) & sales_series.notna() & dNWC.notna()
        k = (dNWC[mask] / dS[mask]).replace([np.inf,-np.inf], np.nan).dropna()
        if not k.empty:
            k = k.clip(-0.5, 0.5)
            k_dsales = float(np.median(k))

    # clamps
    da_ratio    = float(np.clip(da_ratio, 0.0, 0.25))
    capex_ratio = float(np.clip(capex_ratio, 0.0, 0.25))
    k_dsales    = float(np.clip(k_dsales, -0.5, 0.5))
    return da_ratio, capex_ratio, k_dsales

# =========================
# Shares / NetDebt / Close
# =========================

def shares_netdebt_close(fin: Dict[str,pd.DataFrame]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    shares = None
    netdebt = None
    price_file = None

    prof = fin.get("Company_Profile")
    if prof is not None and not prof.empty:
        df = prof.copy()
        for c in ["sharesOutstanding","shareOutstanding","sharedOutstanding","shares_outstanding"]:
            if c in df.columns and not is_finite(shares):
                shares = first_number(df[c].iloc[0])
        for pc in ["price","lastPrice","close","previousClose"]:
            if pc in df.columns and not is_finite(price_file):
                price_file = first_number(df[pc].iloc[0])

    ev = fin.get("Enterprise_Values")
    if ev is not None and not ev.empty:
        df = df_sort_by_date(ev)
        # Shares if present
        for sc in ["numberOfShares","shares","Shares"]:
            if sc in df.columns and not is_finite(shares):
                shares = first_number(df[sc].iloc[0])
        # Price if present
        for pc in ["stockPrice","price","close","previousClose"]:
            if pc in df.columns and not is_finite(price_file):
                price_file = first_number(df[pc].iloc[0])
        # MarketCap & EV
        mcap = None; EVv = None; addDebt=None; minusCash=None
        for mc in ["marketCapitalization","marketCap","MarketCap","mktCap"]:
            if mc in df.columns: mcap = first_number(df[mc].iloc[0])
        for ec in ["enterpriseValue","EnterpriseValue"]:
            if ec in df.columns: EVv = first_number(df[ec].iloc[0])
        for ad in ["addTotalDebt","totalDebt","TotalDebt"]:
            if ad in df.columns: addDebt = first_number(df[ad].iloc[0])
        for mcash in ["minusCashAndCashEquivalents","minusCash","MinusCashAndCashEquivalents"]:
            if mcash in df.columns: minusCash = first_number(df[mcash].iloc[0])

        # Prefer direct net debt from EV table if both parts exist:
        nd1 = None
        if is_finite(addDebt) and is_finite(minusCash):
            nd1 = addDebt - minusCash
        nd2 = None
        if is_finite(EVv) and is_finite(mcap):
            nd2 = EVv - mcap

        netdebt = nd1 if is_finite(nd1) else (nd2 if is_finite(nd2) else netdebt)

        # Shares from MarketCap/Price if needed
        if (not is_finite(shares)) and is_finite(mcap) and is_finite(price_file) and price_file > 0:
            shares = mcap / price_file

    # fallback NetDebt from BS_Annual
    if (not is_finite(netdebt)) or netdebt is None:
        bsa = fin.get("BS_Annual")
        if bsa is not None and not bsa.empty:
            bs = df_sort_by_date(bsa)
            cash = first_number(bs.get("cashAndCashEquivalents", pd.Series([np.nan])).iloc[0])
            sti  = first_number(bs.get("shortTermInvestments", pd.Series([np.nan])).iloc[0])
            sdebt= first_number(bs.get("shortTermDebt", pd.Series([np.nan])).iloc[0])
            ldebt= first_number(bs.get("longTermDebt", pd.Series([np.nan])).iloc[0])
            lease= first_number(bs.get("capitalLeaseObligations", pd.Series([0])).iloc[0])
            _cash = (cash or 0.0) + (sti or 0.0)
            _debt = (sdebt or 0.0) + (ldebt or 0.0) + (lease or 0.0)
            netdebt = _debt - _cash if (is_finite(_debt) and is_finite(_cash)) else None

    shares = shares if is_finite(shares) and shares > 0 else None
    netdebt = netdebt if is_finite(netdebt) else None
    price_file = price_file if is_finite(price_file) and price_file > 0 else None
    return shares, netdebt, price_file

def last_close_api(symbol: str) -> Optional[float]:
    if not FMP_API_KEY: return None
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
    params = {"apikey": FMP_API_KEY}
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code in (429,500,502,503,504):
                time.sleep(BACKOFF**attempt); continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data,list) and data:
                row = data[0]
                px = first_number(row.get("price"), row.get("previousClose"),
                                  row.get("close"), row.get("dayClose"))
                return px
            return None
        except Exception:
            if attempt == MAX_RETRIES: return None
            time.sleep(BACKOFF**attempt)
    return None

# =========================
# DCF helpers
# =========================

def discount_factor(wacc: float, t:int) -> float:
    return (1.0 / ((1.0 + wacc)**t)) if (is_finite(wacc) and wacc > 0) else 0.0

def gordon_terminal(fcff_last: float, wacc: float, g: float) -> float:
    if not (is_finite(fcff_last) and is_finite(wacc) and is_finite(g)): return 0.0
    if (wacc - g) <= 0: g = min(g, wacc - 1e-6)
    return fcff_last * (1.0 + g) / (wacc - g)

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
        wacc = safe_float(r["_Wacc_"], np.nan)
        tg   = safe_float(r["_Tg_"], GDP_G_DEFAULT)
        tax  = safe_float(r["_Tax_"], 0.21)

        fin = read_financials(sym)
        if not fin: continue

        is_ann = fin.get("IS_Annual")
        is_q   = fin.get("IS_Quarterly")
        cf_ann = fin.get("CF_Annual")

        latest_rev, latest_year = latest_annual_rev_and_year(is_ann)
        if not is_finite(latest_rev): continue

        base_sales = float(latest_rev)
        base_year  = BASE_YEAR

        # Revenue: monotonic glide
        sales_fwd, yoy_fwd, anchor_year, anchor_date, anchor_sales = revenue_monotone_path(
            base_sales=base_sales, base_year=base_year,
            is_ann=is_ann, est_df=fin.get("Analyst_Estimates"),
            latest_annual_year=latest_year, terminal_g=tg
        )

        # EBIT margin: monotonic glide TTM -> trailing median
        ebit_margins = ebit_margin_curve(is_ann, is_q)

        # Ratios
        da_ratio, capex_ratio, k_dsales = hist_ratios(is_ann, cf_ann)

        # Shares / NetDebt / Close
        sh_f, nd_f, px_file = shares_netdebt_close(fin)
        shares  = sh_f if is_finite(sh_f) else safe_float(r["_SharesOverride_"])
        netdebt = nd_f if is_finite(nd_f) else safe_float(r["_NetDebtOverride_"], 0.0)
        if not is_finite(netdebt): netdebt = 0.0

        px_api = last_close_api(sym)
        close_px = px_api if is_finite(px_api) else safe_float(r["_CloseOverride_"])
        if not is_finite(close_px): close_px = px_file if is_finite(px_file) else np.nan

        # Build per-year FCFF
        years = [base_year + i for i in range(1, PROJ_YEARS + 1)]
        s_prev = base_sales

        fcff_list, pv_list = [], []
        ebit_list, nopat_list, da_list, capex_list, dnwc_list = [], [], [], [], []

        for i, (yr, s, g, m) in enumerate(zip(years, sales_fwd, yoy_fwd, ebit_margins), start=1):
            ebit = s * m
            nopat = ebit * (1.0 - tax)
            da = s * da_ratio
            capex = s * capex_ratio
            ds = s - s_prev
            dNWC = k_dsales * ds

            fcff = nopat - capex - dNWC + da
            df_t = discount_factor(wacc, i)
            pv_t = fcff * df_t

            ebit_list.append(ebit); nopat_list.append(nopat)
            da_list.append(da); capex_list.append(capex); dnwc_list.append(dNWC)
            fcff_list.append(fcff); pv_list.append(pv_t)

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

        # Terminal
        tv = gordon_terminal(fcff_list[-1], wacc, tg)
        df5 = discount_factor(wacc, PROJ_YEARS)
        pv_tv = tv * df5

        EV = float(np.nansum(pv_list)) + pv_tv
        equity = EV - (netdebt if is_finite(netdebt) else 0.0)
        vps = equity / shares if (is_finite(shares) and shares > 0) else np.nan

        market_ev = np.nan
        if is_finite(close_px) and is_finite(shares):
            market_ev = close_px * shares + (netdebt if is_finite(netdebt) else 0.0)
        dcf_vs_market_ev = EV / market_ev if (is_finite(market_ev) and market_ev > 0) else np.nan

        summary_rows.append({
            "Ticker": sym,
            "Industry Group": industry,
            "BaseYear": base_year,
            "Sales_Base": base_sales,
            "WACC": wacc,
            "Terminal_g": tg,
            "Tax": tax,
            "DA_Ratio": da_ratio,
            "Capex_Ratio": capex_ratio,
            "k_DeltaSales_for_DeltaNWC": k_dsales,
            "Analyst_Used": bool(anchor_year is not None),
            "AnchorYear": anchor_year,
            "AnchorDate": anchor_date.date() if isinstance(anchor_date, pd.Timestamp) else None,
            "AnchorSales": anchor_sales,
            "EBIT_Margin_Y1": ebit_margins[0] if ebit_margins else np.nan,
            "EBIT_Margin_Y5": ebit_margins[-1] if ebit_margins else np.nan,
            "Shares": shares,
            "NetDebt": netdebt,
            "ClosePrice": close_px,
            "EV_from_DCF": EV,
            "EquityValue": equity,
            "ValuePerShare": vps,
            "MarketEV_ifPrice": market_ev,
            "DCF_to_MarketEV": dcf_vs_market_ev,
            # show projected sales & YoY
            "Sales_2026": sales_fwd[0] if sales_fwd else np.nan,
            "Sales_2027": sales_fwd[1] if len(sales_fwd)>1 else np.nan,
            "Sales_2028": sales_fwd[2] if len(sales_fwd)>2 else np.nan,
            "Sales_2029": sales_fwd[3] if len(sales_fwd)>3 else np.nan,
            "Sales_2030": sales_fwd[4] if len(sales_fwd)>4 else np.nan,
            "YoY_2026": yoy_fwd[0] if yoy_fwd else np.nan,
            "YoY_2027": yoy_fwd[1] if len(yoy_fwd)>1 else np.nan,
            "YoY_2028": yoy_fwd[2] if len(yoy_fwd)>2 else np.nan,
            "YoY_2029": yoy_fwd[3] if len(yoy_fwd)>3 else np.nan,
            "YoY_2030": yoy_fwd[4] if len(yoy_fwd)>4 else np.nan,
        })

        # attach terminal values onto last per-year row for this ticker
        peryear_rows[-1]["TerminalValue"] = tv
        peryear_rows[-1]["PV_TerminalValue"] = pv_tv

    # Build DataFrames & save
    summary_df = pd.DataFrame(summary_rows)
    peryear_df = pd.DataFrame(peryear_rows)

    sum_cols = [
        "Ticker","Industry Group","BaseYear","Sales_Base","WACC","Terminal_g","Tax",
        "DA_Ratio","Capex_Ratio","k_DeltaSales_for_DeltaNWC",
        "Analyst_Used","AnchorYear","AnchorDate","AnchorSales",
        "EBIT_Margin_Y1","EBIT_Margin_Y5",
        "Shares","NetDebt","ClosePrice",
        "EV_from_DCF","EquityValue","ValuePerShare",
        "MarketEV_ifPrice","DCF_to_MarketEV",
        "Sales_2026","Sales_2027","Sales_2028","Sales_2029","Sales_2030",
        "YoY_2026","YoY_2027","YoY_2028","YoY_2029","YoY_2030",
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

    print(f"Done. Saved to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
