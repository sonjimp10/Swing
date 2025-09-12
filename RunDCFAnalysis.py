import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# USER CONFIG
# =========================
EXCEL_PATH = Path("AAPL_FMP_financials.xlsx")   # your file from earlier script
TICKER = "AAPL"

# Valuation knobs (tweak if you like)
PROJ_YEARS = 5
RISK_FREE = 0.04     # long bond proxy
ERP = 0.055          # equity risk premium
WACC_FALLBACK = 0.085
PRETAX_COD_FALLBACK = 0.045
TAX_FALLBACK = 0.20
G_TERMINAL = 0.025
INITIAL_GROWTH_CAP = 0.12   # cap on initial growth
USE_TTM_FCF = True          # << use TTM from quarterly CF, else 3-yr avg annual
SUBTRACT_SBC = False        # conservative option: subtract stock-based comp from FCF

# Include long-term marketable securities in "cash" (Apple-style portfolio)
INCLUDE_LONG_TERM_SECURITIES_AS_CASH = True

# Scenario grid (for a quick gut check)
SCENARIO_WACCS = [0.065, 0.075, 0.085, 0.095]
SCENARIO_GS = [0.01, 0.02, 0.03]

# =========================
# HELPERS
# =========================
def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_shares(sh):
    """If shares look like 'millions', scale to raw shares."""
    if sh is None or not np.isfinite(sh):
        return None
    if sh < 1e6:        # many APIs give millions
        return sh * 1_000_000
    return sh

def linear_fade(start, end, n):
    if n == 1: return [end]
    return list(np.linspace(start, end, n))

def cagr_from_descending(series):
    """Series assumed most-recent first; CAGR over ~5 points if available."""
    s = to_num(series.dropna())
    s = s[s > 0]
    if len(s) < 2:
        return None
    years = len(s) - 1
    # recent / oldest
    return (s.iloc[0] / s.iloc[-1]) ** (1/years) - 1

def present_value(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows, start=1))

# =========================
# LOAD SHEETS
# =========================
xls = pd.ExcelFile(EXCEL_PATH)
need = {"IS_Annual","BS_Annual","CF_Annual"}
for s in need:
    if s not in xls.sheet_names:
        raise ValueError(f"Missing required sheet: {s}")

is_a = pd.read_excel(EXCEL_PATH, sheet_name="IS_Annual")
bs_a = pd.read_excel(EXCEL_PATH, sheet_name="BS_Annual")
cf_a = pd.read_excel(EXCEL_PATH, sheet_name="CF_Annual")

# Sort recent->old
for df in (is_a, bs_a, cf_a):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)

# Optional sheets
profile = pd.read_excel(EXCEL_PATH, sheet_name="Company_Profile") if "Company_Profile" in xls.sheet_names else pd.DataFrame()
enterprise = pd.read_excel(EXCEL_PATH, sheet_name="Enterprise_Values") if "Enterprise_Values" in xls.sheet_names else pd.DataFrame()
key_metrics = pd.read_excel(EXCEL_PATH, sheet_name="Key_Metrics") if "Key_Metrics" in xls.sheet_names else pd.DataFrame()
cf_q = pd.read_excel(EXCEL_PATH, sheet_name="CF_Quarterly") if "CF_Quarterly" in xls.sheet_names else pd.DataFrame()

# =========================
# BUILD INPUTS
# =========================

# 1) FCF base (prefer TTM from quarterly)
cfo_cols = ["netCashProvidedByOperatingActivities","netCashProvidedByUsedInOperatingActivities","operatingCashFlow"]
capex_cols = ["capitalExpenditure","capitalExpenditures","investmentsInPropertyPlantAndEquipment"]
sbc_cols = ["stockBasedCompensation","stockBasedCompensationExpense"]

def get_fcf_annual(df):
    cfo_col = pick(df, cfo_cols)
    capex_col = pick(df, capex_cols)
    if not cfo_col or not capex_col:
        return None, None, None
    cfo = to_num(df[cfo_col])
    capex = to_num(df[capex_col])  # usually negative on FMP; FCF = CFO - CapEx works
    fcf = cfo - capex
    return fcf, cfo, capex

def get_fcf_ttm_from_quarterly(cf_q):
    if cf_q.empty: return None, None, None
    if "date" in cf_q.columns:
        cf_q = cf_q.copy()
        cf_q["date"] = pd.to_datetime(cf_q["date"], errors="coerce")
        cf_q.sort_values("date", ascending=False, inplace=True)
    cfo_col = pick(cf_q, cfo_cols)
    capex_col = pick(cf_q, capex_cols)
    if not cfo_col or not capex_col:
        return None, None, None
    cfo_ttm = to_num(cf_q[cfo_col]).iloc[:4].sum()
    capex_ttm = to_num(cf_q[capex_col]).iloc[:4].sum()
    fcf_ttm = cfo_ttm - capex_ttm
    return fcf_ttm, cfo_ttm, capex_ttm

# Base FCF
fcf_annual, cfo_annual, capex_annual = get_fcf_annual(cf_a)
fcf_base = None
if USE_TTM_FCF:
    fcf_ttm, cfo_ttm, capex_ttm = get_fcf_ttm_from_quarterly(cf_q)
    if fcf_ttm is not None and np.isfinite(fcf_ttm):
        fcf_base = float(fcf_ttm)
if fcf_base is None:
    # fallback: average of last 3 annuals
    fcf_base = float(to_num(fcf_annual).iloc[:3].mean()) if fcf_annual is not None else None

# Optional SBC subtraction
if SUBTRACT_SBC:
    sbc_col_a = pick(cf_a, sbc_cols)
    if sbc_col_a:
        sbc_ttm = float(to_num(cf_a[sbc_col_a]).iloc[:1])  # use latest annual SBC as proxy
        fcf_base = (fcf_base or 0.0) - sbc_ttm

# 2) Growth path: infer from revenue CAGR, cap at INITIAL_GROWTH_CAP
rev_col = pick(is_a, ["revenue","salesRevenueNet"])
rev_cagr = cagr_from_descending(is_a[rev_col]) if rev_col else None
g_start = min(max(rev_cagr if rev_cagr is not None else 0.06, 0.00), INITIAL_GROWTH_CAP)  # clamp 0..cap
growth_path = linear_fade(g_start, G_TERMINAL, PROJ_YEARS)

# 3) Tax rate (3-yr avg)
pretax_col = pick(is_a, ["incomeBeforeTax","incomeBeforeIncomeTaxes"])
tax_exp_col = pick(is_a, ["incomeTaxExpense","incomeTaxExpenseBenefit"])
if pretax_col and tax_exp_col:
    pretax = to_num(is_a[pretax_col]).iloc[:3]
    taxexp = to_num(is_a[tax_exp_col]).iloc[:3]
    eff_tax = (taxexp / pretax.replace(0, np.nan)).clip(0.00, 0.35).mean()
    tax_rate = float(eff_tax) if np.isfinite(eff_tax) else TAX_FALLBACK
else:
    tax_rate = TAX_FALLBACK

# 4) Cost of debt from IS & BS
int_col = pick(is_a, ["interestExpense","interestExpenseNonOperating"])
debt_col = pick(bs_a, ["totalDebt","totalDebtUSD"])
if int_col and debt_col:
    interest = abs(to_num(is_a[int_col]).iloc[:3].mean())
    debt_now = to_num(bs_a[debt_col]).iloc[0]
    debt_prev = to_num(bs_a[debt_col]).iloc[1] if len(bs_a) > 1 else debt_now
    avg_debt = np.nanmean([debt_now, debt_prev])
    if avg_debt and avg_debt > 0:
        cod_pre = float((interest / avg_debt))
        cod_pre = float(np.clip(cod_pre, 0.01, 0.12))
    else:
        cod_pre = PRETAX_COD_FALLBACK
else:
    cod_pre = PRETAX_COD_FALLBACK
cod_after = cod_pre * (1 - tax_rate)

# 5) Cost of equity via CAPM
beta = None
if "beta" in profile.columns:
    beta = float(to_num(profile["beta"]).dropna().iloc[0])
if beta is None or not np.isfinite(beta):
    beta = 1.10
coe = RISK_FREE + beta * ERP

# 6) Capital structure weights
price = float(to_num(profile["price"]).dropna().iloc[0]) if "price" in profile.columns and not profile.empty else None
mktcap = float(to_num(profile["mktCap"]).dropna().iloc[0]) if "mktCap" in profile.columns and not profile.empty else None
shares = None

# profile often has sharesOutstanding; else use weightedAverageShsOutDil
if "sharesOutstanding" in profile.columns and not profile.empty:
    shares = normalize_shares(float(to_num(profile["sharesOutstanding"]).dropna().iloc[0]))
elif "weightedAverageShsOutDil" in is_a.columns:
    shares = normalize_shares(float(to_num(is_a["weightedAverageShsOutDil"]).dropna().iloc[0]))

# fallback: derive from mktcap/price
if (shares is None or not np.isfinite(shares)) and (mktcap and price and price > 0):
    shares = mktcap / price

# if enterprise sheet has numberOfShares, prefer that (usually raw shares)
if (shares is None or not np.isfinite(shares)) and ("numberOfShares" in enterprise.columns):
    shares = float(to_num(enterprise["numberOfShares"]).dropna().iloc[0])

# derive market cap if missing
if (mktcap is None or not np.isfinite(mktcap)) and (shares and price):
    mktcap = shares * price

# 7) Net cash: include cash & short-term investments; optionally long-term marketable securities
cash_cols = [
    "cashAndCashEquivalents","cashAndShortTermInvestments","shortTermInvestments"
]
lt_sec_cols = [
    "longTermMarketableSecurities","marketableSecurities","longTermInvestments"
]

cash_like = 0.0
for c in cash_cols:
    if c in bs_a.columns:
        cash_like += float(to_num(bs_a[c]).dropna().iloc[0])
if INCLUDE_LONG_TERM_SECURITIES_AS_CASH:
    for c in lt_sec_cols:
        if c in bs_a.columns:
            cash_like += float(to_num(bs_a[c]).dropna().iloc[0])

total_debt = float(to_num(bs_a[debt_col]).dropna().iloc[0]) if debt_col else 0.0
net_debt = total_debt - cash_like

# 8) WACC
E = max(mktcap or 0.0, 0.0)
D = max(total_debt, 0.0)
V = E + D if (E + D) > 0 else 1.0
w_e = E / V
w_d = D / V
wacc = w_e * coe + w_d * cod_after
if not np.isfinite(wacc) or wacc <= 0:
    wacc = WACC_FALLBACK

# =========================
# PROJECTIONS & DCF
# =========================
base_fcf = float(fcf_base or 0.0)
proj_rows = []
fcf = base_fcf
for year, g in enumerate(growth_path, start=1):
    fcf = fcf * (1 + g)
    df = (1 + wacc) ** year
    pv = fcf / df
    proj_rows.append({"Year": year, "Growth": g, "FCF": fcf, "DiscountFactor": df, "PV_FCF": pv})

proj_df = pd.DataFrame(proj_rows)

# Terminal value (perpetuity)
g_term = min(G_TERMINAL, wacc - 0.005)  # safety
tv = proj_df["FCF"].iloc[-1] * (1 + g_term) / (wacc - g_term)
pv_tv = tv / ((1 + wacc) ** PROJ_YEARS)

ev = proj_df["PV_FCF"].sum() + pv_tv
equity_value = ev - net_debt
fv_per_share = (equity_value / shares) if shares and shares > 0 else np.nan
upside = (fv_per_share / price - 1.0) if (price and fv_per_share and price > 0) else np.nan

# =========================
# SCENARIO GRID
# =========================
def fair_value_with(wacc_in, g_term_in):
    # rebuild once using same base_fcf and same growth_path shape but with terminal g swap
    fcf_local = base_fcf
    pvs = []
    for i, g in enumerate(growth_path, start=1):
        fcf_local = fcf_local * (1 + g)
        pvs.append(fcf_local / ((1 + wacc_in) ** i))
    tv_local = fcf_local * (1 + min(g_term_in, wacc_in - 0.005)) / (wacc_in - min(g_term_in, wacc_in - 0.005))
    pv_tv_local = tv_local / ((1 + wacc_in) ** PROJ_YEARS)
    ev_local = sum(pvs) + pv_tv_local
    eq_local = ev_local - net_debt
    return (eq_local / shares) if shares and shares > 0 else np.nan

sc_rows = []
for w in SCENARIO_WACCS:
    for g in SCENARIO_GS:
        sc_rows.append({"WACC": w, "g_terminal": g, "FairValuePerShare": fair_value_with(w, g)})
sc_df = pd.DataFrame(sc_rows)

# =========================
# DEBUG TABLES
# =========================
assumptions = pd.DataFrame({
    "Field":[
        "Ticker","Base FCF source","Base FCF (USD)",
        "Initial growth (g_start)","Terminal growth (g_term)","Years",
        "Risk-free (Rf)","ERP","Beta","Cost of equity",
        "Pre-tax cost of debt","After-tax cost of debt","Tax rate",
        "Weight Equity","Weight Debt","WACC",
        "Total Debt","Cash-like (incl LT secs?)","Net Debt",
        "Shares used","Price","Market Cap"
    ],
    "Value":[
        TICKER, "TTM Quarterly" if USE_TTM_FCF else "Avg of last 3 annual", base_fcf,
        g_start, g_term, PROJ_YEARS,
        RISK_FREE, ERP, beta, coe,
        cod_pre, cod_after, tax_rate,
        w_e, w_d, wacc,
        total_debt, cash_like, net_debt,
        shares, price, mktcap
    ]
})

# =========================
# WRITE BACK
# =========================
with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
    assumptions.to_excel(w, sheet_name="DCF_Assumptions", index=False)
    proj_out = proj_df.copy()
    proj_out.loc[len(proj_out)] = {
        "Year":"Terminal","Growth":g_term,"FCF":proj_df["FCF"].iloc[-1]*(1+g_term),
        "DiscountFactor":(1+wacc)**PROJ_YEARS,"PV_FCF":pv_tv
    }
    proj_out.to_excel(w, sheet_name="DCF_Projections", index=False)
    pd.DataFrame({
        "Metric":["Enterprise Value","Equity Value","Fair Value / Share","Current Price","Implied Upside"],
        "Value":[ev, equity_value, fv_per_share, price, upside]
    }).to_excel(w, sheet_name="DCF_Summary", index=False)
    sc_df.to_excel(w, sheet_name="DCF_Scenarios", index=False)

print("\nDCF updated. Sheets: DCF_Assumptions, DCF_Projections, DCF_Summary, DCF_Scenarios")
print(f"Fair Value / Share = {fv_per_share:,.2f}" if np.isfinite(fv_per_share) else "Fair Value / Share: n/a")
if np.isfinite(upside):
    print(f"Implied Upside = {upside*100:,.1f}%")
