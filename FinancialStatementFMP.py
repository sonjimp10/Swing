import os
import time
import sys
import json
from typing import Dict, Any, List, Optional
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_URL = "https://financialmodelingprep.com/api/v3"
SYMBOL = "AAPL"

# Option A (recommended): read from environment variable
#API_KEY = os.getenv("FMP_API_KEY")

# Option B: hardcode here (less secure)
API_KEY = "BUy1zjTqw4dpREy1p96iqGvG4npO9qJg"

OUTPUT_XLSX = f"{SYMBOL}_FMP_financials.xlsx"
TIMEOUT = 20  # seconds
MAX_RETRIES = 5
BACKOFF_SECS = 2  # exponential backoff base

if not API_KEY:
    print(
        "No API key found. Set FMP_API_KEY environment variable or paste your key into API_KEY.",
        file=sys.stderr,
    )
    sys.exit(1)

session = requests.Session()
session.headers.update({"User-Agent": "fmp-dcf-retail-script/1.0"})

def get_with_retry(url: str, params: Dict[str, Any]) -> Optional[requests.Response]:
    """GET with retries/backoff for 429 & 5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
            # Retry on common transient errors
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_SECS ** attempt
                print(f"[{resp.status_code}] Retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            # For connection timeouts etc., backoff then retry
            if attempt == MAX_RETRIES:
                print(f"Request failed after {MAX_RETRIES} attempts: {e}", file=sys.stderr)
                return None
            wait = BACKOFF_SECS ** attempt
            print(f"[Error] {e}. Retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
    return None

def fetch_fmp_json(endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch JSON data from FMP endpoint.
    Returns a list of dicts (empty list on issues).
    """
    url = f"{BASE_URL}/{endpoint.strip('/')}"
    params = {**params, "apikey": API_KEY}
    resp = get_with_retry(url, params)
    if resp is None:
        return []
    try:
        data = resp.json()
        # FMP sometimes returns dicts (with 'historical') or lists; normalize to list
        if isinstance(data, dict) and "historical" in data:
            return data["historical"] or []
        if isinstance(data, dict):
            # Some endpoints return dicts. Wrap into list to keep pandas happy.
            return [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print("Failed to parse JSON from FMP response.", file=sys.stderr)
        return []

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dicts to DataFrame, handling empty gracefully."""
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    # Sort by date if present
    for date_col in ("date", "fillingDate", "filingDate", "acceptedDate"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # Prefer reverse chronological order if 'date' exists
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df

def get_statements(symbol: str, period: str = "annual", limit: int = 40) -> Dict[str, pd.DataFrame]:
    """
    Download Income Statement, Balance Sheet, Cash Flow for a symbol.
    period: 'annual' or 'quarter'
    """
    assert period in ("annual", "quarter"), "period must be 'annual' or 'quarter'"
    params = {"period": period, "limit": limit}

    endpoints = {
        "income_statement": f"income-statement/{symbol}",
        "balance_sheet": f"balance-sheet-statement/{symbol}",
        "cash_flow": f"cash-flow-statement/{symbol}",
    }

    frames = {}
    for name, ep in endpoints.items():
        print(f"Fetching {name.replace('_',' ').title()} ({period})…")
        data = fetch_fmp_json(ep, params)
        df = to_dataframe(data)
        frames[name] = df
        print(f"  -> {len(df):,} rows, {len(df.columns)} columns")

    return frames

def main():
    # Annual statements
    annual = get_statements(SYMBOL, period="annual", limit=40)
    # Quarterly statements
    quarterly = get_statements(SYMBOL, period="quarter", limit=40)

    # Optional: basic metadata (company profile, enterprise value, key metrics)
    print("Fetching company profile / enterprise values / key metrics…")
    profile = to_dataframe(fetch_fmp_json(f"profile/{SYMBOL}", {}))
    enterprise_vals = to_dataframe(fetch_fmp_json(f"enterprise-values/{SYMBOL}", {"period": "annual", "limit": 40}))
    key_metrics = to_dataframe(fetch_fmp_json(f"key-metrics/{SYMBOL}", {"period": "annual", "limit": 40}))

    # Save everything to one Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        # Annual
        annual["income_statement"].to_excel(writer, sheet_name="IS_Annual", index=False)
        annual["balance_sheet"].to_excel(writer, sheet_name="BS_Annual", index=False)
        annual["cash_flow"].to_excel(writer, sheet_name="CF_Annual", index=False)

        # Quarterly
        quarterly["income_statement"].to_excel(writer, sheet_name="IS_Quarterly", index=False)
        quarterly["balance_sheet"].to_excel(writer, sheet_name="BS_Quarterly", index=False)
        quarterly["cash_flow"].to_excel(writer, sheet_name="CF_Quarterly", index=False)

        # Meta
        profile.to_excel(writer, sheet_name="Company_Profile", index=False)
        enterprise_vals.to_excel(writer, sheet_name="Enterprise_Values", index=False)
        key_metrics.to_excel(writer, sheet_name="Key_Metrics", index=False)

    print(f"\nDone. Saved to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
