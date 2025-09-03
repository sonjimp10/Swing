import time
import pandas as pd
import requests

API_KEY   = "BUy1zjTqw4dpREy1p96iqGvG4npO9qJg"
INPUT_FILE = "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/Short/SmallCap.xlsx"
WORKSHEET      = "Sheet1"

def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=WORKSHEET)
    df["Ticker"] = df["Ticker"].str.upper().str.strip()

    floats = {}
    for symbol in df["Ticker"].unique():
        print(f"Getting Float for - {symbol}")
        url    = "https://financialmodelingprep.com/api/v4/shares_float"
        params = {"symbol": symbol, "apikey": API_KEY}
        resp   = requests.get(url, params=params)
        resp.raise_for_status()

        data = resp.json()
        # data is a list of one dict, e.g. [{"symbol":"AAPL","floatShares":16501118000}]
        if isinstance(data, list) and data:
            floats[symbol] = data[0].get("floatShares")
        else:
            floats[symbol] = None

        time.sleep(0.25)  # keep under 300 calls/minute

    # map back into your DataFrame
    df["FloatShares"] = df["Ticker"].map(floats)

    # write it back out
    from openpyxl import load_workbook
    from pandas import ExcelWriter

    # assume `df` is the DataFrame you want to save, and WORKSHEET is your sheet name
    with ExcelWriter(INPUT_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=WORKSHEET, index=False)
    print(f"✅ Wrote FloatShares for {len(floats)} tickers into {INPUT_FILE}")

if __name__ == "__main__":
    main()
