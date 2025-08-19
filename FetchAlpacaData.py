import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import pandas as pd
import config
import time
import gc

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ✅ Use shared API object — no session param
api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_API_SECRET,
    config.BASE_URL,
    api_version='v2'
)


def fetch_5min_data(api, ticker, days_back=30):
    """
    Fetch 5-minute bar data from Alpaca for the last 'days_back' days using SIP feed.
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    end_str = end_date.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    start_str = start_date.replace(microsecond=0).isoformat().replace('+00:00', 'Z')

    print(f"Fetching 5-minute SIP bars for {ticker} from {start_str} to {end_str}")
    bars = api.get_bars(ticker, timeframe='5Min', start=start_str, end=end_str, feed='sip')
    df = bars.df

    return df


def fetch_with_retries(api, ticker, days_back=30, max_retries=3, delay=3):
    for attempt in range(max_retries):
        try:
            return fetch_5min_data(api, ticker, days_back)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(delay * (attempt + 1))
    print(f"❌ Skipping {ticker} after {max_retries} attempts")
    return pd.DataFrame()


def filter_rth_only(df):
    df.index = df.index.tz_convert('US/Eastern')
    return df.between_time("09:30", "16:00").copy()


def main():
    #input_file = "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/Statistical Ratios.csv"
    input_file = "LargeCap.csv"
    # input_file = "/Users/jimutmukhopadhyay/Dummy Trading/AlpacaTrading/filtered_tickers_All.xlsx"
    # worksheet = "Combined Selected Tickers"
    #worksheet = "Sheet1"
    tickers_df = pd.read_csv(input_file)
    #tickers_df = pd.read_excel(input_file, sheet_name=worksheet)
    #tickers_df = tickers_df.head(200)
    output_dir = "ALPACA_DATA"
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tickers_df.iterrows():
        ticker = str(row['Ticker']).strip()

        df = fetch_with_retries(api, ticker, days_back=7)

        if df.empty:
            print(f"No data for {ticker}")
            continue

        df.rename(
            columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
            inplace=True
        )

        df_rth = filter_rth_only(df)

        if df_rth.empty:
            print(f"No RTH data for {ticker}.")
            continue

        df_rth.index = df_rth.index.tz_convert('Europe/Berlin')
        df_rth.loc[:, 'Date'] = df_rth.index.strftime('%Y%m%d %H:%M:%S')
        df_rth = df_rth[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        csv_path = os.path.join(output_dir, f"{ticker}.csv")
        df_rth.to_csv(csv_path, index=False)
        #print(f"✅ Saved {ticker} data to {csv_path}")

        # 🔄 Clean up memory every 10 tickers
        del df, df_rth
        if idx % 10 == 0:
            gc.collect()


if __name__ == "__main__":
    main()
