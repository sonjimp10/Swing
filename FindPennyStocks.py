import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import alpaca_trade_api as tradeapi
from tqdm import tqdm
import config  # Your config.py with API keys

# Step 1: Connect to Alpaca API
api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_API_SECRET,
    config.BASE_URL,
    api_version='v2'
)

# Step 2: Get all NASDAQ and NYSE tradable tickers
def get_all_nasdaq_nyse_tickers():
    assets = api.list_assets(status='active')
    return [
        asset.symbol for asset in assets
        if asset.exchange in ['NASDAQ', 'NYSE'] and asset.tradable and asset.symbol.isalpha()
    ]

# Step 3: Fetch 1-day historical bars using SIP feed
def fetch_1day_data_sip(api, tickers, days_back=45):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    end_str = end_date.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    start_str = start_date.replace(microsecond=0).isoformat().replace('+00:00', 'Z')

    print(f"Fetching 1-day SIP bars from {start_str} to {end_str}")
    return api.get_bars(tickers, timeframe='1Day', start=start_str, end=end_str, feed='sip').df

# Step 4: Calculate ATR
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# Step 5: Analyze tickers using Alpaca historical price data
def analyze_tickers_alpaca(tickers):
    data = []
    chunk_size = 100
    os.makedirs("price_data", exist_ok=True)

    for i in tqdm(range(0, len(tickers), chunk_size)):
        chunk = tickers[i:i+chunk_size]
        try:
            bars = fetch_1day_data_sip(api, chunk)

            if bars.empty:
                continue

            if 'symbol' not in bars.columns:
                if len(chunk) == 1:
                    bars['symbol'] = chunk[0]
                else:
                    print(f"Error in chunk {i}-{i+chunk_size}: 'symbol' column missing")
                    continue

            grouped = bars.groupby('symbol')
            for symbol, df in grouped:
                if len(df) < 20:
                    continue

                df.to_csv(f"price_data/{symbol}.csv")  # Save raw data for inspection

                start = df['close'].iloc[0]
                end = df['close'].iloc[-1]
                one_month_return = ((end - start) / start) * 100

                last_price = df['close'].iloc[-1]
                avg_volume = df['volume'].mean()

                atr_series = calculate_atr(df)
                atr_max = atr_series.max()
                atr_min = atr_series.min()
                atr_avg = (atr_max + atr_min) / 2
                atr_percent = atr_avg / last_price if last_price else 0

                print(f"Checked {symbol} | Price: {last_price}, Avg Vol: {avg_volume}, ATR Max: {atr_max}, ATR Min: {atr_min}")

                if last_price > 1 and last_price < 20:
                    if avg_volume > 1000000: #and avg_volume < 1000000:
                        data.append({
                            "Ticker": symbol,
                            "Price": round(last_price, 2),
                            "1M Return (%)": round(one_month_return, 2),
                            "Volume": int(avg_volume),
                            "ATR Max": round(atr_max, 4),
                            "ATR Min": round(atr_min, 4),
                            "ATR Avg": round(atr_avg, 4),
                            "ATR % of Price": round(atr_percent * 100, 2)
                        })
                        print(f"Selected {symbol}: Price={last_price}, Return={round(one_month_return, 2)}%")
                    else:
                        print(f"Skipped {symbol}: Low volume {avg_volume}")
                else:
                    print(f"Skipped {symbol}: Price too high {last_price}")
        except Exception as e:
            print(f"Error in chunk {i}-{i+chunk_size}: {e}")
    return pd.DataFrame(data)

# Run analysis
def get_penny_list():
    print("Fetching tradable tickers from Alpaca…")
    tickers = get_all_nasdaq_nyse_tickers()
    print(f"Tickers retrieved: {len(tickers)}")

    print("Analyzing price data and storing outputs…")
    df = analyze_tickers_alpaca(tickers)

    # apply your post‐filters here if desired, e.g. Price <2 & Return >20
    #df_filtered = df[(df["Price"] > 100) & (df["1M Return (%)"] > 0)]
    # sort by ATR % of Price descending and take top 10
    df_filtered = df.sort_values("ATR % of Price", ascending=False)
    return df_filtered

if __name__ == "__main__":
    df = get_penny_list()
    if not df.empty:
        df.to_excel("NASDAQTickerListSC1.xlsx", index=False)
        print("Saved to NasdaQTickerList.xlsx")
    else:
        print("No valid penny stocks found.")