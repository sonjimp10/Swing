import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import time  # this is the time class
import yfinance as yf
import dash
import subprocess
import pytz
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from BuySellSignalsDaily import add_buy_signals, add_sell_signals, add_macd_crossover_state, add_macd_crossover_state_for_sell  # Import the functions
from SupportResistanceLevels import dynamic_support_resistance, support, resistance, adjust_window_size
from uptrick_indicator import uptrick
# Set display options to show all rows
pd.set_option('display.max_rows', None)
os.chdir("/Users/jimutmukhopadhyay/Dummy Trading/AlpacaTrading/")

# Step 1: Download ticker data and save to CSV
def download_and_save_data(tickers, interval, period, save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for ticker in tickers:
        try:
            yf_ticker = ticker.replace(".", "-")
            print(f"Downloading 5-minute data for {ticker} (yfinance symbol: {yf_ticker})...")
            df = yf.download(yf_ticker, interval=interval, period=period)
            if df.empty:
                print(f"No data available for {ticker}. Skipping.")
                continue
            df.reset_index(inplace=True)  # Ensure index is reset to include the timestamp
            file_path = os.path.join(save_dir, f"{ticker}.csv")
            df.to_csv(file_path, index=False)
            print(f"Data for {ticker} saved to {file_path}.")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}. Skipping.")

# Step 2: Load tickers from file
def load_tickers(excel_file, worksheet):
    try:
        # Read the Excel worksheet
        df = pd.read_excel(excel_file, sheet_name=worksheet)
        
        # Drop rows where either 'Ticker' or 'Spread percentage' is missing
        df = df[['Ticker', 'TBUSpreadPercentage','Avg High Low','RSI_Buy_Threshold','RSI_Sell_Threshold']].dropna()
        # Filter for TSLA only
        #df = df[df['Ticker'] == 'TSLA']
        #print(f"DataFrame content before creating dictionary:\n{df.head()}")
        # Convert the DataFrame to a dictionary mapping tickers to Spread Percentage
        ticker_spread_dict = dict(zip(df['Ticker'], zip(df['TBUSpreadPercentage'],df['Avg High Low'])))
        #print(f"Ticker Spread Dictionary: {ticker_spread_dict}")
        return ticker_spread_dict
    except Exception as e:
        print(f"Error loading tickers and Spread Percentage: {e}")
        return {}


# Step 3: Load historical data from saved CSVs
#def load_data(tickers, save_dir="IBKR_DATA", macd_thresholds=None):
def load_data(ticker_spread_dict, save_dir="/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ALPACA_DAILY_DATA", macd_thresholds=None):
    data = {}
    signal_log = []  # To store signal information
    for ticker, (spread_percentage, Avg_HL_Sprd) in ticker_spread_dict.items():
        #print(f"Processing Ticker: {ticker}, Spread Percentage: {spread_percentage}, Avg_HL_Sprd: {Avg_HL_Sprd}")
        try:
            file_path = os.path.join(save_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                print(f"File for {ticker} not found. Skipping.")
                continue
            df = pd.read_csv(file_path)
            # Rename the 'Date' column to 'Datetime'
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
            # print(f"Loaded data for {ticker}: {df.shape} rows and columns: {df.columns.tolist()}")  # Debug
            # print("Head Data for Ticker", ticker,"\n",df.head(),"\n")

            # Ensure required numeric columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaNs
            df.dropna(subset=required_columns, inplace=True)
            if df.empty:
                print(f"Data for {ticker} is empty after cleaning. Skipping.")
                continue

            if 'Datetime' not in df.columns:
                print(f"Missing 'Datetime' column in data for {ticker}. Skipping.")
                continue
            #df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)

            # Ensure 'Datetime' is in datetime format
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

            # Drop any rows with invalid datetime values
            df = df.dropna(subset=['Datetime'])

            # Check if Datetime is already tz-aware
            # If naive (no tz), assume it was Berlin time and convert to New York.
            if df['Datetime'].dt.tz is None:
                df['Datetime'] = (
                    df['Datetime']
                    .dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='NaT')
                    .dt.tz_convert('America/New_York')
                )
            else:
                df['Datetime'] = df['Datetime'].dt.tz_convert('America/New_York')

            df = df.copy()

            # if df['Datetime'].dt.tz is None:
            #     df['Datetime'] = df['Datetime'].dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='NaT').dt.tz_convert('America/New_York')
            # else:
            #     df['Datetime'] = df['Datetime'].dt.tz_convert('America/New_York')
            # df = df[
            #     (df['Datetime'].dt.time >= datetime.strptime("09:30", "%H:%M").time()) &
            #     (df['Datetime'].dt.time <= datetime.strptime("16:00", "%H:%M").time())
            # ].copy()

            df = add_indicators(df)
            # Debug indicators
            #print(f"Added indicators for {ticker}. Columns now include: {df.columns.tolist()}")
            # Step 2: Calculate MACD Crossover State
            df = add_macd_crossover_state(df)  # Ensure this is called before add_buy_signals
            df = add_macd_crossover_state_for_sell(df)
            # Calculate Fibonacci retracement levels
            high = df['High'].max()
            low = df['Low'].min()
            fib_levels = calculate_fibonacci_retracement(high, low)

            # Add buy signals with MACD thresholds
            thresholds = macd_thresholds.get(ticker, {}) if macd_thresholds else {}
            # Calculate support and resistance with default n1=8 and n2=6
            df = dynamic_support_resistance(df, window=10, n1=8, n2=6)
            # Debug: Print the last 20 rows to check calculated levels
            #print(df[['Datetime', 'Low', 'High', 'Dynamic_Support', 'Dynamic_Resistance']].dropna(subset=['Dynamic_Support', 'Dynamic_Resistance']).tail(20))
            df, last_buy_price = add_buy_signals(df, ticker_name=ticker, fib_levels=fib_levels, macd_thresholds=thresholds, signal_log=signal_log)
            # df = add_sell_signals(df, ticker_name=ticker, fib_levels=fib_levels, last_buy_price=last_buy_price, 
            #                       macd_thresholds=thresholds, signal_log=signal_log)
            df = add_sell_signals(df, ticker_name=ticker, 
                                  spread_percentage=spread_percentage,
                                  Avg_HL_Sprd=Avg_HL_Sprd,
                                  fib_levels=fib_levels, last_buy_price=last_buy_price, 
                                  macd_thresholds=thresholds, signal_log=signal_log)
            # Save signal log to Excel
            signals_df = pd.DataFrame(signal_log)
            # 2) Immediately inspect what columns you’ve got
            #print(f"Columns in signals_df: {signals_df.columns.tolist()}")
            #signals_df['Date Time'] = pd.to_datetime(signals_df['Date Time'], errors='coerce')
            # ► Only convert to datetime if "Date Time" is actually a column
            if 'Date Time' in signals_df.columns:
                signals_df['Date Time'] = pd.to_datetime(signals_df['Date Time'], errors='coerce')
            else:
                # No "Date Time" column → keep signals_df as-is (empty or missing headers)
                # Optionally, create the column so downstream code sees it:
                signals_df['Date Time'] = pd.NaT
           # 3) If there is no 'Ticker' column, skip everything that relies on signals_df
            if 'Ticker' not in signals_df.columns:
                print(f"No 'Ticker' column in signals_df for {ticker}. Skipping signal processing.")
                #—IMPORTANT—stop processing this ticker’s signals entirely:
                continue

            # Only reached if 'Ticker' is present:
            ticker_order = signals_df['Ticker'].unique()

            # 4) Ensure all other expected signal columns exist (fill with NaN if missing)
            for col in ['Buy Signal', 'Buy High price', 'Buy Low', 'Buy RSI', 'bars_to_zero', 'VWAP Condition']:
                if col not in signals_df.columns:
                    signals_df[col] = pd.NA

            # … now proceed with everything that uses ticker_order and the other columns …
            for t in ticker_order:
                t_signals = signals_df[signals_df['Ticker'] == t]
            # Remove timezone information before saving
            #signals_df['Date Time'] = signals_df['Date Time'].dt.tz_localize(None)
            signals_df['Signal Type'] = signals_df.apply(lambda row: 'Buy' if row['Buy Signal'] else 'Sell', axis=1)
            # Ensure tickers follow the original order from the input file
            ticker_order = signals_df['Ticker'].unique()
            signals_df['Ticker'] = pd.Categorical(signals_df['Ticker'], categories=ticker_order, ordered=True)
            # Sort by Ticker (preserves original order), Date Time, and Signal Type
            #signals_df = signals_df.sort_values(by=['Ticker', 'Date Time', 'Signal Type'], ascending=[True, True, False])
            signals_df = signals_df.sort_values(by=['Ticker','Date Time'], ascending=[True,True])
            #signals_df = signals_df.sort_values(by=['Ticker', 'Date Time', 'Signal Type'], ascending=[True, True, False])
            #signals_df.to_excel("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/BuySellSignals.xlsx", index=False)
            file_path = "/Users/jimutmukhopadhyay/Dummy Trading/AlpacaTrading/BuySellSignalsDaily.xlsx"
            # Step 2: Save the new file
            signals_df.to_excel(file_path, index=False)
            #print(f"New file created successfully: {file_path}")
            print("Signals logged and saved to 'BuySellSignals.xlsx' for Ticker - ", ticker)
            # Verify final DataFrame structure
            #print(f"Final processed data for {ticker}: {df.shape} rows, {df.columns.tolist()}")

            #print(f"Data for {ticker} after adding Buy_Signal:\n{df.head(100)}\n")
            data[ticker] = df
            #print(f"Successfully processed {ticker}: {df.shape[0]} rows.")  # Debug
            #print("Data DF is \n", data.head(100))
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    return data

# Step 4: Add indicators (ATR, RSI, MACD, Volume Average, Bollinger Bands)
def add_indicators(df):
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal'], df['Short_EMA'], df['Long_EMA'] = calculate_macd(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    df['Volume_Avg'] = df['Volume'].rolling(window=14).mean()
    df['Middle_BB'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['Middle_BB'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_BB'] = df['Middle_BB'] - 2 * df['Close'].rolling(window=20).std()
    df = calculate_vwap(df)
    df = calculate_impulse_macd(df)  # Add Impulse MACD
    # Add the 50 EMA for intraday signal generation
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df = calculate_heiken_ashi(df)
    df['TSI'] = calculate_tsi(df['Close'])
    df['Williams_%R'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
    uptrick_df = uptrick(
        df.rename(columns={'High':'high','Low':'low','Close':'close'}),
        alma_window=9,
        alma_offset=0.85,
        alma_sigma=6.0,
        ema_window=9,
        rsi_window=14,
        atr_window=14,
        zscore_window_price=20,
        zscore_window_rsi=20,
        atr_multiplier=1.0,
        zscore_threshold_price=2.0,
        zscore_threshold_rsi=2.0
    )
    # merge back only the columns you need
    df['CompositeMA']  = uptrick_df['CompositeMA']
    df['UpperBand']    = uptrick_df['UpperBand']
    df['LowerBand']    = uptrick_df['LowerBand']
    df['UT_Signal']    = uptrick_df['Signal']

    df = calculate_cmf(df, period=20)
    # —— compute the “price‐scaled” CMF for a 5‐bar lookback ——
    n = 5
    low_n  = df['Low'].rolling(window=n).min()
    high_n = df['High'].rolling(window=n).max()

    # CMF is between -1 and +1, so (CMF + 1)/2 brings it to 0…1
    df['CMF_Scaled'] = low_n + ((df['CMF'] + 1) / 2) * (high_n - low_n)
    # ————————————————————————————————————————————————————
    df = calculate_mfi(df, window=14)
    # ——— NEW: compute the “price-scaled” MFI for a 5-bar lookback ———
    n = 5
    low_n   = df['Low'].rolling(window=n).min()
    high_n  = df['High'].rolling(window=n).max()
    df['MFI_Scaled'] = low_n + (df['MFI'] / 100) * (high_n - low_n)
    # ————————————————————————————————————————————————————————————
    df = calculate_obv(df)
    df = calculate_vroc(df, window=14)
    return df

# def calculate_cmf(df, period=20):
#     mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
#     mfv *= df['Volume']
#     df['CMF'] = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum()
#     return df

def calculate_cmf(df, period=20):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfv.fillna(0)  # treat zero-range bars as zero money flow
    mfv *= df['Volume']
    num = mfv.rolling(window=period, min_periods=1).sum()
    den = df['Volume'].rolling(window=period, min_periods=1).sum()
    df['CMF'] = num / den
    return df

def calculate_mfi(df, window=14):
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    delta = typical.diff()
    up = (delta.where(delta>0, 0) * df['Volume']).ewm(span=window).mean()
    down = (-delta.where(delta<0, 0) * df['Volume']).ewm(span=window).mean()
    df['MFI'] = 100 * up / (up + down)
    return df

def calculate_obv(df):
    direction = np.where(df['Close'] > df['Close'].shift(), 1, 
                 np.where(df['Close'] < df['Close'].shift(), -1, 0))
    df['OBV'] = (direction * df['Volume']).cumsum()
    df['OBV_Slope'] = df['OBV'].diff().rolling(window=5).mean()
    return df

def calculate_vroc(df, window=14):
    df['VROC'] = df['Volume'].pct_change(window)
    return df

# ATR calculation
def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calculate_tsi(close, r=25, s=13):
    diff = close.diff()
    abs_diff = diff.abs()

    double_smoothed_diff = diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    double_smoothed_abs = abs_diff.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()

    tsi = 100 * (double_smoothed_diff / double_smoothed_abs)
    return tsi


def calculate_williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


# RSI calculation
def calculate_rsi(series, period=14):
    """
    Calculate RSI using Wilder's smoothing method (commonly used in most charting tools).
    Keeps the same function name, so no other code changes needed.
    """
    delta = series.diff()

    # Separate positive and negative moves
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use Wilder's smoothing (an EMA with alpha = 1/period)
    # min_periods=period ensures we don't get weird initial NaNs before we have enough data
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and then RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    """
    Typical MACD (12, 26, 9). If you prefer faster signals, you can still pass in (5, 13, 6).
    Keeps the same function name to avoid code changes elsewhere.
    """
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal, short_ema, long_ema

# Heiken Ashi computation
def calculate_heiken_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha.loc[:, 'HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # Initialize HA_Open properly using .loc to avoid chained assignment
    ha.loc[ha.index[0], 'HA_Open'] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(ha)):
        cur_idx = ha.index[i]
        prev_idx = ha.index[i - 1]
        ha.loc[cur_idx, 'HA_Open'] = (ha.loc[prev_idx, 'HA_Open'] + ha.loc[prev_idx, 'HA_Close']) / 2
    ha['HA_High'] = ha[['HA_Open', 'HA_Close']].join(df['High']).max(axis=1)
    ha['HA_Low'] = ha[['HA_Open', 'HA_Close']].join(df['Low']).min(axis=1)
    df = df.join(ha)
    # Heiken Ashi EMAs and SMA [generally taken 10 and 30 ema]
    df.loc[:, 'HA_EMA10'] = df['HA_Close'].ewm(span=10, adjust=False).mean()
    df.loc[:, 'HA_EMA30'] = df['HA_Close'].ewm(span=30, adjust=False).mean()
    df.loc[:, 'HA_SMA200'] = df['HA_Close'].rolling(window=200).mean()
    return df

# VWAP calculation
# def calculate_vwap(df, window=20):
#     typical_price = (df['High'] + df['Low'] + df['Close']) / 3
#     df['VWAP'] = (typical_price * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
#     df['Upper_VWAP'] = df['VWAP'] + 2 * df['Close'].rolling(window=20).std()  # Example threshold
#     df['Lower_VWAP'] = df['VWAP'] - 2 * df['Close'].rolling(window=20).std()  # Example threshold
#     return df

def calculate_vwap(df, window=20):
    """
    Calculates intraday (cumulative) VWAP and upper/lower bands for a DataFrame.
    
    - Typical Price = (High + Low + Close) / 3.
    - VWAP is computed as the cumulative sum of (Typical Price × Volume) divided by the cumulative Volume, resetting each day.
    - Upper_VWAP and Lower_VWAP are computed as VWAP ± 2 times the daily standard deviation of the Typical Price.
    
    This function assumes that the DataFrame contains a 'Datetime' column.
    """
    # Ensure Datetime is a datetime type and sort the DataFrame
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Calculate Typical Price
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    # Extract date (assuming one trading session per day)
    df['Date'] = df['Datetime'].dt.date

    # Prepare output columns
    df['VWAP'] = 0.0
    df['Upper_VWAP'] = 0.0
    df['Lower_VWAP'] = 0.0

    output_df = []
    # Process each trading day separately
    for d, group in df.groupby('Date'):
        group = group.copy()
        # Compute cumulative totals
        group['Cum_TPV'] = (group['Typical'] * group['Volume']).cumsum()
        group['Cum_Vol'] = group['Volume'].cumsum()
        # Calculate cumulative VWAP
        group['VWAP'] = group['Cum_TPV'] / group['Cum_Vol']
        # Compute daily standard deviation for the typical price
        daily_std = group['Typical'].std()
        k = 2  # Number of standard deviations for the bands
        group['Upper_VWAP'] = group['VWAP'] + k * daily_std
        group['Lower_VWAP'] = group['VWAP'] - k * daily_std
        output_df.append(group)
    
    result_df = pd.concat(output_df).reset_index(drop=True)
    # Optionally remove helper columns so the DataFrame matches your original structure
    result_df = result_df.drop(columns=['Cum_TPV', 'Cum_Vol', 'Typical', 'Date'])
    
    return result_df

######################### IMPULSE MACD CALCULATION ##############################################
# Custom SMMA Calculation
def calc_smma(series, length):
    smma = series.ewm(alpha=1 / length, adjust=False).mean()
    return smma

# Custom ZLEMA Calculation
def calc_zlema(series, length):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    d = ema1 - ema2
    zlema = ema1 + d
    return zlema

# Impulse MACD Calculation
def calculate_impulse_macd(df, length_ma=34, length_signal=9):
    src = (df['High'] + df['Low'] + df['Close']) / 3  # HLC3
    hi = calc_smma(df['High'], length_ma)  # Smoothed high
    lo = calc_smma(df['Low'], length_ma)  # Smoothed low
    mi = calc_zlema(src, length_ma)  # ZLEMA of HLC3

    # Impulse MACD Logic
    df['Impulse_MD'] = np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0))
    df['Impulse_Signal'] = df['Impulse_MD'].rolling(window=length_signal).mean()
    df['Impulse_Histogram'] = df['Impulse_MD'] - df['Impulse_Signal']

    # Color Coding
    df['Impulse_Color'] = np.where(
        src > mi,
        np.where(src > hi, 'lime', 'green'),
        np.where(src < lo, 'red', 'orange')
    )

    return df
######################### END OF IMPULSE MACD CALCULATION #######################################

######################### Calculate Fibonacci Retracement Levels ################################
def calculate_fibonacci_retracement(high, low):
    """
    Calculate Fibonacci retracement levels for a given high and low price.
    """
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * (high - low),
        "38.2%": high - 0.382 * (high - low),
        "50.0%": high - 0.5 * (high - low),
        "61.8%": high - 0.618 * (high - low),
        "100.0%": low
    }
    return levels
######################### End Of Fibonacci Retracement Calculation ################################
# Dash app setup
app = dash.Dash(__name__)
app.title = "Enhanced TradingView-Like Chart Viewer"

# Main execution
def main():
    # File containing tickers
    excel_file = "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/Short/Unique_Tickers.xlsx"
    worksheet = "Sheet1"

    # Define interval and period for 5-minute data
    interval = "5m"
    period = "1mo"

    # Read MACD thresholds from Excel
    common_signals_df = pd.read_excel(excel_file, sheet_name=worksheet)

    # Load tickers
    ticker_spread_dict = load_tickers(excel_file, worksheet)
    if not ticker_spread_dict:
        print("No tickers found or Spread Percentage missing. Exiting.")
        return
    tickers = list(ticker_spread_dict.keys())
    if not tickers:
        print("No tickers found. Exiting.")
        return
    
    ticker_spread_dict = {k: ticker_spread_dict[k] for k in list(ticker_spread_dict.keys())[:650]}
    # new: last 2
    #ticker_spread_dict = {k: ticker_spread_dict[k] for k in list(ticker_spread_dict.keys())[-2:]}
    #print("Ticker_Spread is -- ** ", ticker_spread_dict)

    macd_thresholds_dict = {}
    for ticker in tickers:
        macd_data = common_signals_df[common_signals_df["Ticker"] == ticker]
        if not macd_data.empty:
            macd_thresholds_dict[ticker] = {
                "MACD_Bullish_Mean": macd_data["MACD Bullish Mean"].values[0],
                "Average_MACD_Bullish": macd_data["Average MACD Bullish"].values[0],
                "Impulse_Bullish_Mean": macd_data["Impulse Bullish Mean"].values[0],  # Include Impulse_Bullish_Mean
                "MACD_Bearish_Mean": macd_data["MACD Bearish Mean"].values[0],
                "Average_MACD_Bearish": macd_data["Average MACD Bearish"].values[0],
                "Spread_Percentage": macd_data["TBUSpreadPercentage"].values[0],  # Include Spread Percentage
                "Avg_HL_Sprd":macd_data['Avg High Low'].values[0],
                "RSI Buy":macd_data['RSI_Buy_Threshold'].values[0],
                "RSI Sell":macd_data['RSI_Sell_Threshold'].values[0]
            }

    # Download data for tickers and save to CSV
    #download_and_save_data(tickers, interval, period)

    # Load data from CSVs
    data = load_data(ticker_spread_dict=ticker_spread_dict, macd_thresholds=macd_thresholds_dict)
    #print("Data is - ", data)
    if not data:
        print("No data loaded. Exiting.")
        return
        # Find Long and Short Candidates

    long_candidates = []
    short_candidates = []
    strong_buy_list = []
    strong_sell_list = []

    for ticker, df in data.items():
        df = df.sort_values("Datetime")
        if len(df) < 2:
            continue  # not enough data

        # Dynamically calculate the Histogram
        df['Histogram'] = df['MACD'] - df['Signal']

        # ---- New daily‐bar logic: directly compare the last two rows ----
        # Yesterday’s histogram = row at index -2, Today’s histogram = row at index -1
        hist_prev = df['Histogram'].iloc[-2]
        hist_curr = df['Histogram'].iloc[-1]

        # Yesterday’s MACD value and Today’s latest RSI & MFI
        macd_prev = df['MACD'].iloc[-2]
        latest    = df.iloc[-1]         # This is today’s full row
        rsi       = latest['RSI']
        mfi       = latest['MFI']
        # ------------------------------------------------------------------

        LongCondCross = (
            (rsi <= 40)
            and (hist_prev < 0)
            and (hist_curr > 0)
            and (macd_prev < 0)
            and (mfi < 60)
        )
        ContiniousLong = (
            (rsi <= 75)
            and (hist_prev > 0)
            and (hist_curr > 0)
            and (hist_curr > hist_prev)
            and (mfi < 80)
        )
        # # Check for Long
        if LongCondCross or ContiniousLong:
            # 6) Now apply the “last‐4‐bar average vs prior 30‐bar low” filter:
            if len(df) >= 6:
                # compute average close of last 4 bars
                avg_last4 = df['Close'].iloc[-1:].mean()
                # compute lowest Low of bars from index -34 up to (but not including) -4
                lowest_30_excl = df['Low'].iloc[-(1 + 5) : -1].min()
        
                # only append if avg_last4 is *greater than* that 30‐bar low
                if avg_last4 > lowest_30_excl:
                    long_candidates.append(ticker)
                else:
                    # skip this ticker, since its last‐4 average is <= the prior 30‐bar low
                    pass
            else:
                # fewer than 34 total bars → skip the extra check and append anyway
                long_candidates.append(ticker)
        # if LongCondCross or ContiniousLong:
        #     long_candidates.append(ticker)

        ShortCondCross = (
            (rsi >= 50)
            and (hist_prev > 0)
            and (hist_curr < 0)
            and (hist_prev > hist_curr)
            and (macd_prev > 0)
            and (mfi > 50)
        )
        ContiniousShort = (
            (rsi >= 50)
            and (hist_prev < 0)
            and (hist_curr < 0)
            and (hist_curr < hist_prev)
            and (mfi > 50)
        )
        # # Check for Short
        # apply the “last 4‐bar avg must be the MIN of prior 30 bars” filter:
        if ShortCondCross or ContiniousShort:
            if len(df) >= 6:
                # average close of last 4 bars
                avg_last4 = df['Close'].iloc[-1:].mean()
                # lowest Low of bars [−34 : −4), i.e. 30 bars immediately before the last 4
                highest_7_excl = df['Low'].iloc[-(1 + 5) : -1].max()
        
                # append only if avg_last4 is truly ≤ that prior 30‐bar low
                if avg_last4 <= highest_7_excl:
                    short_candidates.append(ticker)
                else:
                    # skip; last‐4 average is not low enough
                    pass
            else:
                # fewer than 34 bars → skip the extra check and append anyway
                short_candidates.append(ticker)
        # if ShortCondCross or ContiniousShort:
        #     short_candidates.append(ticker)

        if (
            (hist_curr > hist_prev)
            and (hist_curr < 0)
            and (hist_prev < 0)
            and (rsi < 50)
            and (latest['MACD'] < 0)
        ):
            strong_buy_list.append(ticker)

        # For strong sells: today’s hist < yesterday’s hist, both positive, RSI > 60, MFI > 80, MACD > 0
        if (
            (hist_curr < hist_prev)
            and (hist_curr > 0)
            and (hist_prev > 0)
            and (rsi > 60)
            and (mfi > 80)
            and (latest['MACD'] > 0)
        ):
            strong_sell_list.append(ticker)

    # Save to Excel
    long_df = pd.DataFrame({'Ticker': long_candidates})
    short_df = pd.DataFrame({'Ticker': short_candidates})


    # Add "Strong Buy" column to long_df
    long_df['Strong Buy'] = long_df['Ticker'].apply(lambda x: 'Yes' if x in strong_buy_list else '')
    # Add "Strong Sell" column to short_df
    short_df['Strong Sell'] = short_df['Ticker'].apply(lambda x: 'Yes' if x in strong_sell_list else '')

    # Path to ATR file
    atr_file = "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/ATR_Results.xlsx"
    atr_df = pd.read_excel(atr_file)

    # Calculate final Avg ATR
    atr_df['Avg ATR'] = (atr_df['ATR_Max'] + atr_df['ATR_Min']) / 2

    # Calculate Profit Percentage
    atr_df['Profit Percentage'] = (atr_df['Avg ATR'] / atr_df['Closing Price']) * 100

    # Merge ATR data with long and short dataframes
    long_merged = pd.merge(long_df, atr_df, on="Ticker", how="left")
    short_merged = pd.merge(short_df, atr_df, on="Ticker", how="left")

    # Sort by Profit Percentage descending
    long_merged = long_merged.sort_values(by='Profit Percentage', ascending=False)
    short_merged = short_merged.sort_values(by='Profit Percentage', ascending=False)

    output_file = 'long_short_candidates.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        long_merged.to_excel(writer, sheet_name='Long', index=False)
        short_merged.to_excel(writer, sheet_name='Short', index=False)

    print(f"Long and Short candidates saved to {output_file}")

    print(f"Data successfully loaded for the following tickers: {list(data.keys())}")

    # Dash Layout
    app.layout = html.Div([
        html.H1("Enhanced TradingView-Like Chart Viewer", style={"textAlign": "center"}),

        html.Div([
            html.Label("Select a Ticker:", style={"margin-right": "10px"}),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": ticker, "value": ticker} for ticker in data.keys()],
                value=list(data.keys())[0],
                style={"width": "300px", "display": "inline-block", "margin-right": "20px"}
            ),

            html.Label("Select Period:", style={"margin-right": "10px"}),
            dcc.Dropdown(
                id="period-dropdown",
                options=[
                    {"label": "1 Week", "value": "1w"},
                    {"label": "2 Weeks", "value": "2w"},
                    {"label": "1 Month", "value": "1m"},
                    {"label": "3 Months", "value": "3m"},
                    {"label": "6 Months", "value": "6m"}
                ],
                value="1m",
                style={"width": "200px", "display": "inline-block", "margin-right": "20px"}
            ),

            html.Label("Select Indicator:", style={"margin-right": "10px"}),
            dcc.Dropdown(
                id="indicator-dropdown",
                options=[
                    {"label": "Bollinger Bands", "value": "BB"},
                    {"label": "VWAP", "value": "VWAP"},
                    {"label": "Heiken Ashi EMAs", "value": "HA"},
                    {"label": "Uptrick Z-Score", "value": "UPTRICK"},
                    {"label": "Chaikin Money Flow", "value": "CMF"},
                    {"label": "Money Flow Index Scaled", "value": "MFI_Scaled"},
                    {"label": "CMF Scaled", "value": "CMF_Scaled"},
                    {"label": "Money Flow Index", "value": "MFI"},
                    {"label": "On-Balance Volume", "value": "OBV"},
                    {"label": "Volume ROC", "value": "VROC"}
                ],
                value="BB",
                style={"width": "300px", "display": "inline-block"}
            )
        ], style={"display": "flex", "align-items": "center"}),

        dcc.Graph(id="candlestick-chart", style={"height": "500px"}, config={"editable": True}),
        dcc.Graph(id="rsi-chart", style={"height": "300px"}),
        dcc.Graph(id="macd-chart", style={"height": "300px"}),
        dcc.Graph(id="Impulse-macd-chart", style={"height": "300px"})
    ])


    # Callback to update the charts based on selected ticker and date
    @app.callback(
    [
        Output("candlestick-chart", "figure"),
        Output("rsi-chart", "figure"),
        Output("macd-chart", "figure"),
        Output("Impulse-macd-chart", "figure")
    ],
    [
        Input("ticker-dropdown", "value"),
        Input("period-dropdown", "value"),  # CHANGED from 'date-picker' to 'period-dropdown'
        Input("indicator-dropdown", "value")
    ]
    )
    def update_charts(selected_ticker, selected_period, selected_indicator):
        if selected_ticker not in data:
            print(f"Ticker {selected_ticker} not found in data.")
            return {}, {}, {}, {}

        df = data[selected_ticker]

        # Determine lookback period in days
        period_map = {
            "1w": 7,
            "2w": 14,
            "1m": 30,
            "3m": 90,
            "6m": 180
        }
        lookback_days = period_map.get(selected_period, 30)  # Default to 1 month

        # Filter data for selected period
        end_datetime = datetime.now(pytz.timezone('America/New_York'))
        start_datetime = end_datetime - timedelta(days=lookback_days)
        df_period = df[(df['Datetime'] >= start_datetime) & (df['Datetime'] <= end_datetime)].copy()


        # Debug: Check if the DataFrame has necessary columns
        required_columns = ["Datetime", "Buy_Signal", "Sell_Signal", "MACD", "Signal", "Close"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing column '{col}' in data for {selected_ticker}.")
                return {}, {}, {}, {}

        # Determine the time delta based on the selected period
        
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)

        if selected_period == "1w":
            start_datetime = now - timedelta(weeks=1)
        elif selected_period == "2w":
            start_datetime = now - timedelta(weeks=2)
        elif selected_period == "1m":
            start_datetime = now - timedelta(days=30)
        elif selected_period == "3m":
            start_datetime = now - timedelta(days=90)
        elif selected_period == "6m":
            start_datetime = now - timedelta(days=180)
        else:
            # Default to 1 month if unknown
            start_datetime = now - timedelta(days=30)

        end_datetime = now

        ny_tz = pytz.timezone('America/New_York')
        if df_period['Datetime'].dt.tz is None:
            df_period['Datetime'] = df_period['Datetime'].dt.tz_localize('America/New_York')
        else:
            df_period['Datetime'] = df_period['Datetime'].dt.tz_convert('America/New_York')
        # Filter to trading hours only: 9:30 AM - 4:00 PM NY time
        # df_period['Time'] = df_period['Datetime'].dt.time
        # market_open = time(9, 30)
        # market_close = time(16, 0)
        # df_filtered = df_period[(df_period['Time'] >= market_open) & (df_period['Time'] <= market_close)].copy()
        # df_filtered.drop(columns='Time', inplace=True)
        df_filtered = df_period.copy()
        # Filter the DataFrame for the selected period
        #df_filtered = df[(df['Datetime'] >= start_datetime) & (df['Datetime'] <= end_datetime)]
        if df_filtered.empty:
            print(f"Filtered DataFrame is empty for {selected_ticker} in the last {selected_period}.")
            return {}, {}, {}, {}

        # Filter for Buy and Sell signals
        buy_signals = df_filtered[df_filtered["Buy_Signal"] == True]
        sell_signals = df_filtered[df_filtered["Sell_Signal"] == True]
        print(f"DF Filtered ticker {selected_ticker}\n", df_filtered.head(), "\n")
        print(f"Buy signals found: {len(buy_signals)}, Sell signals found: {len(sell_signals)} for {selected_ticker}.")

        # Create candlestick chart
        candlestick_fig = go.Figure()
        candlestick_fig.add_trace(
            go.Candlestick(
                x=df_filtered['Datetime'],
                open=df_filtered['Open'],
                high=df_filtered['High'],
                low=df_filtered['Low'],
                close=df_filtered['Close'],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red"
            )
        )

        # The rest of your chart code continues here...


        # Calculate 50-period Exponential Moving Average (EMA) if not already calculated.
        if 'EMA50' not in df_filtered.columns:
            df_filtered['EMA50'] = df_filtered['Close'].ewm(span=50, adjust=False).mean()

        # Add the 50-period EMA as a line plot.
        candlestick_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['EMA50'],
                mode='lines',
                name='50-Period EMA',
                line=dict(color='blue', width=2, dash='solid')
            )
        )

        # 2) Capture references to those first two traces
        price_candles = candlestick_fig.data[0]
        ema50_line   = candlestick_fig.data[1]

        # … your existing candlestick trace(s) …

        # now add CMF, MFI, OBV, VROC all on y2
        candlestick_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['CMF'],
                mode='lines',
                name='CMF',
                line=dict(color='magenta', width=1.5),  # custom color
                yaxis='y2'
            )
        )
        candlestick_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['MFI'],
                mode='lines',
                name='MFI',
                line=dict(color='cyan', width=1.5),
                yaxis='y2'
            )
        )
        candlestick_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['OBV'],
                mode='lines',
                name='OBV',
                line=dict(color='yellow', width=1.5),
                yaxis='y2'
            )
        )
        candlestick_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['VROC'],
                mode='lines',
                name='VROC',
                line=dict(color='lime', width=1.5),
                yaxis='y2'
            )
        )

        candlestick_fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Indicator",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            xaxis=dict(title="Datetime", rangeslider=dict(visible=True), type="date"),
            template="plotly_dark",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
)


        candlestick_fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Indicator",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            xaxis=dict(title="Datetime", rangeslider=dict(visible=True), type="date"),
            template="plotly_dark",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )


        # configure your axes
        candlestick_fig.update_layout(
            title=f"Price + Volume/Momentum Indicators for {selected_ticker}",
            xaxis=dict(type='date', rangeslider=dict(visible=True)),
            yaxis=dict(title="Price"),
            yaxis2=dict(
                title="Volume / Momentum",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            template="plotly_dark",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
                    
        #buy_signals = df_filtered[df_filtered['Buy_Signal']]
        # Add buy signal markers if they exist
        if not buy_signals.empty:
            candlestick_fig.add_trace(go.Scatter(
            x=buy_signals['Datetime'],
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            name='Buy Signal'
        ))
            
        # Add sell signals (red triangles)
        if not sell_signals.empty:
            candlestick_fig.add_trace(go.Scatter(
            x=sell_signals['Datetime'],
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            name='Sell Signal'
        ))

        buy_trace  = candlestick_fig.data[-2]
        sell_trace = candlestick_fig.data[-1]

        if selected_indicator == "BB":
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Middle_BB'], mode='lines', name='Middle BB', line=dict(color='blue', dash='dot')))
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Upper_BB'], mode='lines', name='Upper BB', line=dict(color='blue', dash='solid')))
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Lower_BB'], mode='lines', name='Lower BB', line=dict(color='blue', dash='solid')))
            candlestick_fig.add_traces([
            go.Scatter(
                x=pd.concat([df_filtered['Datetime'], df_filtered['Datetime'][::-1]]),
                y=pd.concat([df_filtered['Upper_BB'], df_filtered['Lower_BB'][::-1]]),
                fill='toself',
                fillcolor='rgba(135, 206, 250, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='BB Area',
                showlegend=False
            )
        ])
        elif selected_indicator == "VWAP":
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', dash='solid')))
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Upper_VWAP'], mode='lines', name='Upper VWAP', line=dict(color='orange', dash='dot')))
            candlestick_fig.add_trace(go.Scatter(x=df_filtered['Datetime'], y=df_filtered['Lower_VWAP'], mode='lines', name='Lower VWAP', line=dict(color='orange', dash='dot')))
            candlestick_fig.add_traces([
            go.Scatter(
                x=pd.concat([df_filtered['Datetime'], df_filtered['Datetime'][::-1]]),
                y=pd.concat([df_filtered['Upper_VWAP'], df_filtered['Lower_VWAP'][::-1]]),
                fill='toself',
                fillcolor='rgba(135, 206, 250, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='VWAP Area',
                showlegend=False
            )
            ])
        elif selected_indicator == "HA":
            # EMA10
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['HA_EMA10'],
                    mode='lines',
                    name='HA EMA10',
                    line=dict(color='magenta', dash='solid')
                )
            )
            # EMA30
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['HA_EMA30'],
                    mode='lines',
                    name='HA EMA30',
                    line=dict(color='orange', dash='dot')
                )
            )
            # Fill area between EMA10 & EMA30
            candlestick_fig.add_trace(
                go.Scatter(
                    x=pd.concat([df_filtered['Datetime'], df_filtered['Datetime'][::-1]]),
                    y=pd.concat([df_filtered['HA_EMA10'], df_filtered['HA_EMA30'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(128, 0, 128, 0.1)',  # light purple
                    line=dict(color='rgba(255,255,255,0)'),
                    name='HA EMA Area',
                    showlegend=False
                )
            )

        if selected_indicator == "UPTRICK":
            # Composite Moving Average
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['CompositeMA'],
                    mode='lines',
                    name='Composite MA',
                    line=dict(color='cyan', dash='solid')
                )
            )
            # Upper & Lower Bands
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['UpperBand'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='cyan', dash='dot')
                )
            )
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['LowerBand'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='cyan', dash='dot')
                )
            )
            # Filled area between bands
            candlestick_fig.add_trace(
                go.Scatter(
                    x=pd.concat([df_filtered['Datetime'], df_filtered['Datetime'][::-1]]),
                    y=pd.concat([df_filtered['UpperBand'], df_filtered['LowerBand'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,255,255,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Band Area',
                    showlegend=False
                )
            )
            # Optionally, plot buy/sell markers
            buys = df_filtered[df_filtered['UT_Signal'] == 1]
            sells = df_filtered[df_filtered['UT_Signal'] == -1]
            candlestick_fig.add_trace(
                go.Scatter(
                    x=buys['Datetime'], y=buys['Close'],
                    mode='markers', marker_symbol='triangle-up',
                    marker_color='darkgreen', marker_size=10,
                    name='UT Buy'
                )
            )
            candlestick_fig.add_trace(
                go.Scatter(
                    x=sells['Datetime'], y=sells['Close'],
                    mode='markers', marker_symbol='triangle-down',
                    marker_color='darkmagenta', marker_size=10,
                    name='UT Sell'
                )
            )

            # Finally, update layout
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - Uptrick Z-Score",
                xaxis=dict(title="Datetime", rangeslider=dict(visible=True), type="date"),
                yaxis=dict(title="Price"),
                height=500, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,1)", plot_bgcolor="rgba(30,30,30,1)",
                font=dict(color="white"), margin=dict(l=50,r=50,t=50,b=50)
            )

        # ... somewhere in your callback after reading the dropdown ...
        if selected_indicator == "VROC":
            # 2) Reset the figure to just those two traces
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]

            # 3) Now add your VROC overlay on y2
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['VROC'],
                    mode='lines',
                    name='VROC',
                    line=dict(color='lime', width=2),
                    yaxis='y2'
                )
            )

            # 3) full layout override
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - VROC",
                template="plotly_dark",
                xaxis=dict(title="Datetime", type="date", rangeslider=dict(visible=True)),
                yaxis=dict(title="Price", side="left"),
                yaxis2=dict(
                    title="VROC",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    # automatic range to fit your data
                    autorange=True
                ),
                height=600, margin=dict(l=50, r=50, t=50, b=50)
            )
            
        elif selected_indicator == "CMF":
            # reset to only price / EMA50 / buy / sell
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]

            # add CMF
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['CMF'],
                    mode='lines',
                    name='CMF',
                    line=dict(color='magenta', width=2),
                    yaxis='y2'
                )
            )

            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - CMF",
                template="plotly_dark",
                xaxis=dict(title="Datetime", type="date", rangeslider=dict(visible=True)),
                yaxis=dict(title="Price", side="left"),
                yaxis2=dict(
                    title="CMF",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[-1,1]    # CMF always between -1 and +1
                ),
                height=600, margin=dict(l=50, r=50, t=50, b=50)
            )

        elif selected_indicator == "CMF_Scaled":
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]

            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['CMF_Scaled'],
                    mode='lines',
                    name='CMF_Scaled',
                    line=dict(color='white', width=2)
                )
            )

            # 3) Full layout override, including yaxis2 tailored to price scale
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - CMF (scaled)",
                template="plotly_dark",
                xaxis=dict(
                    title="Datetime",
                    type="date",
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    title="Price",
                    side="left",
                ),
                yaxis2=dict(
                    title="CMF Scaled (≈ Price)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    autorange=True,      # lets Plotly pick exactly the min/max of your MFI_Scaled
                    # optionally, you can force a padding around your expected price range:
                    # range=[df_filtered['MFI_Scaled'].min()*0.98, df_filtered['MFI_Scaled'].max()*1.02]
                ),
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
            )


        elif selected_indicator == "MFI_Scaled":
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]

            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['MFI_Scaled'],
                    mode='lines',
                    name='MFI_Scaled',
                    line=dict(color='cyan', width=2)
                )
            )

            # 3) Full layout override, including yaxis2 tailored to price scale
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - MFI (scaled)",
                template="plotly_dark",
                xaxis=dict(
                    title="Datetime",
                    type="date",
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    title="Price",
                    side="left",
                ),
                yaxis2=dict(
                    title="MFI Scaled (≈ Price)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    autorange=True,      # lets Plotly pick exactly the min/max of your MFI_Scaled
                    # optionally, you can force a padding around your expected price range:
                    # range=[df_filtered['MFI_Scaled'].min()*0.98, df_filtered['MFI_Scaled'].max()*1.02]
                ),
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
            )

        elif selected_indicator == "MFI":
            # reset to only price / EMA50 / buy / sell
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]

            # 1) Plot the SCALED MFI on y2
            candlestick_fig.add_trace(
                go.Scatter(
                    x=df_filtered['Datetime'],
                    y=df_filtered['MFI'],   # <-- use your price‐scaled MFI here
                    mode='lines',
                    name='MFI',
                    line=dict(color='cyan', width=2),
                    yaxis='y2'                     # <-- this is the key line
                )
            )

            # 2) Full layout override (including the new 0–100 right axis)
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - MFI",
                template="plotly_dark",
                xaxis=dict(
                    title="Datetime",
                    type="date",
                    rangeslider=dict(visible=True)
                ),
                yaxis=dict(
                    title="Price",
                    side="left",
                ),
                yaxis2=dict(
                    title="MFI (0-100)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    range=[0, 100],
                    autorange=False
                ),
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
            )

        
        elif selected_indicator == "OBV":
            candlestick_fig.data = [price_candles, ema50_line, buy_trace, sell_trace]
            candlestick_fig.add_trace(go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['OBV'],
                mode='lines',
                name='OBV',
                line=dict(color='yellow', width=2),
                yaxis='y2'
            ))
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - OBV",
                template="plotly_dark",
                xaxis=dict(title="Datetime", type="date", rangeslider=dict(visible=True)),
                yaxis=dict(title="Price", side="left"),
                yaxis2=dict(
                    title="OBV",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    autorange=True
                ),
                height=600, margin=dict(l=50, r=50, t=50, b=50)
            )

        # Update layout for Bollinger Bands
        if selected_indicator == "BB":
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - Bollinger Bands",
                xaxis=dict(
                title="Datetime",
                rangeslider=dict(visible=True),
                type="date"
                ),
                yaxis=dict(title="Price"),
                height=500,
                template="plotly_dark",  # Dark trading view template
                paper_bgcolor="rgba(0, 0, 0, 1)",  # Transparent paper background
                plot_bgcolor="rgba(30, 30, 30, 1)",  # Dark gray plot background for trading view style
                font=dict(color="white"),  # White font for readability
                margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins to prevent cutoff
                )

        # Update layout for VWAP
        elif selected_indicator == "VWAP":
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - VWAP",
                xaxis=dict(
                title="Datetime",
                rangeslider=dict(visible=True),
                type="date"
                ),
                yaxis=dict(title="Price"),
                height=500,
                template="plotly_dark",  # Dark trading view template
                paper_bgcolor="rgba(0, 0, 0, 1)",  # Transparent paper background
                plot_bgcolor="rgba(30, 30, 30, 1)",  # Dark gray plot background for trading view style
                font=dict(color="white"),  # White font for readability
                margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins to prevent cutoff
                )
            
                # Update layout for Heikin Ashi EMAs
        elif selected_indicator == "HA":
            candlestick_fig.update_layout(
                title=f"Price Chart for {selected_ticker} - Heikin Ashi EMAs",
                xaxis=dict(
                    title="Datetime",
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(title="Price"),
                height=500,
                template="plotly_dark",
                paper_bgcolor="rgba(0, 0, 0, 1)",
                plot_bgcolor="rgba(30, 30, 30, 1)",
                font=dict(color="white"),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Create RSI chart
        rsi_fig = go.Figure()
        rsi_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple'),
                showlegend=True  # Ensure legend visibility
            )
        )

        # Add hazy area for RSI thresholds
        rsi_fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=25,
            y1=70,
            fillcolor="rgba(135, 206, 250, 0.1)",  # Light blue transparent fill
            layer="below",
            line_width=0,
            name="Threshold Area"  # Name is not used in shapes
        )
        rsi_fig.update_layout(
            title=f"RSI Chart for {selected_ticker}",
            xaxis=dict(
            title="Datetime",
            rangeslider=dict(visible=True),
            type="date"
            ),
            yaxis_title="RSI",
            height=300,
            template="plotly_dark",
            showlegend=True  # Ensure legend visibility in layout
            #legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )

        # Create MACD chart
        # Create MACD chart
        macd_fig = go.Figure()
        macd_fig.add_trace(
        go.Bar(
        x=df_filtered['Datetime'],
        y=df_filtered['MACD'] - df_filtered['Signal'],
        name='Histogram',
        marker=dict(color=["green" if diff > 0 else "red" for diff in (df_filtered['MACD'] - df_filtered['Signal'])])
        )
    )
        macd_fig.add_trace(
        go.Scatter(
        x=df_filtered['Datetime'],
        y=df_filtered['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='blue')
    )
)
        macd_fig.add_trace(
        go.Scatter(
        x=df_filtered['Datetime'],
        y=df_filtered['Signal'],
        mode='lines',
        name='Signal',
        line=dict(color='orange')
    )
)
        macd_fig.update_layout(
            title=f"MACD Chart for {selected_ticker}",
            xaxis=dict(title="Datetime", rangeslider=dict(visible=True)),
            yaxis_title="MACD",
            height=300,
            template="plotly_dark"
        )

################### Charting For Impulse MACD #################################
        # Create Impulse MACD chart
        Imp_macd_fig = go.Figure()

        # Add Histogram with colors
        Imp_macd_fig.add_trace(
            go.Bar(
                x=df_filtered['Datetime'],
                y=df_filtered['Impulse_Histogram'],
                marker_color=df_filtered['Impulse_Color'],
                name='Impulse Histogram'
            )
        )

        # Add Signal Line
        Imp_macd_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=df_filtered['Impulse_Signal'],
                mode='lines',
                name='Impulse Signal',
                line=dict(color='maroon')
            )
        )

        # Add Midline (0 Line)
        Imp_macd_fig.add_trace(
            go.Scatter(
                x=df_filtered['Datetime'],
                y=[0] * len(df_filtered),
                mode='lines',
                name='Midline',
                line=dict(color='gray', dash='dot')
            )
        )

        Imp_macd_fig.update_layout(
            title=f"Impulse MACD Chart for {selected_ticker}",
            xaxis=dict(title="Datetime", rangeslider=dict(visible=True)),
            yaxis_title="Impulse MACD",
            height=300,
            template="plotly_dark"
        )

################### END Of Charting For Impulse MACD ##########################

        return candlestick_fig, rsi_fig, macd_fig, Imp_macd_fig

    app.run_server(debug=True, use_reloader=False, port=8052)

def run_dash_server():
    """
    Function that runs the Dash server.
    This function will be launched in a separate process.
    """
    #app.run_server(debug=True, use_reloader=False, port=8051)

import sys
def run_script(script_path):
    """
    Run a Python script as a subprocess and capture its output and errors.

    Args:
        script_path (str): Path to the script to run.
    """
    try:
        # Run the subprocess
        result = subprocess.run(
            [sys.executable, script_path],  # Use the same Python interpreter
            check=True,                     # Raise an exception on failure
            stdout=sys.stdout,  # Redirect stdout to parent process
            stderr=sys.stderr       # Capture standard erro                    # Decode bytes to string
        )
        # Log output
        print(f"Output of {script_path}:\n{result.stdout}")
        print(f"Executed {script_path} successfully.")
    except subprocess.CalledProcessError as e:
        # Log errors
        print(f"Error executing {script_path}: {e}")
        print(f"Error output:\n{e.stderr}")

import time as time_module
# Run the Dash app
if __name__ == "__main__":
    # Record the start time
    start_time = time_module.time()
    print(f"Script started at: {datetime.now()}")

    file_path = "/Users/jimutmukhopadhyay/Dummy Trading/AlpacaTrading/BuySellSignalsDaily.xlsx"

    # Step 1: Delete the existing file if it exists
    if os.path.exists(file_path):
        print(f"Deleting existing file: {file_path}")
        os.remove(file_path)  # Delete the old file before writing the new one

    # Run the first script
    # run_script("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/AccountInfo/reqExcPosAcntDetails.py")

    # # Main logic (replace 'main()' with the actual function you are running)
    # try:
    #     main()
    # except Exception as e:
    #     print(f"Error during main execution: {e}")

    # # Run the second script
    # run_script("/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/AccountInfo/ReqOpenOrderToExcel.py")
    import concurrent.futures
    # Now, instead of sequentially calling run_script and main, we submit them as tasks.
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        # Submit each task to the executor.
        #future_reqAccDetails = executor.submit(run_script, "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/AccountInfo/reqExcPosAcntDetails.py")
        future_main = executor.submit(main)
        #future_reqOpenOrders = executor.submit(run_script, "/Users/jimutmukhopadhyay/Dummy Trading/IntraDay Trading/AccountInfo/ReqOpenOrderToExcel.py")

        # Optionally, wait for all tasks to finish and capture any exceptions.
        futures = [future_main]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()  # If tasks return output, print it here.
                if result is not None:
                    print(result, flush=True)
            except Exception as exc:
                print(f"A task generated an exception: {exc}", flush=True)

    # Record the end time and calculate the execution time
    end_time = time_module.time()
    execution_time = end_time - start_time
    print(f"Script finished at: {datetime.now()}")
    print(f"Script executed in {execution_time:.2f} seconds.")