import numpy as np
import math
import pandas as pd
import os
from SupportResistanceLevels import dynamic_support_resistance, support, resistance, adjust_window_size
##### Add MACD Crossover State Function #####
def add_macd_crossover_state(df):
    """
    Identify MACD crossover and continuing crossover state.
    Tracks the state of crossover until MACD and Signal both drop below zero.
    Additionally, triggers on the first histogram zero-crossing (negative to positive).
    """
    # Ensure MACD_Histogram exists
    if 'MACD_Histogram' not in df.columns:
        df['MACD_Histogram'] = df['MACD'] - df['Signal']

    df['MACD_Crossover_State'] = False
    crossover_active = False

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        # Histogram zero-crossing: from negative to positive
        hist_prev = df['MACD_Histogram'].iat[i-1]
        hist_curr = df['MACD_Histogram'].iat[i]

        # Primary trigger: MACD line crosses above Signal line
        macd_cross = (prev['MACD'] < prev['Signal']) and (curr['MACD'] > curr['Signal'])
        # Secondary trigger: histogram flips from below zero to above
        hist_cross = (hist_prev < 0) and (hist_curr > 0)

        if macd_cross or hist_cross:
            crossover_active = True
            df.at[i, 'MACD_Crossover_State'] = True

        elif crossover_active:
            # Maintain state while MACD remains above Signal
            if curr['MACD'] > curr['Signal']:
                df.at[i, 'MACD_Crossover_State'] = True
            # Exit state when both lines fall below zero
            elif curr['MACD'] < 0 and curr['Signal'] < 0:
                crossover_active = False

    return df

import numpy as np

def calculate_3pt_angle(y1, y2, y3):
    # Points in space: assume x-axis is time with equal spacing
    A = np.array([1, y1 - y2])
    B = np.array([1, y3 - y2])

    # Cosine of angle between A and B
    cos_theta = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # prevent domain error
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


##### # Add buy signals based on MACD, Histogram, and Impulse MACD ##############################
def add_buy_signals(df, ticker_name="Unknown Ticker", fib_levels=None, macd_thresholds=None, signal_log=None):
    """
    5-min buy signals with simplified histogram test,
    corrected EMA10/30 crossover, and full logging to Excel.
    """
    import os
    import numpy as np
    os.makedirs("BuySellLog", exist_ok=True)

    df = df.reset_index(drop=True)
    df['Buy_Signal'] = False
    df['Mid'] = (df['Open'] + df['Close']) / 2
    prev_sig_idx = -1
    last_buy = None
    df = add_macd_crossover_state(df)
    full_log = []
    if signal_log is None:
        signal_log = []

    for i in range(10, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        two = df.iloc[i - 2]
        thr = df.iloc[i - 3]

        # cooldown reset after 20 bars
        if prev_sig_idx != -1 and (i - prev_sig_idx) >= 10:
            prev_sig_idx = -1

        # look back up to 20 bars
        start = max(0, i - 20)
        macd_window = df['MACD'].iloc[start:i]
        macd_ever_pos = (macd_window > 0).any()

        hist_window = (df['MACD'] - df['Signal']).iloc[start:i]
        hist_deep_enough = (hist_window < -0.4).any()

        # histogram values
        h0 = thr['MACD'] - thr['Signal']
        h1 = two['MACD'] - two['Signal']
        h2 = prev['MACD'] - prev['Signal']
        h3 = cur['MACD'] - cur['Signal']
        hist_convergence = (h0 < h1 < h2 < h3) and (h3 < 0)
        increasing_histogram = (h3 > h2 > h1)
        macd_up = (two['MACD'] < cur['MACD'] < 0) #and (cur['MACD'] - two['MACD'] > 0.005)
        signal_up = (two['Signal'] < prev['Signal'] < cur['Signal'] < 0)
        macd_higher_than_signal = (cur['MACD'] > cur['Signal']) and (prev['MACD'] > prev['Signal']) and (two['MACD'] > two['Signal'])
        Higher_Trend = signal_up and macd_higher_than_signal and macd_up and increasing_histogram
        # HMA 10 logic

        HMA_Up = (cur['HA_EMA10'] - two['HA_EMA10']) > 0

        # Custom Predictive MACD Logic
        hist_block = df['MACD_Histogram'].iloc[i - 10:i]
        hist_curr = df['MACD_Histogram'].iloc[i]
        hist_prev = df['MACD_Histogram'].iloc[i - 1]
        hist_two  = df['MACD_Histogram'].iloc[i - 2]
        macd_prev = df['MACD'].iloc[i - 1]
        signal_prev = df['Signal'].iloc[i-1]
        rsi_prev = df['RSI'].iloc[i - 1]
        rsi_curr = df['RSI'].iloc[i]
        rsi = (rsi_prev + rsi_curr) / 2
        mfi_prev = df['MFI'].iloc[i - 1]
        mfi_curr = df['MFI'].iloc[i]
        mfi_avg = (mfi_prev + mfi_curr) / 2

        # add this:
        cond_cross = (hist_prev < 0) and (hist_curr > 0) and (hist_two < hist_prev) # -0.004
        negative_macd = macd_prev < 0
        Higher_Trend_confirmed = False
        # 2) If Higher_Trend is true at bar i, look BACK up to 10 bars
        if Higher_Trend:
            # Determine the earliest j we can check: j >= 2, and no further back than i-10.
            j_start = max(2, i - 10)
            # Loop forward from j_start up to i-1 (inclusive). 
            # At each j we test whether that bar satisfied "cond_cross".
            for j in range(j_start, i):
                hist_prev = df["MACD_Histogram"].iloc[j - 1]
                hist_curr = df["MACD_Histogram"].iloc[j]
                hist_two  = df["MACD_Histogram"].iloc[j - 2]

                # The exact same condition you wrote for cond_cross:
                if (hist_prev < 0) and (hist_curr > 0) and (hist_two < hist_prev):
                    Higher_Trend_confirmed = True
                    break
        # Step 1: Always extract the most recent negative block from the last 10 bars
        hist_recent = df['MACD_Histogram'].iloc[i - 10:i]
        neg_block = [] 
        for j in reversed(range(i - 5, i)):
            if df['MACD_Histogram'].iloc[j] < 0:
                neg_block.insert(0, df['MACD_Histogram'].iloc[j])
            else:
                break


        # Step 3: Confirm prev bar is lowest in negative block
        cond_1 = (
            len(neg_block) > 0 and
            hist_two == min(neg_block)
        )

        # three-bar rising histogram
        cond_hist_3 = (hist_curr > hist_prev) and (hist_prev > hist_two)
        #macd_up = (two['MACD'] < prev['MACD'] < cur['MACD'] < 0)
        # two bars ago red, then two consecutive greens
        from datetime import time

        # replace your old cond_2 with:
        cond_2 = cond_hist_3 and macd_up #and cond_candle_3
        cond_3 = macd_prev < 0.8 * (hist_prev) #and signal_prev < 0.9 * (hist_prev)
                # old:
        # cond_4 = (rsi_prev and rsi_curr <= 35)

        if mfi_avg < 65:
            cond_4 = (rsi < 50)
        else:
            cond_4 = (rsi < 35)

        epsilon = 0  # tolerance for near-zero green bars
        # Step 1: Collect consecutive negative histogram bars ending at `i`
        neg_hist_vals = []
        neg_hist_indices = []
        for j in range(i, i - 20, -1):  # max lookback to 20 bars
            if j < 0:
                break
            h = df['MACD_Histogram'].iloc[j]
            if h < epsilon:
                neg_hist_vals.append(h)
                neg_hist_indices.append(j)
            else:
                break

        # Flip to chronological order (oldest to latest)
        neg_hist_vals = neg_hist_vals[::-1]
        neg_hist_indices = neg_hist_indices[::-1]

        # Initial default values to avoid UnboundLocalError
        num_bars_before_min = None
        bars_to_zero = np.inf
        #cond_5 = False
        cond_5 = True

        # in your per-bar loop:
        cur = df.iloc[i]
    
        if ticker_name in ("APP", "TSLA"):
            cond_6 = True
        else:
            mfi = mfi_avg
            mfi_scaled = cur['MFI_Scaled']
            mfi_scaled_prev = prev['MFI_Scaled']

            if mfi < 25:
                # always allow deep‐oversold readings
                cond_6 = True
            elif mfi <= 55:
                cond_6 = (cur['Low'] < mfi_scaled + 0.1 < cur['High']) or (prev['Low'] < mfi_scaled_prev < prev['High']) or (cur['High'] < mfi_scaled)
            else:
                # over 50: no signal
                cond_6 = False

        prev_vroc = prev['VROC']
        cur_vroc = cur['VROC']
        vroc_avg = (prev_vroc + cur_vroc) / 2
        cond_7 = -0.8 < vroc_avg < 5
        # grab your raw & scaled values
        cmf_raw     = cur['CMF']
        cmf_scaled  = cur['CMF_Scaled']

        cond_8_scaled = cur['Low'] < cmf_scaled < cur['High']

        # pull out the local time of this bar
        bar_time = cur['Datetime'].time()
        # if it’s between 09:30 and 11:00, always true
        if time(9, 30) <= bar_time <= time(11, 0):
            cond_8_raw = (-0.9 < cmf_raw <= 0.25)
        else:
            cond_8_raw = (-0.9 < cmf_raw <= 0.2)

        # combine with your existing raw CMF check
        cond_8 = (cond_8_raw) #and cond_8_scaled
        #cond_9 = df.at[i, 'MACD_Crossover_State']

        #print(f"current Low and current MFI - {cur['Low']}, {cur['MFI']}")
        # Final custom MACD condition
        macd_cond = (
        (cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and HMA_Up and macd_up and cond_7 and cond_8 and prev_sig_idx == -1)
        or
        #(cond_cross and negative_macd and cond_4 and cond_5 and cond_6 and cond_7 and cond_8 and prev_sig_idx == -1)
        (Higher_Trend_confirmed and cond_4 and cond_7 and cond_8 and HMA_Up and prev_sig_idx == -1)
        )

        #macd_cond = (Higher_Trend_confirmed and cond_4 and cond_7 and cond_8 and prev_sig_idx == -1)


        vwap_status = (
            df['Mid'].iloc[i]   < df['Lower_VWAP'].iloc[i] and
            df['Mid'].iloc[i-1] < df['Lower_VWAP'].iloc[i-1] and
            df['Mid'].iloc[i-2] < df['Lower_VWAP'].iloc[i-2] and
            df['Mid'].iloc[i-3] < df['Lower_VWAP'].iloc[i-3]
        )
        vwap_condition = (vwap_status and cur['RSI'] < 45 and hist_convergence and (prev_sig_idx == -1))


        ts = cur['Datetime']
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)

        bars_since_last = (i - prev_sig_idx) if prev_sig_idx != -1 else None

        full_log.append({
            "Ticker": ticker_name,
            "Date Time": ts,
            "Cond 1": cond_1,
            "Cond 2": cond_2,
            "Cond 3": cond_3,
            "Cond 4": cond_4,
            "Cond 5": cond_5,
            "Cond 6": cond_6,
            "Cond 7": cond_7,
            "Cond 8": cond_8,
            "Cond 9": cond_cross,
            "MFI_Scaled": cur['MFI_Scaled'],
            "CMF_Scaled": cur['CMF_Scaled'],
            "VROC": cur['VROC'],
            "Two Before Hist": hist_two,
            "One Before Hist": hist_prev,
            "Current Histogra": hist_curr,
            "Current Low": cur['Low'],
            "Cond 8 Scaled": cond_8_scaled,
            "hist_conv": hist_convergence,
            "MACD Previous": macd_prev,
            "MACD UP": macd_up,
            "0.9 Hist_Previous": hist_prev,
            "Number of bars before Min": num_bars_before_min,
            "bars_to_zero": bars_to_zero,
            "prev_sig_idx": prev_sig_idx,
            "bars_since_last": bars_since_last,
            "macd_condition": macd_cond,
        })

        if macd_cond or vwap_condition:
            df.at[i, 'Buy_Signal'] = True
            prev_sig_idx = i
            last_buy = cur['Close']
            # Determine source of signal
            if macd_cond:
                signal_type = "MACD"
            # elif ema_cond:
            #     signal_type = "EMA"
            elif vwap_condition:
                signal_type = "VWAP"

            signal_log.append({
                "Ticker": ticker_name,
                "Date Time": pd.Timestamp(ts),
                "Buy Signal": signal_type,
                "Buy High price": cur['High'],
                "Buy Low": cur['Low'],
                "Buy RSI": cur['RSI'],
                "bars_to_zero": bars_to_zero,
                "VWAP Condition": vwap_condition
            })

    df_full = pd.DataFrame(full_log)
    df_signals = pd.DataFrame(signal_log)

    for D in (df_full, df_signals):
        if 'Date Time' in D.columns:
            series = D['Date Time']
            if series.dtype.kind == 'M' and series.dt.tz is not None:
                D['Date Time'] = series.dt.tz_localize(None)
            else:
                D['Date Time'] = pd.to_datetime(series, errors='coerce')

    out = os.path.join("BuySellLog", f"{ticker_name}_5min_logs.xlsx")
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        df_signals.to_excel(writer, sheet_name="Buy_Signals", index=False)
        df_full.to_excel(writer, sheet_name="Full_Log", index=False)

    print(f"Wrote logs for {ticker_name} → {out}")
    return df, last_buy

################### End Of Adding Buy Signal #####################################################

##### Add MACD Crossover State for Sell Function #####
def add_macd_crossover_state_for_sell(df):
    """
    Identify MACD crossover for bearish signals and continuing crossover state.
    Tracks the state of crossover until MACD and Signal both rise above zero.
    """
    df['MACD_Crossover_Sell_State'] = False  # Initialize column
    crossover_active = False  # State flag for MACD crossover

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        # Detect the initial MACD bearish crossover (MACD falls below Signal)
        if previous['MACD'] > previous['Signal'] and current['MACD'] < current['Signal']:
            crossover_active = True
            df.at[i, 'MACD_Crossover_Sell_State'] = True

        # Continue the crossover state until MACD and Signal rise above zero
        elif crossover_active:
            if current['MACD'] < current['Signal']:
                df.at[i, 'MACD_Crossover_Sell_State'] = True
            elif current['MACD'] > 0 and current['Signal'] > 0:
                crossover_active = False  # Reset the active state

    return df


####################### ADD Sell Signals ####################################################
def add_sell_signals(
    df,
    ticker_name="Unknown Ticker",
    fib_levels=None,
    last_buy_price=None,
    macd_thresholds=None,
    signal_log=None,
    spread_percentage=None,
    Avg_HL_Sprd=None
):
    """
    5-min sell signals: three distinct exit conditions (MACD, LongExit, VWAP) plus a ShortSell.
    Exact inverses of buy logic where applicable; commented sections preserved.
    """
    import os
    import numpy as np
    from datetime import time
    os.makedirs("BuySellLog", exist_ok=True)

    df = df.reset_index(drop=True)
    df['Sell_Signal'] = False
    df['Mid'] = (df['Open'] + df['Close']) / 2
    prev_sig_idx = -1
    last_sell = None

    # ensure MACD crossover state exists
    df = add_macd_crossover_state(df)

    full_log = []
    if signal_log is None:
        signal_log = []

    for i in range(10, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i-1]
        two  = df.iloc[i-2]
        thr  = df.iloc[i-3]

        # cooldown reset after 10 bars
        if prev_sig_idx != -1 and (i - prev_sig_idx) >= 10:
            prev_sig_idx = -1

        # rolling windows
        start = max(0, i-20)
        macd_window = df['MACD'].iloc[start:i]
        hist_window = (df['MACD'] - df['Signal']).iloc[start:i]

        # histogram values
        hist_curr = df['MACD_Histogram'].iat[i]
        hist_prev = df['MACD_Histogram'].iat[i-1]
        hist_two  = df['MACD_Histogram'].iat[i-2]
        macd_prev = df['MACD'].iat[i-1]
        signal_prev = df['Signal'].iat[i-1]
        rsi_prev  = df['RSI'].iat[i-1]
        rsi_curr  = df['RSI'].iat[i]

        # 1) Positive histogram block for sell
        pos_block = []
        for j in reversed(range(i-5, i)):
            if df['MACD_Histogram'].iat[j] > 0:
                pos_block.insert(0, df['MACD_Histogram'].iat[j])
            else:
                break
        #cond_1 = len(pos_block) > 0 and (hist_two == max(pos_block))
        cond_1 = len(pos_block) > 0 and (hist_prev == max(pos_block))
        # 2) Three-bar falling histogram + candle check
        #cond_hist_3 = (hist_curr < hist_prev) and (hist_prev < hist_two)
        cond_hist_3 = (hist_curr < hist_prev)
        bar_time = cur['Datetime'].time()
        # if ticker_name in ("APP","TSLA") or time(9,30) <= bar_time <= time(11,0):
        #     cond_candle = True
        # else:
        #     cond_candle = (
        #         df['Close'].iat[i-2] > df['Open'].iat[i-2] and
        #         df['Close'].iat[i-1] < df['Open'].iat[i-1]
        #     )
        cond_candle = True
        cond_2 = cond_hist_3 and cond_candle

        # 3) MACD value check
        cond_3 = macd_prev > 0.9 * hist_prev  # # and signal_prev > 0.9*hist_prev

        # 4) RSI check (mirror of buy oversold)
        cond_4 = (
            (rsi_curr >= 55) or (rsi_curr >= 55 and rsi_prev > 55)
        )

        # 5) Exhaustion skip (always True for simplicity)
        cond_5 = True

        # 6) MFI-based inverse
        mfi = cur['MFI']
        low = cur['Low']
        high = cur['High']
        mfi_scaled = cur['MFI_Scaled']
        cmf_scaled = cur['CMF_Scaled']

        if mfi > 80:
            cond_6 = True
        elif mfi >= 55:
            cond_6 = low < mfi_scaled < high
        else:
            cond_6 = False

        # 7) VROC inverse threshold
        cond_7 = cur['VROC'] > 0
        # 8) CMF raw inverse, morning vs later
        cmf_raw = cur['CMF']
        cond_8 = cmf_raw > 0 and low < cmf_scaled < high

        if i >= 9:
            cond_mfi_uptrend = (df['MFI'].iat[i] - df['MFI'].iat[i-9]) > 0
        else:
            cond_mfi_uptrend = False

        # cross from positive to negative histogram
        cond_cross = (hist_prev > 0) and (hist_curr < 0)

        # Define the exit conditions:
                # ShortSell: full sell logic plus MFI >80 (always include APP/TSLA)
        cond_short = (
            cond_1 and cond_2 and cond_3 and cond_4 and
            cond_5 and cond_7 and cond_8 and (mfi > 80) and cond_mfi_uptrend and
            prev_sig_idx == -1
        )
        # full MACD exit
        cond_macd = (
            cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and
            cond_6 and cond_7 and cond_8 and prev_sig_idx == -1
        )
        # LongExit on histogram cross
        cond_long_exit = (
            cond_cross and cond_4 and cond_5 and cond_6 and
            cond_7 and cond_8 and prev_sig_idx == -1
        )
        # VWAP-based Exit
        vwap_status = (
            df['Mid'].iat[i]   > df['Upper_VWAP'].iat[i]   and
            df['Mid'].iat[i-1] > df['Upper_VWAP'].iat[i-1] and
            df['Mid'].iat[i-2] > df['Upper_VWAP'].iat[i-2] and
            df['Mid'].iat[i-3] > df['Upper_VWAP'].iat[i-3]
        )
        cond_vwap = vwap_status and cur['RSI'] > 60 and prev_sig_idx == -1

        # consolidate into a single flag and assign type
        sell_flag   = False
        signal_type = None
        if cond_short:
            sell_flag   = True
            signal_type = 'ShortSell'
        elif cond_macd:
            sell_flag   = True
            signal_type = 'MACD'
        elif cond_long_exit:
            sell_flag   = True
            signal_type = 'LongExit'
        elif cond_vwap:
            sell_flag   = True
            signal_type = 'VWAP'

        ts = cur['Datetime']
        if hasattr(ts,'tzinfo') and ts.tzinfo is not None:
            ts = ts.tz_localize(None)

        full_log.append({
            'Ticker': ticker_name,
            'Date Time': ts,
            'Cond1': cond_1, 'Cond2': cond_2,
            'Cond3': cond_3, 'Cond4': cond_4,
            'Cond5': cond_5, 'Cond6': cond_6,
            'Cond7': cond_7, 'Cond8': cond_8,
            'Cross': cond_cross,
            "VROC": cur['VROC'],
            "CMF": cur['CMF'],
            "MFI": cur['MFI'],
            'sell_flag': sell_flag,
            'SignalType': signal_type
        })

        if sell_flag:
            df.at[i,'Sell_Signal'] = True
            prev_sig_idx = i
            last_sell = cur['Close']
            signal_log.append({
                'Ticker': ticker_name,
                'Date Time': pd.Timestamp(ts),
                "Buy Signal":         "",
                'Sell Signal': signal_type,
                'Sell High price': cur['High'],
                'Sell Low': cur['Low'],
                'Sell RSI': cur['RSI']
            })

    df_sell_full    = pd.DataFrame(full_log)
    # (Optional) strip tzinfo / coerce datetimes exactly as you did:
    for D in  [df_sell_full]:
        if 'Date Time' in D.columns:
            ser = D['Date Time']
            if ser.dtype.kind=='M' and ser.dt.tz is not None:
                D['Date Time'] = ser.dt.tz_localize(None)
            else:
                D['Date Time'] = pd.to_datetime(ser, errors='coerce')

    from openpyxl import load_workbook

    out = os.path.join("BuySellLog", f"{ticker_name}_5min_logs.xlsx")
    # 1) if the file doesn’t exist, write buy sheets with xlsxwriter as before

    # 2) later, when you want to add sell sheets:
    with pd.ExcelWriter(out, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # this will keep the existing buy sheets intact
        df_sell_full   .to_excel(writer, sheet_name="Sell_Full_Log",   index=False)

    print(f"Wrote buy+sell logs for {ticker_name} → {out}")
    return df
############################ End of adding Sell Signal ##########################################
