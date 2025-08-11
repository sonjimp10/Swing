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

def first_eligible_ema50_signal_index(df, gap_pct, tz="America/New_York", near_tol=0.0):
    """
    For gap_pct > -2:
      - scan each intraday 'touch' where EMA50 is between the bar's open/close
        (optionally allow a small near-tolerance),
      - for each touch j, if j+1 and j+2 both close above EMA50,
        and at j+2 volume_ok & cmf_ok hold, return the global index for j+2.
    Returns None if no eligible touch is found.
    """
    if gap_pct is None or gap_pct <= -2:
        return None

    # group by trading day in ET
    try:
        trade_date = df["Datetime"].dt.tz_convert(tz).dt.date
    except Exception:
        trade_date = pd.to_datetime(df["Datetime"], errors="coerce").dt.tz_localize(tz).dt.date

    green = df["Close"] > df["Open"]
    vol   = df["Volume"]

    for day in pd.unique(trade_date):
        mask = (trade_date == day)
        g = df.loc[mask].copy()
        if g.empty:
            continue

        o = g["Open"].to_numpy()
        c = g["Close"].to_numpy()
        e = g["EMA50"].to_numpy()

        # strict body-touch
        touch = ((o - e) * (c - e) <= 0)

        if near_tol > 0:
            # allow "near touch" within a fraction of EMA (e.g., 0.001 for 0.1%)
            near = (np.minimum(o, c) <= e * (1 + near_tol)) & (np.maximum(o, c) >= e * (1 - near_tol))
            touch = touch | near

        touch_idx = np.flatnonzero(touch)
        if len(touch_idx) == 0:
            continue

        # try each touch in chronological order
        for j in touch_idx:
            if j + 2 >= len(g):
                break

            above1 = g["Close"].iat[j+1] >= g["EMA50"].iat[j+1]
            above2 = g["Close"].iat[j+2] >= g["EMA50"].iat[j+2]
            if not (above1 and above2):
                continue

            # evaluate filters at the *confirmation* bar (j+2) in global index
            i_global = g.index[j+2]

            # CMF rising over last 3 bars
            if i_global >= 2:
                cmf3 = df["CMF"].iloc[i_global-2:i_global+1]
                cmf_slope = np.polyfit([0,1,2], cmf3.values, 1)[0]
                cmf_ok = cmf3.is_monotonic_increasing and (cmf_slope > 0)
            else:
                cmf_ok = False

            # volume OK (same rule you use elsewhere)
            if i_global >= 3:
                last4_green = green.iloc[i_global-3:i_global+1]
                prev3_avg   = vol.iloc[i_global-3:i_global].mean()
                cond2 = green.iloc[i_global-2:i_global+1].all()     # last 3 all green
                cond3 = (last4_green.sum() >= 3)                    # ≥3 green of last 4
                volume_ok = cond2 or cond3
            else:
                volume_ok = False

            if cmf_ok and volume_ok:
                return int(i_global)  # confirmation bar index

    return None


##### # Add buy signals based on MACD, Histogram, and Impulse MACD ##############################
def add_buy_signals(df, ticker_name="Unknown Ticker", fib_levels=None, gap_pct = None, MarketBodyRatio = None, macd_thresholds=None, signal_log=None):
    """
    5-min buy signals with simplified histogram test,
    corrected EMA10/30 crossover, and full logging to Excel.
    """
    import os
    import numpy as np
    os.makedirs("BuySellLog", exist_ok=True)
        # DEBUG: print the extra parameters for this ticker
    #print(f"[DEBUG] {ticker_name}  gap_pct={gap_pct},  market_body_ratio={MarketBodyRatio}")

    df = df.reset_index(drop=True)
    df['Buy_Signal'] = False
    df['Mid'] = (df['Open'] + df['Close']) / 2
    prev_sig_idx = -1
    last_buy = None
    df = add_macd_crossover_state(df)

    # ── Day/EMA50 precomputations for new rules ───────────────────────────────
    try:
        trade_date = df["Datetime"].dt.tz_convert("America/New_York").dt.date
    except Exception:
        trade_date = pd.to_datetime(df["Datetime"], errors="coerce").dt.date
    df["TradeDate"] = trade_date

    # Candle body "touches/crosses" EMA50 (EMA50 lies between Open and Close)
    df["body_cross_ema50"] = ((df["Open"] - df["EMA50"]) * (df["Close"] - df["EMA50"]) <= 0) & (df["Open"] != df["Close"])

    # Mark the very first such touch per day
    df["first_touch_today"] = False
    for d, grp in df.groupby("TradeDate"):
        idx = grp.index[grp["body_cross_ema50"]]
        if len(idx) > 0:
            df.loc[idx[0], "first_touch_today"] = True

    # Per-day: is the second bar's close > first bar's open?
    df["_second_close_gt_first_open"] = False
    for d, grp in df.groupby("TradeDate"):
        if len(grp) >= 2:
            first_i, second_i = grp.index[0], grp.index[1]
            ok = df.loc[second_i, "Close"] > df.loc[first_i, "Open"]
            df.loc[grp.index, "_second_close_gt_first_open"] = ok

    # Decide whether to enforce cond_5 by day (Rule B)
    if (gap_pct is not None) and (gap_pct < -2) and (MarketBodyRatio is not None) and (MarketBodyRatio > 0):
        # If second close ≤ first open → ignore cond_5 (treat as True)
        df["_allow_cond5"] = df["_second_close_gt_first_open"]
    else:
        df["_allow_cond5"] = True  # default: enforce cond_5


    full_log = []
    if signal_log is None:
        signal_log = []

    # ── EMA50 path: re-arm on every touch; fire on touch+2 ─────────────────────
    ema50_idx = first_eligible_ema50_signal_index(df, gap_pct, tz="America/New_York", near_tol=0.0)

    for i in range(10, len(df)):
        cur = df.iloc[i]
        prev = df.iloc[i - 1]
        two = df.iloc[i - 2]
        thr = df.iloc[i - 3]
        # EMA50 path → trigger only at that exact index
        ema50_gap_buy = (ema50_idx is not None and i == ema50_idx)
        # cooldown reset after 20 bars
        if prev_sig_idx != -1 and (i - prev_sig_idx) >= 2:
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
        # allow up to 4 initial positives before the negative block
        max_initial_positives = 2
        
        neg_hist_vals = []
        neg_hist_indices = []
        neg_block = []
        initial_pos_count = 0
        found_negative   = False

        # look back at most (max_initial_positives + window) bars
        window = 5
        start  = max(0, i - (window + max_initial_positives))
        
        for j in range(i - 1, start - 1, -1):
            h = df['MACD_Histogram'].iat[j]

            if not found_negative:
                # we haven't seen our first negative yet
                if h < 0:
                    found_negative = True
                    neg_block.insert(0, h)
                else:
                    # still in the “initial positives” region
                    initial_pos_count += 1
                    if initial_pos_count > max_initial_positives:
                        # too many leading positives → give up
                        break
                    # otherwise skip it
                    continue
            else:
                # we've started collecting negatives
                if h < 0:
                    neg_block.insert(0, h)
                else:
                    # hit a positive *after* the neg run → stop
                    break

        # Step 3: Confirm prev bar is lowest in negative block
        cond_1 = (
            len(neg_block) > 0 
            # and
            # hist_prev == min(neg_block)
        )

        # compute your two key values
        zero_cross = (hist_prev <= 0) & (hist_curr > 0)
        still_neg_rising = (hist_curr > hist_prev) & (hist_curr <= 0)

        # combine them: either you’re still in the negative-rise regime, or you’re
        # on the very first positive bar after the zero‐cross
        cond_hist_3 = zero_cross | still_neg_rising

        from datetime import time

        # replace your old cond_2 with:
        cond_2 = cond_hist_3 #and macd_up #and cond_candle_3
        cond_3 = macd_prev < 0.9 * (hist_prev) #and signal_prev < 0.9 * (hist_prev)
        cond_4 = rsi < 50

        # Flip to chronological order (oldest to latest)
        neg_hist_vals = neg_hist_vals[::-1]
        neg_hist_indices = neg_hist_indices[::-1]

        # Initial default values to avoid UnboundLocalError
        num_bars_before_min = None
        bars_to_zero = np.inf
        #cond_5 = False

        # in your per-bar loop:
        cur = df.iloc[i]
        # CMF gate (cond_5), with daily override (Rule B)
        cmf_raw   = cur["CMF"]
        cond_5_base = (cmf_raw > 0.001)
        allow_c5    = bool(df["_allow_cond5"].iat[i])
        cond_5      = cond_5_base if allow_c5 else True  # ignore CMF gate if day says so
        # ——— Volume confirmation (green‐volume tests) ————————————————
        # define a boolean Series once (you could even move this to top of function)
        green_vol = df["Close"] > df["Open"]

        vol       = df["Volume"]

        if i >= 3:
            # last four bars, including the current one
            last4 = green_vol.iloc[i-3 : i+1]

            # 1) ≥2 green bars **AND** current volume > avg(volume of prior 3 bars)
            prev3_avg_vol = vol.iloc[i-3 : i].mean()
            cond_vol1 = (
                last4.sum() >= 2
                and
                vol.iat[i] > prev3_avg_vol
            )

            # 2) last three bars are all green
            cond_vol2 = green_vol.iloc[i-2 : i+1].all()

            # 3) ≥3 green bars in those same four
            cond_vol3 = (last4.sum() >= 3)

            volume_ok = cond_vol2 or cond_vol3
        else:
            volume_ok = False

        #print(f"current Low and current MFI - {cur['Low']}, {cur['MFI']}")
        # Final custom MACD condition
        macd_cond = (
        (cond_1 and cond_2 and cond_3 and cond_4 and cond_5 and volume_ok and prev_sig_idx == -1))


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

        # Full log row
        full_log.append({
            "Ticker": ticker_name,
            "Date Time": ts,
            "Cond 1": cond_1,
            "Cond 2": cond_2,
            "Cond 3": cond_3,
            "Cond 4": cond_4,
            "Cond 5": cond_5,
            "Volume OK": volume_ok,
            "Cond 9 (zero cross)": cond_cross,
            "MFI_Scaled": cur.get("MFI_Scaled", np.nan),
            "CMF_Scaled": cur.get("CMF_Scaled", np.nan),
            "VROC": cur.get("VROC", np.nan),
            "Two Before Hist": hist_two,
            "One Before Hist": hist_prev,
            "Current Histogram": hist_curr,
            "Current Low": cur["Low"],
            "hist_conv": hist_convergence,
            "MACD Previous": macd_prev,
            "MACD UP": macd_up,
            "0.9 Hist_Previous": hist_prev,
            "prev_sig_idx": prev_sig_idx,
            "bars_since_last": bars_since_last,
            "macd_condition": macd_cond,
            "EMA50_Gap_Buy": ema50_gap_buy,
            "Allow_Cond5_Today": allow_c5,
        })

        # Final decision: any of the three paths can fire a buy
        if macd_cond or vwap_condition or ema50_gap_buy:
            df.at[i, "Buy_Signal"] = True
            prev_sig_idx = i
            last_buy = cur["Close"]

            if ema50_gap_buy:
                signal_type = "EMA50_TOUCH"
            elif vwap_condition:
                signal_type = "VWAP"
            else:
                signal_type = "MACD"

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
    Sell signals = exact logical opposite of the 5-min buy conditions,
    plus your VWAP-exit logic. Structure and variable names unchanged.
    """
    import numpy as np, math

    sell_full_log = []
    if signal_log is None:
        signal_log = []

    df = df.reset_index(drop=True)
    df['Sell_Signal'] = False
    df['Mid'] = (df['Open'] + df['Close']) / 2

    last_buy_price_val = last_buy_price
    previous_signal_index = -1

    # helper for MACD crossover state
    df = add_macd_crossover_state_for_sell(df)

    # thresholds
    if macd_thresholds is None:
        macd_thresholds = {"MACD_Bearish_Mean": 0.2, "Average_MACD_Bearish": 0.3}
    macd_bearish_mean = macd_thresholds["MACD_Bearish_Mean"]
    avg_macd_bearish  = macd_thresholds["Average_MACD_Bearish"]
    macd_max_threshold = max(macd_bearish_mean, avg_macd_bearish)

    for i in range(10, len(df)):
        cur  = df.iloc[i]
        prev = df.iloc[i-1]

        # update last buy price
        if cur['Buy_Signal']:
            last_buy_price_val = cur['Close']

        # cooldown reset
        if previous_signal_index != -1 and (i - previous_signal_index) >= 2:
            previous_signal_index = -1

        #
        # === Inverse of Buy Conditions ===
        #

        # cond_1: positive block instead of negative
        # pos_block = []
        # for j in reversed(range(i-7, i)):
        #     if df['MACD_Histogram'].iat[j] > 0:
        #         pos_block.insert(0, df['MACD_Histogram'].iat[j])
        #     else:
        #         break
        hist_prev = df['MACD_Histogram'].iat[i-1]
        # cond_1 = (len(pos_block) > 0) and (hist_prev == max(pos_block))

        # cond_2: falling histogram
        hist_curr = df['MACD_Histogram'].iat[i]
        cond_2    = hist_curr < hist_prev

        # cond_3: MACD stronger than 0.9*hist_prev
        macd_prev = df['MACD'].iat[i-1]
        cond_3    = macd_prev > 0.9 * hist_prev

        # cond_4: RSI ≥ 65 instead of ≤ 35
        # rsi_curr = df['RSI'].iat[i]
        # cond_4   = rsi_curr >= 55

        # --- Step 2b: look back for initial positives then a positive run ----
        max_initial_negatives = 2    # allow up to 2 leading negatives before the run
        pos_block = []
        found_positive = False
        initial_neg_count = 0
        window = 5
        start = max(0, i - (window + max_initial_negatives))

        for j in range(i-1, start-1, -1):
            h = df['MACD_Histogram'].iat[j]

            if not found_positive:
                # still in the “initial negatives” region
                if h > 0:
                    found_positive = True
                    pos_block.insert(0, h)
                else:
                    initial_neg_count += 1
                    if initial_neg_count > max_initial_negatives:
                        # too many leading negatives → abort
                        break
                    # otherwise keep skipping through negatives
                    continue
            else:
                # we've started collecting positives
                if h > 0:
                    pos_block.insert(0, h)
                else:
                    # hit a negative after the positive run → stop
                    break

        # Step 3b: Confirm the bar just before i is the peak of that positive block
        hist_two = df['MACD_Histogram'].iat[i-1]
        cond_4 = (
            len(pos_block) > 0
            and
            hist_prev == max(pos_block)
        )

        # # cond_6: High > MFI_Scaled instead of Low < MFI_Scaled
        # cond_6 = cur['High'] > cur['MFI_Scaled']

        # # cond_7: VROC > 2.5 instead of < 2.5
        # cond_7 = cur['VROC'] > 2

        # # — now define cond_8, still inside the `for i` loop but *after* cond_5’s inner loop
        # macd_prev = df['MACD'].iat[i-1]
        # macd_curr = df['MACD'].iat[i]                   # ← new
        # hist_prev = df['MACD_Histogram'].iat[i-1]

        # macd_window = df['MACD'].iloc[max(0, i-20):i]
        # cond_8 = (
        #     (macd_prev == macd_window.max()) and      # previous MACD was the highest in last 20
        #     (hist_prev > 0) and                       # that bar had a positive histogram
        #     (macd_curr < macd_prev)                   # **and** MACD has now fallen below it
        # )

        cond_9 = cur['CMF'] >= 0
        #
        # === VWAP-based Exit (restored) ===
        #
        vwap_status = (
            df['Mid'].iat[i]   > df['Upper_VWAP'].iat[i]   and
            df['Mid'].iat[i-1] > df['Upper_VWAP'].iat[i-1] and
            df['Mid'].iat[i-2] > df['Upper_VWAP'].iat[i-2] and
            df['Mid'].iat[i-3] > df['Upper_VWAP'].iat[i-3]
        )
        vwap_condition = (
            vwap_status and
            cur['RSI'] > 60 and
            # reuse your original histogram_divergence & angle check here if desired
            previous_signal_index == -1
        )

        #
        # === Final Combined Trigger ===
        #
        sell_cond = (
            cond_2 and cond_4 and cond_9 and
            #cond_5 and cond_6 and cond_7 and
            previous_signal_index == -1
        )

        LongExit = (cond_2 and cond_3 and cond_4 and cond_9 and previous_signal_index == -1)

        ts = cur['Datetime']
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.tz_localize(None) if hasattr(ts, 'tz_localize') else ts.replace(tzinfo=None)

        sell_full_log.append({
            "Ticker": ticker_name,
            "Date Time": ts,
            #"Cond 1": cond_1,
            "Cond 2": cond_2,
            "Cond 3": cond_3,
            "Cond 4": cond_4,
            #"Cond 5": cond_5,
            #"Cond 6": cond_6,
            #"Cond 7": cond_7,
            #"Cond 8": cond_8,
            "Cond 9": cond_9,
            "MACD Previous": macd_prev,
            "0.9 Hist_Previous": hist_prev,
            #"Number of bars before Max": num_bars_before_max,
            #"bars_to_zero": bars_to_zero,
            "RSI": cur['RSI'],
        })

        if sell_cond or vwap_condition or LongExit:
            df.at[i, 'Sell_Signal'] = True
            previous_signal_index = i

            # Pick the reason in priority order
            if LongExit:
                reason = "LongExit"
            elif sell_cond:
                reason = "MACD"
            else:
                reason = "VWAP"
            signal_log.append({
                "Ticker":             ticker_name,
                "Date Time":          ts,
                "Buy Signal":         "",
                "Buy Closing price":  "",
                "Last Buy Price":     last_buy_price_val,
                "Sell Signal":        reason,
                "Sell Low price":     cur['Low'],
                "Sell High Price":    cur['High'],
                "Profit Threshold Reached": "",
                "Sell RSI":           cur['RSI'],
                "MACD Angle":         None,
                "MACD Line Angle":    None
            })

    df_sell_full    = pd.DataFrame(sell_full_log)
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