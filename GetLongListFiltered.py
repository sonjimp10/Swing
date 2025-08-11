import os
import pandas as pd
from datetime import datetime

# CONFIGURATION
PM_FOLDER   = "ALPACA_PM_DATA"
MKT_FOLDER  = "ALPACA_DATA"
INPUT_CSV   = "LargeCap.csv"
OUTPUT_XLSX = "Gappers_Analysis_LC.xlsx"

# Set TARGET_DATE for backtesting in YYYY-MM-DD format, or None for latest available
TARGET_DATE = "2025-08-11"

def main():
    # 1) Load tickers & floats
    df_tickers = pd.read_csv(INPUT_CSV, dtype=str)
    df_tickers["Ticker"] = df_tickers["Ticker"].str.upper().str.strip()
    df_tickers["FloatShares"] = pd.to_numeric(df_tickers["FloatShares"], errors="coerce")
    valid_tickers = set(df_tickers["Ticker"])
    float_map     = df_tickers.set_index("Ticker")["FloatShares"].to_dict()

    # 2) Find symbols with market data
    all_files = [f[:-4] for f in os.listdir(MKT_FOLDER) if f.endswith('.csv')]
    tickers   = [t for t in all_files if t in valid_tickers]

    results = []
    for ticker in tickers:
        pm_path  = os.path.join(PM_FOLDER,  f"{ticker}.csv")
        mkt_path = os.path.join(MKT_FOLDER, f"{ticker}.csv")
        res = {"Ticker": ticker}

        if not os.path.exists(pm_path) or not os.path.exists(mkt_path):
            res["Status"] = "missing_data"
            res["Matched"] = False
            results.append(res)
            continue

        # 3) Read & align dates
        df_pm   = pd.read_csv(pm_path,  parse_dates=["Date"])
        df_mkt  = pd.read_csv(mkt_path, parse_dates=["Date"])
        df_pm["date"]  = df_pm["Date"].dt.date
        df_mkt["date"] = df_mkt["Date"].dt.date

        days = sorted(df_mkt["date"].unique())
        if TARGET_DATE:
            target = datetime.strptime(TARGET_DATE, "%Y-%m-%d").date()
            if target not in days:
                res.update({"Status":"target_date_not_found", "Matched": False})
                results.append(res)
                continue
            idx = days.index(target)
            if idx == 0:
                res.update({"Status":"no_prev_day", "Matched": False})
                results.append(res)
                continue
            today, prev_day = days[idx], days[idx-1]
        else:
            if len(days) < 2:
                res.update({"Status":"insufficient_history", "Matched": False})
                results.append(res)
                continue
            today, prev_day = days[-1], days[-2]

        df_pm_today  = df_pm[df_pm["date"] == today]
        df_mkt_today = df_mkt[df_mkt["date"] == today]
        if df_pm_today.empty or df_mkt_today.empty:
            res.update({"Status":"no_trading_data", "Matched": False})
            results.append(res)
            continue

        # 4) Compute metrics
        prev_close     = df_mkt[df_mkt["date"] == prev_day]["Close"].iloc[-1]
        open_price     = df_mkt_today.iloc[0]["Open"]
        gap_pct        = (open_price - prev_close) / prev_close * 100

        pm_vol         = df_pm_today["Volume"].sum()
        avg_pm         = df_pm[df_pm["date"] < today]\
                           .groupby("date")["Volume"].sum().mean()
        pm_rvol        = pm_vol / avg_pm if avg_pm else None
        pm_high        = df_pm_today["High"].max()
        pm_close_max   = df_pm_today["Close"].max()

        # NEW: pre-market open & close trend
        pm_open_price  = df_pm_today["Open"].iloc[0]
        pm_close_price = df_pm_today["Close"].iloc[-1]
        pm_trend_pct   = (
            (pm_close_price - pm_open_price) / pm_open_price * 100
            if pm_open_price and pm_open_price != 0
            else None
        )

        first_time     = df_mkt_today.iloc[0]["Date"].time()
        f5_vol         = df_mkt_today[df_mkt_today["Date"].dt.time == first_time]\
                           ["Volume"].sum()
        avg_f5         = df_mkt[df_mkt["Date"].dt.time == first_time]\
                           .groupby("date")["Volume"].sum().mean()
        f5_rvol        = f5_vol / avg_f5 if avg_f5 else None

        mkt_high       = df_mkt_today["High"].iloc[0]
        market_close   = df_mkt_today["Close"].iloc[0]
        market_open    = df_mkt_today["Open"].iloc[0]
        marketbody     = market_open - market_close
        marketbody_ratio = (
            marketbody / market_open
            if (market_open is not None and market_open != 0)
            else None
        )


        CloseGap_Ratio = (
            (pm_close_max - market_close) / pm_close_max
            if (pm_close_max is not None and pm_close_max > 0)
            else None
        )


        fs             = float_map.get(ticker)
        pm_turn        = pm_vol / fs if fs and fs > 0 else None
        f5_turn        = f5_vol / fs if fs and fs > 0 else None
        turnover_ratio = (
            pm_turn / f5_turn
            if (pm_turn is not None and f5_turn is not None and f5_turn > 0)
            else None
        )

        # 1) Pre-market up/down volume
        pm_green_vol = df_pm_today.loc[
            df_pm_today["Close"] > df_pm_today["Open"],
            "Volume"
        ].sum()
        pm_red_vol   = df_pm_today.loc[
            df_pm_today["Close"] < df_pm_today["Open"],
            "Volume"
        ].sum()

        # 2) Sentiment flag or ratio
        if pm_green_vol + pm_red_vol > 0:
            pm_sentiment_ratio = pm_green_vol / (pm_green_vol + pm_red_vol)
            # e.g. 0.8 means 80% of the volume traded on up-ticks
            pm_sentiment_flag  = "bullish" if pm_green_vol > pm_red_vol else "bearish"
        else:
            pm_sentiment_ratio = None
            pm_sentiment_flag  = None


        res.update({
            "Status":            "processed",
            "Date":              today,
            "Gap%":              round(gap_pct,2),
            "PreMarketVol":      int(pm_vol),
            "AvgPreMarketVol":   round(avg_pm,0) if avg_pm else None,
            "PM_RVOL":           round(pm_rvol,2) if pm_rvol else None,
            "First5Vol":         int(f5_vol),
            "AvgFirst5Vol":      round(avg_f5,0) if avg_f5 else None,
            "F5_RVOL":           round(f5_rvol,2) if f5_rvol else None,
            "PreMarketHigh":     pm_high,
            "PreMarketCloseMax": pm_close_max,
            "MarketHigh":        mkt_high,
            "MarketClose":       market_close,
            "MarketOpen":        market_open,
            "MarketBody":        marketbody,
            "MarketBodyRatio":   marketbody_ratio,
            "FloatShares":       fs,
            "PM_Turnover":       round(pm_turn,4) if pm_turn else None,
            "F5_Turnover":       round(f5_turn,4) if f5_turn else None,
            "TurnoverRatio":     round(turnover_ratio,4) if turnover_ratio else None,
            "CloseGap_Ratio":    round(CloseGap_Ratio,4) if CloseGap_Ratio else None,
            # ─── NEW PRE-MARKET TREND/SENTIMENT METRICS ────────────────────────────────
            "PreMarketTrend%":      round(pm_trend_pct,2)    if pm_trend_pct   is not None else None,
            "PM_VolSentiment%":     round(pm_sentiment_ratio*100,1) if pm_sentiment_ratio is not None else None,
            "PM_VolSentimentFlag":  pm_sentiment_flag
        })

        # 5) Filter criteria
        matched = (mkt_high  > pm_high) and (market_close > pm_close_max)
        res["Matched"] = matched
        results.append(res)

    # 6) Save to Excel
    df_results  = pd.DataFrame(results)
    # ensure no NaN in Matched
    df_results["Matched"] = df_results["Matched"].fillna(False)

    # Add RVOLRatio column
    df_results["RVOLRatio"] = df_results["PM_RVOL"] / df_results["F5_RVOL"]

    df_filtered = df_results[df_results["Matched"]]

    # # Revised filter & sort for df_final
    # df_final = (
    #     df_results[
    #         (df_results["Matched"] == True) &
    #         (df_results["Gap%"].abs() >= 2) &
    #         (df_results["PM_RVOL"]  >= 1.5) &
    #         (df_results["RVOLRatio"] >= 2) &
    #         (df_results["F5_RVOL"]  >= 1)
    #     ]
    #     .sort_values("Gap%", ascending=False)
    #     .reset_index(drop=True)
    # )

    neg = (
        (df_results["Gap%"] >= -2) &
        (df_results["Gap%"] <   -0.5) &
        (df_results["MarketBodyRatio"] >  -0.03) &
        (df_results["MarketBodyRatio"] <=  0.05)
    )

    pos = (
        (df_results["Gap%"] >    3) &
        (df_results["Gap%"] <=  20) &
        (df_results["MarketBodyRatio"] >  -0.02) &
        (df_results["MarketBodyRatio"] <   0.03)
    )


    df_final = (
        df_results[
            (
                (neg | pos)
                & (df_results["PreMarketVol"] > 1000)
                & (df_results["PM_RVOL"]     >= 1.2)
                & (df_results["F5_RVOL"]     >= 1.2)
                #& (df_results["MarketBodyRatio"]     <= 0.05)
                & (df_results["FloatShares"] < 600_000_000)
            )
        ]
        .sort_values(by="Gap%", key=lambda c: c.abs(), ascending=False)
        .reset_index(drop=True)
        .head(60)
    )



    df_SmallPrice = (
        df_results[
            (df_results["MarketClose"] < 25) &
            (df_results["RVOLRatio"] > 1) &
            (df_results["TurnoverRatio"] > 1) &
            (df_results["FloatShares"] < 150000000)
        ]
        .sort_values("RVOLRatio", ascending=False)
        .reset_index(drop=True)
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="FullResults",   index=False)
        df_filtered.to_excel(writer, sheet_name="FilteredList", index=False)
        df_final.to_excel(writer, sheet_name="Long", index=False)
        df_final.to_excel(writer, sheet_name="Short", index=False)
        df_SmallPrice.to_excel(writer, sheet_name="SmallPrice", index=False)

    print(f"✅ Done: {len(df_results)} tickers processed, {len(df_filtered)} matched → {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()
