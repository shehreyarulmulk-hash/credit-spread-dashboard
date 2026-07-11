"""
build_history.py

ONE-TIME (or occasional) build script. Run this LOCALLY on your Mac, not on
Streamlit Cloud -- it needs network access to pull data, and the output is a
static file you commit to git so it persists forever without needing a
database.

What it does:
  1. Pulls HY OAS from two sources and stitches them together:
     - GitHub mirror (1996-2021) for the 2018-2021 portion
     - Live FRED feed (rolling 3yr window) for the recent portion
     - NOTE: there is a known gap ~March 2021 to mid-2023 that neither free
       source covers (FRED restricted ICE BofA series to a 3yr window in
       April 2026; the old full-history mirror stops March 2021). The chart
       will show this as a visible gap rather than faking continuity.
  2. Pulls S&P 500 daily closes 2018-present via yfinance (continuous, no gap)
  3. Runs the SAME 2-of-3 confirmation logic from credit_spread_monitor.py
     across the whole history to find every "confirmed alert" date
  4. Saves everything to historical_spread_data.csv -- commit this file to
     git once and it's there permanently, no need to ever rerun this unless
     you want to extend the range or fill the 2021-2023 gap if a paid
     source becomes available.

Usage:
    python build_history.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import io
import requests

from credit_spread_monitor import ROC_ALERT_BPS, CONFIRMATION_REQUIRED

GITHUB_MIRROR_URL = "https://raw.githubusercontent.com/csaladenes/eco/main/BAMLH0A0HYM2.csv"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"

OUTPUT_PATH = "historical_spread_data.csv"


def fetch_github_mirror() -> pd.Series:
    """1996-2021 portion, in basis points."""
    resp = requests.get(GITHUB_MIRROR_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date")
    return df["value"] * 100  # percent -> bps


def fetch_fred_recent() -> pd.Series:
    """Rolling 3-year window from FRED, in basis points."""
    resp = requests.get(FRED_CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date")
    return df["value"] * 100


def fetch_sp500(start="2018-01-01") -> pd.Series:
    """Continuous S&P 500 close, no gap issues here."""
    data = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data.index = pd.to_datetime(data.index)
    return data


def compute_confirmed_alerts(hy_bps: pd.Series) -> pd.DataFrame:
    """
    Re-runs the same 2-of-3 ROC confirmation logic from
    credit_spread_monitor.py across the full history to flag every date a
    real (confirmed) widening signal fired.
    """
    hy_bps = hy_bps.sort_index()
    windows = {"5d": 5, "10d": 10, "20d": 20}
    records = []

    for i in range(len(hy_bps)):
        breached = 0
        detail = []
        for label, w in windows.items():
            if i >= w:
                roc = hy_bps.iloc[i] - hy_bps.iloc[i - w]
                threshold = ROC_ALERT_BPS[label]
                if roc >= threshold:
                    breached += 1
                    detail.append(f"{label}:+{roc:.0f}bps")
        records.append({
            "date": hy_bps.index[i],
            "hy_oas_bps": hy_bps.iloc[i],
            "num_breached": breached,
            "confirmed": breached >= CONFIRMATION_REQUIRED,
            "detail": ", ".join(detail),
        })

    return pd.DataFrame(records).set_index("date")


def main():
    print("Fetching GitHub mirror (1996-2021)...")
    mirror = fetch_github_mirror()

    print("Fetching live FRED window (last 3 years)...")
    recent = fetch_fred_recent()

    print("Stitching HY OAS series (gap may exist ~2021-2023, see docstring)...")
    hy_combined = pd.concat([mirror, recent])
    hy_combined = hy_combined[~hy_combined.index.duplicated(keep="last")]
    hy_combined = hy_combined.sort_index()
    hy_combined = hy_combined[hy_combined.index >= "2018-01-01"]

    # If there's a real gap between the two sources, insert an explicit NaN
    # row right after the mirror ends. Without this, Plotly draws a
    # straight line connecting the last pre-gap point to the first
    # post-gap point, which LOOKS like real data but isn't -- this makes
    # the gap show up as an honest break in the chart instead.
    gap_start = mirror.index.max()
    gap_end = recent.index.min()
    if gap_end > gap_start:
        nan_marker_date = gap_start + pd.Timedelta(days=1)
        hy_combined.loc[nan_marker_date] = np.nan
        hy_combined = hy_combined.sort_index()

    print("Fetching S&P 500 (2018-present)...")
    sp500 = fetch_sp500(start="2018-01-01")

    print("Computing confirmed-alert dates across full history...")
    alerts_df = compute_confirmed_alerts(hy_combined)

    print("Merging with S&P 500...")
    merged = alerts_df.join(sp500.rename("sp500_close"), how="outer")
    merged = merged.sort_index()

    merged.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_PATH}")

    confirmed_dates = merged[merged["confirmed"] == True]
    print(f"\n{len(confirmed_dates)} confirmed alert days found across the full history:")
    for date, row in confirmed_dates.iterrows():
        print(f"  {date.date()}  HY OAS={row['hy_oas_bps']:.0f}bps  ({row['detail']})")

    # flag the data gap explicitly so it's visible in the console too
    gap_start = mirror.index.max()
    gap_end = recent.index.min()
    if gap_end > gap_start:
        print(f"\nNOTE: data gap between {gap_start.date()} and {gap_end.date()} "
              f"-- no free source covers this window. Chart will show it as missing.")


if __name__ == "__main__":
    main()
