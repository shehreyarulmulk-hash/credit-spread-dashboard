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


# Cluster tuning -- see discussion: groups consecutive confirmed days into
# one "episode" instead of flagging every individual day, then filters out
# episodes too small to matter, then extends each episode's shaded window
# from first trigger through to when spreads actually calm back down
# (not just to the peak) -- gives a duration reading, not just a trigger point.
MAX_GAP_DAYS = 3          # bridge gaps of up to N trading days within a cluster
TRAILING_WINDOW_DAYS = 252  # ~1 trading year, for the relative magnitude floor
MAGNITUDE_FLOOR_BPS = 75   # cluster peak must exceed trailing median by this much
CALM_DOWN_MARGIN_BPS = 75  # "calmed down" = back under trailing median + this


def build_clusters(alerts_df: pd.DataFrame, hy_bps: pd.Series) -> pd.DataFrame:
    """
    Groups confirmed days into episodes, filters by a relative magnitude
    floor, and finds each episode's calm-down date.

    Returns one row per real episode: start, peak_date, peak_bps,
    calm_date (end of shaded span), and the trailing median used for
    the floor (for transparency/debugging).
    """
    hy_bps = hy_bps.sort_index()
    trailing_median = hy_bps.rolling(TRAILING_WINDOW_DAYS, min_periods=30).median()

    confirmed_dates = alerts_df.index[alerts_df["confirmed"]].sort_values()
    if len(confirmed_dates) == 0:
        return pd.DataFrame(columns=["start", "peak_date", "peak_bps", "calm_date", "trailing_median_at_start"])

    # group into raw clusters, bridging gaps <= MAX_GAP_DAYS trading days
    date_list = list(hy_bps.index)
    idx_of = {d: i for i, d in enumerate(date_list)}

    clusters = []
    current = [confirmed_dates[0]]
    for d in confirmed_dates[1:]:
        gap = idx_of[d] - idx_of[current[-1]]
        if gap <= MAX_GAP_DAYS:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]
    clusters.append(current)

    episodes = []
    for cluster in clusters:
        start = cluster[0]
        window = hy_bps.loc[start:cluster[-1]]
        peak_date = window.idxmax()
        peak_bps = window.max()

        floor_ref = trailing_median.loc[start] if not pd.isna(trailing_median.loc[start]) else hy_bps.loc[:start].median()

        # magnitude floor: skip small episodes that triggered the ROC logic
        # but never actually got meaningfully elevated vs their own regime
        if peak_bps < floor_ref + MAGNITUDE_FLOOR_BPS:
            continue

        # extend to calm-down: walk forward from peak until spread falls
        # back under trailing_median + CALM_DOWN_MARGIN_BPS, or data ends
        calm_target = floor_ref + CALM_DOWN_MARGIN_BPS
        post_peak = hy_bps.loc[peak_date:]
        calmed = post_peak[post_peak <= calm_target]
        calm_date = calmed.index[0] if len(calmed) > 0 else hy_bps.index[-1]

        episodes.append({
            "start": start,
            "peak_date": peak_date,
            "peak_bps": round(peak_bps, 1),
            "calm_date": calm_date,
            "trailing_median_at_start": round(floor_ref, 1),
        })

    return pd.DataFrame(episodes)


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

    print("Grouping into episodes (clustering, magnitude floor, calm-down extension)...")
    clusters_df = build_clusters(alerts_df, hy_combined)
    clusters_df.to_csv("historical_spread_episodes.csv", index=False)

    print("Merging with S&P 500...")
    merged = alerts_df.join(sp500.rename("sp500_close"), how="outer")
    merged = merged.sort_index()

    merged.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(merged)} rows to {OUTPUT_PATH}")
    print(f"Saved {len(clusters_df)} episodes to historical_spread_episodes.csv")

    print(f"\n{len(clusters_df)} real episodes found (raw confirmed days collapsed via "
          f"clustering + {MAGNITUDE_FLOOR_BPS}bps magnitude floor):")
    for _, row in clusters_df.iterrows():
        duration = (row["calm_date"] - row["start"]).days
        print(f"  {row['start'].date()} -> {row['calm_date'].date()} "
              f"({duration}d)  peak={row['peak_bps']:.0f}bps on {row['peak_date'].date()}")

    # flag the data gap explicitly so it's visible in the console too
    gap_start = mirror.index.max()
    gap_end = recent.index.min()
    if gap_end > gap_start:
        print(f"\nNOTE: data gap between {gap_start.date()} and {gap_end.date()} "
              f"-- no free source covers this window. Chart will show it as missing.")


if __name__ == "__main__":
    main()
