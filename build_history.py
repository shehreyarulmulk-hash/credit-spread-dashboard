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
from scipy import stats as scipy_stats

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


def build_clusters(alerts_df: pd.DataFrame, hy_bps: pd.Series, min_breach: int = CONFIRMATION_REQUIRED) -> pd.DataFrame:
    """
    Groups days where >= min_breach of the 3 ROC windows fired into
    episodes, filters by a relative magnitude floor, and finds each
    episode's calm-down date.

    min_breach=2 (default) is the "confirmed" trigger used elsewhere.
    min_breach=1 is a looser, earlier trigger -- fires sooner but with
    more false positives -- used to test whether an earlier trigger
    captures more lead time on the actual drawdown.

    Returns one row per real episode: start, peak_date, peak_bps,
    calm_date (end of shaded span), and the trailing median used for
    the floor (for transparency/debugging).
    """
    hy_bps = hy_bps.sort_index()
    trailing_median = hy_bps.rolling(TRAILING_WINDOW_DAYS, min_periods=30).median()

    trigger_dates = alerts_df.index[alerts_df["num_breached"] >= min_breach].sort_values()
    if len(trigger_dates) == 0:
        return pd.DataFrame(columns=["start", "peak_date", "peak_bps", "calm_date", "trailing_median_at_start"])

    # group into raw clusters, bridging gaps <= MAX_GAP_DAYS trading days
    date_list = list(hy_bps.index)
    idx_of = {d: i for i, d in enumerate(date_list)}

    clusters = []
    current = [trigger_dates[0]]
    for d in trigger_dates[1:]:
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


def compare_trigger_lead_time(confirmed_eps: pd.DataFrame, early_eps: pd.DataFrame) -> pd.DataFrame:
    """
    For each confirmed (2-of-3) episode, finds the early (1-of-3) episode
    whose span contains the confirmed episode's start date, and computes
    how many trading days earlier the early trigger fired. This is the
    actual lead-time gain from loosening the trigger, not a guess.
    """
    rows = []
    for _, conf in confirmed_eps.iterrows():
        match = early_eps[
            (early_eps["start"] <= conf["start"]) &
            (early_eps["calm_date"] >= conf["start"])
        ]
        if len(match) == 0:
            continue
        early_start = match.iloc[0]["start"]
        lead_days = (conf["start"] - early_start).days
        rows.append({
            "confirmed_start": conf["start"],
            "early_start": early_start,
            "calendar_days_earlier": lead_days,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Statistics: does spread widening actually precede price drops?
# --------------------------------------------------------------------------

def daily_correlation(hy_bps: pd.Series, sp500: pd.Series) -> dict:
    """
    Pearson correlation between day-to-day HY OAS changes (bps) and
    day-to-day S&P 500 returns (%). A real inverse relationship shows up
    as a negative, statistically significant r.
    """
    df = pd.DataFrame({"hy_chg": hy_bps.diff(), "sp_ret": sp500.pct_change()}).dropna()
    r, p = scipy_stats.pearsonr(df["hy_chg"], df["sp_ret"])
    return {"n": len(df), "correlation": round(r, 4), "p_value": p}


def lead_lag_analysis(hy_bps: pd.Series, sp500: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """
    For each lag k = 1..max_lag trading days, correlates today's HY OAS
    5-day rate of change against the S&P 500's forward return over the
    NEXT k days. If spreads genuinely lead price, the correlation should
    be negative and get stronger (more negative) at some specific lag,
    not just at lag=0 -- that's what actually demonstrates "spread moves
    first" rather than "they move together."
    """
    hy_5d_roc = hy_bps.diff(5)
    results = []
    for lag in range(1, max_lag + 1):
        fwd_ret = sp500.pct_change(lag).shift(-lag)
        df = pd.DataFrame({"hy_roc": hy_5d_roc, "fwd_ret": fwd_ret}).dropna()
        if len(df) < 30:
            continue
        r, p = scipy_stats.pearsonr(df["hy_roc"], df["fwd_ret"])
        results.append({"lag_days": lag, "correlation": round(r, 4), "p_value": p, "n": len(df)})
    return pd.DataFrame(results)


def event_study(episodes: pd.DataFrame, sp500: pd.Series, horizons=(5, 10, 20, 60)) -> pd.DataFrame:
    """
    For each real episode, measures the S&P 500's actual forward return
    from the episode's START date over each horizon, then compares that
    sample's mean against ALL possible equivalent-length windows in the
    full history (the unconditional baseline) via a one-sample t-test.

    A significant negative result means: returns following a confirmed
    widening episode are worse than a random window would typically be --
    that's the actual statistical proof, not just an eyeballed chart.
    """
    rows = []
    for h in horizons:
        # unconditional baseline: every possible h-day forward return in the full series
        baseline = sp500.pct_change(h).shift(-h).dropna()

        # conditional: forward return starting from each episode's start date
        episode_returns = []
        for start_date in episodes["start"]:
            if start_date not in sp500.index:
                future_idx = sp500.index[sp500.index >= start_date]
                if len(future_idx) == 0:
                    continue
                start_date = future_idx[0]
            loc = sp500.index.get_loc(start_date)
            if loc + h >= len(sp500):
                continue
            ret = sp500.iloc[loc + h] / sp500.iloc[loc] - 1
            episode_returns.append(ret)

        if len(episode_returns) < 2:
            continue

        episode_returns = np.array(episode_returns)
        t_stat, p_val = scipy_stats.ttest_1samp(episode_returns, baseline.mean())

        rows.append({
            "horizon_days": h,
            "n_episodes": len(episode_returns),
            "mean_return_after_episode": round(episode_returns.mean() * 100, 2),
            "baseline_mean_return_pct": round(baseline.mean() * 100, 2),
            "difference_pct": round((episode_returns.mean() - baseline.mean()) * 100, 2),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "significant_at_5pct": bool(p_val < 0.05),
        })

    return pd.DataFrame(rows)


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
    clusters_df = build_clusters(alerts_df, hy_combined, min_breach=CONFIRMATION_REQUIRED)
    clusters_df.to_csv("historical_spread_episodes.csv", index=False)

    print("Building early-trigger episodes (1-of-3 windows, for lead-time comparison)...")
    early_clusters_df = build_clusters(alerts_df, hy_combined, min_breach=1)
    early_clusters_df.to_csv("historical_spread_episodes_early.csv", index=False)

    lead_time_df = compare_trigger_lead_time(clusters_df, early_clusters_df)
    lead_time_df.to_csv("early_trigger_lead_time.csv", index=False)

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

    # -------------------- statistics --------------------
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    corr = daily_correlation(hy_combined, sp500)
    pd.DataFrame([corr]).to_csv("daily_correlation.csv", index=False)
    print(f"\nDaily correlation (HY OAS change vs S&P return): "
          f"r={corr['correlation']}, p={corr['p_value']:.2e}, n={corr['n']}")
    sig = "significant" if corr["p_value"] < 0.05 else "NOT significant"
    print(f"  -> {sig} at 5% level. Negative r = spreads up, price down (as expected).")

    lead_lag_df = lead_lag_analysis(hy_combined, sp500)
    lead_lag_df.to_csv("lead_lag_analysis.csv", index=False)
    if not lead_lag_df.empty:
        best = lead_lag_df.loc[lead_lag_df["correlation"].idxmin()]
        print(f"\nLead-lag analysis (saved to lead_lag_analysis.csv):")
        print(f"  Strongest negative correlation at lag={int(best['lag_days'])} days: "
              f"r={best['correlation']}, p={best['p_value']:.2e}")
        print(f"  -> Spread widening tends to precede price weakness by ~{int(best['lag_days'])} trading days")

    event_df = event_study(clusters_df, sp500)
    event_df.to_csv("event_study.csv", index=False)
    print(f"\nEvent study (confirmed, 2-of-3 trigger — saved to event_study.csv):")
    for _, row in event_df.iterrows():
        sig_flag = "***" if row["significant_at_5pct"] else "(not significant)"
        print(f"  {int(row['horizon_days'])}d after episode start: "
              f"mean return {row['mean_return_after_episode']}% vs baseline "
              f"{row['baseline_mean_return_pct']}% (diff {row['difference_pct']}pp, "
              f"p={row['p_value']}) {sig_flag}")

    event_df_early = event_study(early_clusters_df, sp500)
    event_df_early.to_csv("event_study_early.csv", index=False)
    print(f"\nEvent study (EARLY, 1-of-3 trigger — {len(early_clusters_df)} episodes vs "
          f"{len(clusters_df)} confirmed — saved to event_study_early.csv):")
    for _, row in event_df_early.iterrows():
        sig_flag = "***" if row["significant_at_5pct"] else "(not significant)"
        print(f"  {int(row['horizon_days'])}d after episode start: "
              f"mean return {row['mean_return_after_episode']}% vs baseline "
              f"{row['baseline_mean_return_pct']}% (diff {row['difference_pct']}pp, "
              f"p={row['p_value']}) {sig_flag}")

    if not lead_time_df.empty:
        avg_lead = lead_time_df["calendar_days_earlier"].mean()
        print(f"\nLead time gained by using the earlier (1-of-3) trigger:")
        print(f"  Matched {len(lead_time_df)} of {len(clusters_df)} confirmed episodes to an earlier trigger")
        print(f"  Average lead time gained: {avg_lead:.1f} calendar days")
        print(f"  Trade-off: {len(early_clusters_df)} early episodes vs {len(clusters_df)} confirmed "
              f"({len(early_clusters_df) - len(clusters_df)} extra episodes, i.e. potential false positives)")

    # flag the data gap explicitly so it's visible in the console too
    gap_start = mirror.index.max()
    gap_end = recent.index.min()
    if gap_end > gap_start:
        print(f"\nNOTE: data gap between {gap_start.date()} and {gap_end.date()} "
              f"-- no free source covers this window. Chart will show it as missing.")


if __name__ == "__main__":
    main()
