"""
app_credit_spread.py

Streamlit page for the credit spread monitor. Self-updates on a schedule
(via cached fetches with a TTL) rather than requiring a manual re-run.

Run standalone:
    streamlit run app_credit_spread.py

Or merge into your existing app.py: copy the "PAGE BODY" section below
into wherever you want this panel to appear, and keep credit_spread_monitor.py
alongside it.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from credit_spread_monitor import (
    run_credit_check,
    PLAIN_ENGLISH_EXPLANATION,
    LOG_PATH,
)

# --------------------------------------------------------------------------
# Auto-refresh (optional dependency)
# --------------------------------------------------------------------------
# streamlit has no built-in "run this every N minutes" -- the standard way
# is the streamlit-autorefresh package, which just triggers a page rerun.
# Falls back gracefully if it isn't installed: pip install streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# --------------------------------------------------------------------------
# Cached data fetch -- this is what makes it "self-updating"
# --------------------------------------------------------------------------
# FRED's HY/IG OAS series update once per business day (T+1 morning), so an
# hourly cache is already more than fresh enough -- no point hammering the
# endpoint every rerun. Streamlit reruns the whole script on every
# interaction, so without this cache you'd refetch on every click.

@st.cache_data(ttl=3600, show_spinner="Fetching latest credit spread data...")
def get_credit_check():
    return run_credit_check(log=True)


def load_log_history():
    try:
        return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        return pd.DataFrame()


# ==========================================================================
# PAGE BODY -- copy this section into app.py if merging rather than running
# standalone
# ==========================================================================

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("📉 Credit Spread Monitor")

# --- refresh controls ---
col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    if HAS_AUTOREFRESH:
        refresh_minutes = st.slider("Auto-refresh every (minutes)", 5, 60, 15)
        st_autorefresh(interval=refresh_minutes * 60 * 1000, key="credit_spread_autorefresh")
    else:
        st.caption("Install `streamlit-autorefresh` for automatic page refresh. "
                   "Data itself still updates hourly via cache regardless.")
with col_c:
    if st.button("🔄 Refresh now"):
        get_credit_check.clear()
        st.rerun()

result = get_credit_check()
st.caption(f"Last data fetch: {result['timestamp']} (cached up to 1 hour)")

# --- headline metrics ---
m1, m2, m3 = st.columns(3)
m1.metric("HY OAS", f"{result['hy_oas_bps']:.0f} bps")
m2.metric("IG OAS", f"{result['ig_oas_bps']:.0f} bps")

status = result["status"]
status_display = {
    "WIDENING": "🔴 WIDENING",
    "WATCH": "🟡 WATCH",
    "STABLE": "🟢 STABLE",
}
m3.metric("Status", status_display.get(status, status))

# --- alerts ---
if result["alerts"]:
    st.error("**Confirmed alerts:**")
    for a in result["alerts"]:
        st.write(f"- {a}")
elif result.get("unconfirmed_notes"):
    st.warning("**Watch (unconfirmed — one signal only, likely noise):**")
    for n in result["unconfirmed_notes"]:
        st.write(f"- {n}")
else:
    st.success("No alerts. Spreads stable relative to recent regime.")

# --- explanation ---
with st.expander("How this works"):
    st.markdown(PLAIN_ENGLISH_EXPLANATION)

# --- historical chart: confirmed alerts vs S&P 500 ---
st.subheader("History: 2018–present")
st.caption(
    "Static file, built once with build_history.py and committed to git — "
    "persists across redeploys, unlike the live log above."
)

try:
    hist = pd.read_csv("historical_spread_data.csv", parse_dates=["date"], index_col="date")
    # CSV round-trip turns booleans into the strings "True"/"False" -- convert
    # back, otherwise `== True` silently matches nothing and no green dots show.
    if hist["confirmed"].dtype == object:
        hist["confirmed"] = hist["confirmed"].map({"True": True, "False": False})

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["hy_oas_bps"],
        name="HY OAS (bps)", yaxis="y1",
        line=dict(color="orange", width=1.5),
        connectgaps=False,
    ))

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["sp500_close"],
        name="S&P 500", yaxis="y2",
        line=dict(color="steelblue", width=1.5),
    ))

    confirmed = hist[hist["confirmed"] == True]

    # one dot per real episode (clustered + magnitude-filtered), at the
    # episode's start date -- that's the actionable moment, not the peak
    try:
        episodes = pd.read_csv(
            "historical_spread_episodes.csv",
            parse_dates=["start", "peak_date", "calm_date"],
        )
        episode_start_bps = hist.reindex(episodes["start"])["hy_oas_bps"].values
        fig.add_trace(go.Scatter(
            x=episodes["start"], y=episode_start_bps,
            mode="markers", name="Episode start",
            marker=dict(color="green", size=9, symbol="circle"),
            yaxis="y1",
            hovertemplate="Episode started %{x}<extra></extra>",
        ))
        n_episodes = len(episodes)
    except FileNotFoundError:
        episodes = pd.DataFrame()
        n_episodes = 0
        st.warning(
            "historical_spread_episodes.csv not found — showing raw daily dots only. "
            "Rerun build_history.py to generate the clustered episode view."
        )
        fig.add_trace(go.Scatter(
            x=confirmed.index, y=confirmed["hy_oas_bps"],
            mode="markers", name="Confirmed alert (unclustered)",
            marker=dict(color="green", size=7),
            yaxis="y1",
        ))

    fig.update_layout(
        yaxis=dict(title="HY OAS (bps)", side="left"),
        yaxis2=dict(title="S&P 500", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
        height=500,
    )

    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"🟢 {n_episodes} real episodes (consecutive confirmed days clustered, "
        f"filtered to those exceeding their own trailing-year median by 75bps+). "
        f"Dot marks when each episode started."
    )

    if not episodes.empty:
        with st.expander("All episodes"):
            display_ep = episodes.copy()
            display_ep["duration_days"] = (display_ep["calm_date"] - display_ep["start"]).dt.days
            st.dataframe(
                display_ep[["start", "peak_date", "peak_bps", "calm_date", "duration_days"]]
                    .sort_values("start", ascending=False),
                width='stretch',
            )

except FileNotFoundError:
    st.info(
        "No historical file yet. Run `python build_history.py` locally, then "
        "`git add historical_spread_data.csv && git commit -m \"add history\" && git push` "
        "to make it appear here."
    )
except Exception as e:
    st.error(f"Couldn't load the history chart: {e}")
    st.caption(
        "Likely a malformed or incomplete historical_spread_data.csv. "
        "Try rerunning `python build_history.py` locally and re-pushing."
    )

# --- statistics: does this actually hold up? ---
st.subheader("Does this actually hold up statistically?")

try:
    daily_corr_df = pd.read_csv("daily_correlation.csv")
    dc = daily_corr_df.iloc[0]
    dc_col1, dc_col2, dc_col3 = st.columns(3)
    dc_col1.metric("Daily correlation (r)", f"{dc['correlation']:.3f}")
    dc_col2.metric("p-value", f"{dc['p_value']:.2e}")
    dc_col3.metric("Sample size", f"{int(dc['n']):,} days")
    st.caption("Negative r = when spreads widen day-to-day, the S&P tends to fall that same day, and vice versa.")
except FileNotFoundError:
    pass

try:
    lead_lag_df = pd.read_csv("lead_lag_analysis.csv")
    event_df = pd.read_csv("event_study.csv")

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("**Lead-lag correlation**")
        st.caption("HY OAS 5-day change vs S&P forward return at each lag. More negative = stronger inverse relationship.")
        st.line_chart(lead_lag_df.set_index("lag_days")["correlation"])
        best_row = lead_lag_df.loc[lead_lag_df["correlation"].idxmin()]
        st.metric(
            f"Strongest signal at lag {int(best_row['lag_days'])}d",
            f"r = {best_row['correlation']}",
            f"p = {best_row['p_value']:.2e}",
        )

    with col_y:
        st.markdown("**Event study: returns after a confirmed episode starts**")
        st.caption("Comparing average S&P return after episodes vs the market's normal average return over the same length window.")
        display_event = event_df.copy()
        display_event["significant_at_5pct"] = display_event["significant_at_5pct"].map(
            {True: "✅ yes", False: "❌ no", "True": "✅ yes", "False": "❌ no"}
        )
        st.dataframe(
            display_event.rename(columns={
                "horizon_days": "Days after",
                "mean_return_after_episode": "Return after episode (%)",
                "baseline_mean_return_pct": "Normal avg return (%)",
                "difference_pct": "Difference (pp)",
                "p_value": "p-value",
                "significant_at_5pct": "Significant?",
            })[["Days after", "Return after episode (%)", "Normal avg return (%)",
                "Difference (pp)", "p-value", "Significant?"]],
            width='stretch',
            hide_index=True,
        )

    st.caption(
        "Read this as: **p < 0.05** means the pattern is unlikely to be random chance. "
        "A negative 'Difference' means returns were worse than normal following a confirmed "
        "widening episode — which is the actual evidence for the thesis, not just the chart above."
    )

except FileNotFoundError:
    st.info(
        "No statistics file yet. Rerun `python build_history.py` (now computes these "
        "automatically), then commit `lead_lag_analysis.csv` and `event_study.csv`."
    )
except Exception as e:
    st.error(f"Couldn't load statistics: {e}")

# --- live log history ---
st.subheader("Live log (today onward)")
log_df = load_log_history()
if not log_df.empty:
    chart_df = log_df.set_index("timestamp")[["hy_oas_bps", "ig_oas_bps"]]
    st.line_chart(chart_df)
    with st.expander("Raw log"):
        st.dataframe(log_df.sort_values("timestamp", ascending=False), width='stretch')
else:
    st.info("No history yet — check back after this has run a few times.")
