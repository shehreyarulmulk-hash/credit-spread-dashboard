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
    fig.add_trace(go.Scatter(
        x=confirmed.index, y=confirmed["hy_oas_bps"],
        mode="markers", name="Confirmed alert",
        marker=dict(color="green", size=9, symbol="circle"),
        yaxis="y1",
        text=confirmed["detail"],
        hovertemplate="%{x}<br>HY OAS: %{y:.0f}bps<br>%{text}<extra></extra>",
    ))

    fig.update_layout(
        yaxis=dict(title="HY OAS (bps)", side="left"),
        yaxis2=dict(title="S&P 500", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🟢 green dots = confirmed widening signal (2+ of 3 ROC windows fired together). "
        "Compare against the S&P 500 line to see how far ahead of a drawdown the signal fired."
    )

    with st.expander("All confirmed alert dates"):
        st.dataframe(
            confirmed[["hy_oas_bps", "sp500_close", "detail"]].sort_index(ascending=False),
            use_container_width=True,
        )

except FileNotFoundError:
    st.info(
        "No historical file yet. Run `python build_history.py` locally, then "
        "`git add historical_spread_data.csv && git commit -m \"add history\" && git push` "
        "to make it appear here."
    )

# --- live log history ---
st.subheader("Live log (today onward)")
log_df = load_log_history()
if not log_df.empty:
    chart_df = log_df.set_index("timestamp")[["hy_oas_bps", "ig_oas_bps"]]
    st.line_chart(chart_df)
    with st.expander("Raw log"):
        st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)
else:
    st.info("No history yet — check back after this has run a few times.")
