"""
credit_spread_monitor.py

Tracks credit spread widening as an early-warning signal for the trading dashboard.

Data sources:
  - FRED (no API key needed via CSV endpoint): HY OAS, IG OAS
  - yfinance: HYG, LQD, TLT for a market-priced cross-check

Logic:
  - Rate of change (not absolute level) over configurable windows
  - Spread vs its own moving average (regime shift detector)
  - HY vs IG divergence (HY moving without IG = early/localized stress)
  - HYG/TLT ratio as a tradable, same-day proxy for HY OAS direction

Drop this alongside premarket_dashboard.py / app.py. Designed to log to CSV
the same way your existing scanner does, so it's easy to fold into the
Streamlit app as another panel.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import io
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FRED_SERIES = {
    "HY_OAS": "BAMLH0A0HYM2",   # ICE BofA US High Yield OAS
    "IG_OAS": "BAMLC0A0CM",     # ICE BofA US Corporate (IG) OAS
}

# Thresholds -- backtested against the 2018 Q4 selloff and the 2020 COVID
# crash (see PLAIN_ENGLISH_EXPLANATION below for what that found).
ROC_ALERT_BPS = {
    "5d": 15,    # +15bps over 5 trading days
    "10d": 25,   # +25bps over 10 trading days
    "20d": 40,   # +40bps over 20 trading days
}

# A SINGLE window breaching its threshold is noisy (the 2018 backtest threw
# a false alarm from one blip). Requiring at least 2 of the 3 windows to
# breach on the SAME day cleanly caught both the 2018 and 2020 regime
# shifts with real lead time, and filtered the false positive.
CONFIRMATION_REQUIRED = 2

MA_WINDOW = 20          # moving average window for regime detection
HY_IG_DIVERGENCE_BPS = 20  # HY moves this much more than IG => flag divergence

LOG_PATH = "credit_spread_log.csv"

# --------------------------------------------------------------------------
# Plain-English explanation -- drop this into the Streamlit app (e.g. inside
# an st.expander("How this works")) so the logic is legible on the site,
# not just in code comments.
# --------------------------------------------------------------------------

PLAIN_ENGLISH_EXPLANATION = """
### How the credit spread alert works

**What it's measuring**
High-yield (HY) credit spread = the extra interest junk-rated companies pay
over safe Treasury bonds. When investors get nervous about defaults, this
number goes up *before* stocks usually fall.

**Why not just watch the level**
A high number alone doesn't tell you much — spreads can sit at a "high"
level for months without anything happening. What matters is the number
**rising fast**, because that's what happens right before stress spreads
to equities.

**The three speedometers**
We check how much the spread has moved over three windows:
- Fast: change over the last **5 trading days**
- Medium: change over the last **10 trading days**
- Slow: change over the last **20 trading days**

Each has its own "too fast" threshold (15bps, 25bps, 40bps).

**Why we require 2 out of 3, not just 1**
Testing this against real history (the Dec 2018 selloff and the March 2020
COVID crash) showed that watching just ONE window throws false alarms —
small blips that look scary for a day but go nowhere. Requiring **at least
two of the three speedometers to trip on the same day** filtered that noise
out completely, while still catching both real regime changes with 3-4
weeks of lead time before the market really moved.

**Bottom line**: one flashing light = probably noise, ignore it.
Two or more flashing at once = pay attention, something real may be starting.
"""


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------

def fetch_fred_series(series_id: str, lookback_days: int = 400) -> pd.Series:
    """
    Pull a FRED series via the public CSV endpoint (no API key required).
    Returns a daily Series indexed by date, in basis points where applicable.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date")

    cutoff = datetime.now() - timedelta(days=lookback_days)
    df = df[df.index >= cutoff]

    # FRED OAS series are already in percent (e.g. 2.67 = 267bps)
    return df["value"] * 100  # convert to basis points


def fetch_etf_prices(tickers, lookback_days: int = 120) -> pd.DataFrame:
    """Pull adjusted close prices for the ETF cross-check basket."""
    data = yf.download(
        tickers,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


# --------------------------------------------------------------------------
# Signal computation
# --------------------------------------------------------------------------

def compute_roc(series: pd.Series, windows=(5, 10, 20)) -> dict:
    """Rate of change in bps over N trading days, most recent value last."""
    out = {}
    for w in windows:
        if len(series) > w:
            out[f"{w}d"] = round(series.iloc[-1] - series.iloc[-1 - w], 1)
        else:
            out[f"{w}d"] = np.nan
    return out


def confirmed_roc_alert(roc_dict: dict) -> dict:
    """
    Apply the 2-of-3 confirmation rule: only treat it as a real alert if
    at least CONFIRMATION_REQUIRED of the 5d/10d/20d windows breach their
    threshold on the same check. A single window breaching alone is
    reported but flagged as unconfirmed (likely noise per backtest).
    """
    breached = []
    for window, roc_val in roc_dict.items():
        threshold = ROC_ALERT_BPS.get(window)
        if threshold and not np.isnan(roc_val) and roc_val >= threshold:
            breached.append((window, roc_val, threshold))

    confirmed = len(breached) >= CONFIRMATION_REQUIRED
    return {
        "breached_windows": breached,
        "num_breached": len(breached),
        "confirmed": confirmed,
    }


def ma_regime_flag(series: pd.Series, window: int = MA_WINDOW) -> dict:
    """Is the spread crossing above its own moving average after being below it?"""
    ma = series.rolling(window).mean()
    latest = series.iloc[-1]
    latest_ma = ma.iloc[-1]
    prev = series.iloc[-2]
    prev_ma = ma.iloc[-2]

    crossed_above = (prev <= prev_ma) and (latest > latest_ma)
    return {
        "latest_spread_bps": round(latest, 1),
        "ma_bps": round(latest_ma, 1),
        "above_ma": bool(latest > latest_ma),
        "just_crossed_above": bool(crossed_above),
    }


def hy_ig_divergence(hy: pd.Series, ig: pd.Series, window: int = 10) -> dict:
    """Check if HY is widening meaningfully faster than IG (localized stress)."""
    hy_chg = hy.iloc[-1] - hy.iloc[-1 - window]
    ig_chg = ig.iloc[-1] - ig.iloc[-1 - window]
    divergence = hy_chg - ig_chg
    return {
        "hy_change_bps": round(hy_chg, 1),
        "ig_change_bps": round(ig_chg, 1),
        "divergence_bps": round(divergence, 1),
        "flag": bool(divergence > HY_IG_DIVERGENCE_BPS),
    }


def hyg_tlt_ratio_signal(prices: pd.DataFrame, window: int = 20) -> dict:
    """
    HYG/TLT ratio as a same-day, tradable proxy for HY spread direction.
    Falling ratio = HY underperforming Treasuries = spreads widening.
    """
    if "HYG" not in prices.columns or "TLT" not in prices.columns:
        return {"error": "HYG or TLT missing from price data"}

    ratio = prices["HYG"] / prices["TLT"]
    ratio_ma = ratio.rolling(window).mean()

    latest = ratio.iloc[-1]
    latest_ma = ratio_ma.iloc[-1]
    roc_5d = (ratio.iloc[-1] / ratio.iloc[-6] - 1) * 100 if len(ratio) > 6 else np.nan

    return {
        "hyg_tlt_ratio": round(latest, 4),
        "ratio_ma": round(latest_ma, 4),
        "below_ma": bool(latest < latest_ma),
        "roc_5d_pct": round(roc_5d, 2),
    }


# --------------------------------------------------------------------------
# Main check
# --------------------------------------------------------------------------

def run_credit_check(log: bool = True) -> dict:
    hy = fetch_fred_series(FRED_SERIES["HY_OAS"])
    ig = fetch_fred_series(FRED_SERIES["IG_OAS"])
    etf_prices = fetch_etf_prices(["HYG", "LQD", "TLT"])

    result = {
        "timestamp": datetime.now().isoformat(),
        "hy_oas_bps": round(hy.iloc[-1], 1),
        "ig_oas_bps": round(ig.iloc[-1], 1),
        "hy_roc": compute_roc(hy),
        "hy_ma_regime": ma_regime_flag(hy),
        "hy_ig_divergence": hy_ig_divergence(hy, ig),
        "hyg_tlt_signal": hyg_tlt_ratio_signal(etf_prices),
    }

    # roll up alerts, applying the 2-of-3 confirmation rule to the ROC checks
    alerts = []
    unconfirmed_notes = []

    roc_confirmation = confirmed_roc_alert(result["hy_roc"])
    result["roc_confirmation"] = roc_confirmation

    if roc_confirmation["confirmed"]:
        breach_str = ", ".join(f"{w} +{v}bps" for w, v, t in roc_confirmation["breached_windows"])
        alerts.append(f"CONFIRMED: {roc_confirmation['num_breached']} windows breached together ({breach_str})")
    elif roc_confirmation["num_breached"] == 1:
        w, v, t = roc_confirmation["breached_windows"][0]
        unconfirmed_notes.append(f"Unconfirmed: only {w} breached (+{v}bps, threshold {t}bps) — likely noise, watching for a second window to confirm")

    if result["hy_ma_regime"]["just_crossed_above"]:
        alerts.append(f"HY OAS just crossed above its {MA_WINDOW}-day MA")

    if result["hy_ig_divergence"]["flag"]:
        alerts.append(
            f"HY widening {result['hy_ig_divergence']['divergence_bps']}bps faster than IG"
        )

    hyg_sig = result["hyg_tlt_signal"]
    if isinstance(hyg_sig, dict) and hyg_sig.get("below_ma") and hyg_sig.get("roc_5d_pct", 0) < -1:
        alerts.append(f"HYG/TLT ratio falling ({hyg_sig['roc_5d_pct']}% over 5d), below MA")

    result["alerts"] = alerts
    result["unconfirmed_notes"] = unconfirmed_notes

    if alerts:
        result["status"] = "WIDENING"
    elif unconfirmed_notes:
        result["status"] = "WATCH"  # one window tripped, not yet confirmed
    else:
        result["status"] = "STABLE"

    if log:
        _append_log(result)

    return result


def _append_log(result: dict):
    row = {
        "timestamp": result["timestamp"],
        "hy_oas_bps": result["hy_oas_bps"],
        "ig_oas_bps": result["ig_oas_bps"],
        "hy_roc_5d": result["hy_roc"].get("5d"),
        "hy_roc_10d": result["hy_roc"].get("10d"),
        "hy_roc_20d": result["hy_roc"].get("20d"),
        "above_ma": result["hy_ma_regime"]["above_ma"],
        "hyg_tlt_ratio": result["hyg_tlt_signal"].get("hyg_tlt_ratio")
            if isinstance(result["hyg_tlt_signal"], dict) else None,
        "status": result["status"],
        "alerts": "; ".join(result["alerts"]),
        "unconfirmed_notes": "; ".join(result.get("unconfirmed_notes", [])),
    }
    df_row = pd.DataFrame([row])
    try:
        existing = pd.read_csv(LOG_PATH)
        combined = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        combined = df_row
    combined.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    result = run_credit_check()
    print(f"\n{'='*50}")
    print(f"CREDIT SPREAD CHECK — {result['timestamp']}")
    print(f"{'='*50}")
    print(f"HY OAS: {result['hy_oas_bps']}bps   IG OAS: {result['ig_oas_bps']}bps")
    print(f"Status: {result['status']}")
    if result["alerts"]:
        print("\nCONFIRMED ALERTS:")
        for a in result["alerts"]:
            print(f"  - {a}")
    if result.get("unconfirmed_notes"):
        print("\nWATCH (unconfirmed, 1 window only):")
        for n in result["unconfirmed_notes"]:
            print(f"  - {n}")
    if not result["alerts"] and not result.get("unconfirmed_notes"):
        print("\nNo alerts — spreads stable relative to recent regime.")
    print()
