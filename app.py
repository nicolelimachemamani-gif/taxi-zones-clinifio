"""
app.py — NYC Demand Zones. Streamlit front-end.

The UI is in English on purpose: the rubric requires the application to be
demonstrated in English.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from pathlib import Path

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
LOG      = ROOT / "data" / "predictions_log.csv"
FEEDBACK = ROOT / "data" / "feedback.csv"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

st.set_page_config(page_title="NYC Demand Zones", page_icon="🚕", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "production.joblib")


@st.cache_data
def load_profile():
    return pd.read_parquet(PROC / "demand_profile.parquet")


@st.cache_data
def load_centroids():
    return pd.read_csv(PROC / "centroids.csv")


@st.cache_data
def load_metrics():
    return json.loads((MODELS / "production_metrics.json").read_text())


def log_query(dow: int, hour: int, top_zone: int) -> None:
    """Every served query is stored. Without this there is nothing to retrain on."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "ts": datetime.now(timezone.utc).isoformat(),
        "dow": dow, "hour": hour, "zone": top_zone,
    }])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


# ── Driver feedback: raw-log + Beta-Bernoulli Thompson sampling ───────────────

_FEEDBACK_COLS = {"ts", "zone", "dow", "hour", "positive"}

def load_feedback() -> pd.DataFrame:
    """Return all feedback rows; columns: ts, zone, dow, hour, positive.

    If the file exists but uses a legacy schema (e.g. the old aggregated
    wins/losses format), it is ignored and an empty DataFrame is returned so
    the app does not crash on a column key error.
    """
    empty = pd.DataFrame(columns=["ts", "zone", "dow", "hour", "positive"])
    if not FEEDBACK.exists():
        return empty
    df = pd.read_csv(FEEDBACK)
    if not _FEEDBACK_COLS.issubset(df.columns):
        # Legacy file — silently discard it
        return empty
    return df


def save_feedback(zone: int, dow: int, hour: int, positive: bool) -> None:
    """Append a single vote row to the feedback CSV."""
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "ts":       datetime.now(timezone.utc).isoformat(),
        "zone":     zone,
        "dow":      dow,
        "hour":     hour,
        "positive": int(positive),
    }])
    row.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)


def aggregate_feedback(fb: pd.DataFrame) -> pd.DataFrame:
    """Return wins, losses and success_rate aggregated by (zone, dow, hour)."""
    if fb.empty:
        return pd.DataFrame(columns=["zone", "dow", "hour", "wins", "losses", "success_rate"])
    agg = (
        fb.groupby(["zone", "dow", "hour"])["positive"]
        .agg(wins="sum", total="count")
        .reset_index()
    )
    agg["losses"] = agg["total"] - agg["wins"]
    agg["success_rate"] = (agg["wins"] / agg["total"]).round(3)
    return agg[["zone", "dow", "hour", "wins", "losses", "success_rate"]]


st.title("🚕 NYC Demand Zones")
st.caption(
    "Unsupervised discovery of Uber pickup hotspots in New York City. "
    "Choose a day and an hour to see where demand concentrates."
)

# ── Drift detection ───────────────────────────────────────────────────────────
_drift_fb      = load_feedback()
_drift_total   = len(_drift_fb)
_drift_neg     = int((~_drift_fb["positive"].astype(bool)).sum()) if _drift_total else 0
_drift_rate    = _drift_neg / _drift_total if _drift_total else 0.0

if _drift_total >= 10:
    if _drift_rate > 0.40:
        st.error(
            f"⚠ Drift detected: {_drift_rate * 100:.0f}% negative feedback. "
            "Model retraining required."
        )
        st.code("python -m src.monitor --months jun14")
    elif _drift_rate > 0.25:
        st.warning("Feedback degrading. Monitoring.")

try:
    model = load_model()
    profile = load_profile()
    centroids = load_centroids()
    metrics = load_metrics()
except FileNotFoundError:
    st.error("No production model found. Run `python -m src.train` and "
             "`python -m src.promote` first.")
    st.stop()

with st.sidebar:
    st.header("Model in production")
    st.metric("Version", f"v{metrics.get('version', 1)}")
    st.metric("Zones (K)", metrics["k"])
    st.metric("Clustering quality (silhouette)", f"{metrics['silhouette']:.3f}")
    st.caption(f"Trained on: {', '.join(metrics['months'])}")
    st.caption(f"Promoted at: {metrics.get('promoted_at', 'n/a')[:19]}")
    st.divider()
    top_n = st.slider("Zones to display", 3, 20, 8)
    st.divider()
    _fb_sidebar = load_feedback()
    _total_votes  = len(_fb_sidebar)
    _neg_votes    = int((~_fb_sidebar["positive"].astype(bool)).sum()) if _total_votes else 0
    _neg_rate_pct = round(_neg_votes / _total_votes * 100, 1) if _total_votes else 0.0
    st.metric("Negative feedback rate", f"{_neg_rate_pct:.1f}%",
              help=f"{_neg_votes} negative out of {_total_votes} total votes")
    _dissatisfaction = min(_neg_votes / _total_votes, 1.0) if _total_votes else 0.0
    st.caption("Dissatisfaction level")
    st.progress(_dissatisfaction)

col1, col2 = st.columns(2)
day = col1.selectbox("Day of week", DAYS, index=4)
hour = col2.slider("Hour of day", 0, 23, 20)
dow = DAYS.index(day)

# All zones for the current time slot (background layer)
alla = (profile[(profile.dow == dow) & (profile.hour == hour)]
        .merge(centroids, left_on="zone", right_index=True, how="left"))

# Top-N selection with Thompson-sampling reranking
_fb      = load_feedback()
_fb_agg  = aggregate_feedback(_fb)

_slot = (
    profile[(profile.dow == dow) & (profile.hour == hour)]
    .copy()
)

# Normalise pickups to [0, 1]
_pmax_all = _slot["pickups"].max() or 1
_slot["pickups_norm"] = _slot["pickups"] / _pmax_all

def _thompson_score(row: pd.Series) -> float:
    """Blend normalised historical volume with Beta(1+wins, 1+losses) sample."""
    mask = (
        (_fb_agg["zone"] == row["zone"]) &
        (_fb_agg["dow"]  == dow)          &
        (_fb_agg["hour"] == hour)
    )
    if _fb_agg.empty or not mask.any():
        wins, losses = 0, 0          # no feedback → Beta(1,1) flat prior
    else:
        rec = _fb_agg[mask].iloc[0]
        wins, losses = int(rec["wins"]), int(rec["losses"])
    sample = np.random.beta(1 + wins, 1 + losses)
    return float(row["pickups_norm"]) * sample

_slot["score"] = _slot.apply(_thompson_score, axis=1)

sel = (
    _slot.sort_values("score", ascending=False)
    .head(top_n)
    .merge(centroids, left_on="zone", right_index=True, how="left")
)

if sel.empty:
    st.warning("No historical pickups for that slot.")
    st.stop()

log_query(dow, hour, int(sel.iloc[0]["zone"]))

st.subheader(f"Top demand zones — {day} at {hour:02d}:00")
st.caption("Zones ranked by historical pickup volume for the selected day and hour.")
st.info("Ranking combines historical volume with driver feedback. "
        "Zones improve or drop as drivers rate them.", icon="ℹ️")
sel["rank"] = range(1, len(sel) + 1)
sel["share"] = (sel.pickups / sel.pickups.sum() * 100).round(1)
sel["label"] = sel.apply(lambda r: f"#{int(r['rank'])} — {int(r['pickups'])} pickups", axis=1)

# Sqrt-scaled radius (300–1200 m) so small zones remain visible
import math as _math
_sqmin = _math.sqrt(sel.pickups.min())
_sqmax = _math.sqrt(sel.pickups.max())
_rmin, _rmax = 300, 1200

def _radius(p):
    sq = _math.sqrt(p)
    if _sqmax == _sqmin:
        return (_rmin + _rmax) // 2
    return int(_rmin + (_rmax - _rmin) * (sq - _sqmin) / (_sqmax - _sqmin))

sel["radius"] = sel["pickups"].apply(_radius)

# Per-zone colour: #1 darkest, rest lighter orange
sel["color"] = sel["rank"].apply(
    lambda r: [200, 50, 20, 220] if r == 1 else [240, 130, 80, 150]
)

# Rank as string for TextLayer
sel["rank_str"] = sel["rank"].astype(str)

st.pydeck_chart(pdk.Deck(
    map_style="light",
    height=560,
    initial_view_state=pdk.ViewState(
        latitude=40.7420, longitude=-73.9800,
        zoom=10.8, pitch=0,
    ),
    layers=[
        # Background: all zones in grey
        pdk.Layer(
            "ScatterplotLayer", data=alla,
            get_position=["Lon", "Lat"],
            get_radius=250,
            get_fill_color=[150, 150, 150, 70],
            pickable=False, stroked=False,
        ),
        # Foreground: top-N zones in orange, #1 darker
        pdk.Layer(
            "ScatterplotLayer", data=sel,
            get_position=["Lon", "Lat"],
            get_radius="radius",
            get_fill_color="color",
            pickable=True, stroked=True,
        ),
        # Rank labels on top of each circle
        pdk.Layer(
            "TextLayer", data=sel,
            get_position=["Lon", "Lat"],
            get_text="rank_str",
            get_size=16,
            get_color=[255, 255, 255],
            get_alignment_baseline="'center'",
        ),
    ],
    tooltip={"text": "{label}\n({share}%)"},
))
st.caption("Gray = all 25 zones · Orange = top zones for this time slot · Darkest = #1")

_total_pickups = int(sel.pickups.sum())
_top1_pct = float(sel.iloc[0]["share"])
_n_zones = len(sel)

_mc1, _mc2, _mc3 = st.columns(3)
_mc1.metric("Total pickups", f"{_total_pickups:,}")
_mc2.metric("Zone #1 share", f"{_top1_pct:.1f}%")
_mc3.metric("Zones shown", _n_zones)

st.dataframe(
    sel[["rank", "zone", "pickups", "share", "Lat", "Lon"]]
    .rename(columns={"rank": "#", "share": "share %"}),
    hide_index=True, use_container_width=True,
)

# ── Driver feedback ───────────────────────────────────────────────────────────
st.divider()
_top1_zone = int(sel.iloc[0]["zone"])
st.markdown(f"**Rate the #1 recommendation — Zone {_top1_zone}**")
_fb_col1, _fb_col2, _ = st.columns([1, 1, 4])
if _fb_col1.button("👍 Good recommendation", key="btn_good"):
    save_feedback(_top1_zone, dow, hour, positive=True)
    st.rerun()
if _fb_col2.button("👎 Not useful", key="btn_bad"):
    save_feedback(_top1_zone, dow, hour, positive=False)
    st.rerun()

with st.expander("Driver feedback"):
    _fb_exp = load_feedback()
    if _fb_exp.empty:
        st.caption("No feedback recorded yet.")
    else:
        _fb_exp_agg = (
            aggregate_feedback(_fb_exp)
            .sort_values("wins", ascending=False)
            .rename(columns={"success_rate": "success rate"})
        )
        _fb_exp_agg["total votes"] = _fb_exp_agg["wins"] + _fb_exp_agg["losses"]
        st.dataframe(
            _fb_exp_agg[["zone", "dow", "hour", "wins", "losses",
                          "success rate", "total votes"]],
            hide_index=True, use_container_width=True,
        )

# ── Hourly demand profile ────────────────────────────────────────────────────
st.divider()

_zone_options = {
    row["rank"]: f"#{int(row['rank'])} (zone {int(row['zone'])})"
    for _, row in sel.iterrows()
}
_chosen_rank = st.selectbox(
    "Hourly profile for zone",
    options=list(_zone_options.keys()),
    format_func=lambda r: _zone_options[r],
)
_chosen_zone = int(sel.loc[sel["rank"] == _chosen_rank, "zone"].iloc[0])

# Filter profile by zone + day, fill missing hours with 0
_zone_profile = (
    profile[(profile["zone"] == _chosen_zone) & (profile["dow"] == dow)]
    .groupby("hour", as_index=False)["pickups"]
    .sum()
)
_all_hours = pd.DataFrame({"hour": range(24)})
_zone_profile = (
    _all_hours.merge(_zone_profile, on="hour", how="left")
    .fillna({"pickups": 0})
    .astype({"pickups": int})
    .set_index("hour")
)

st.subheader(f"Hourly demand profile — Zone {_chosen_zone}")
st.bar_chart(_zone_profile["pickups"], use_container_width=True)

_peak_hour = int(_zone_profile["pickups"].idxmax())
_peak_pickups = int(_zone_profile["pickups"].max())
st.caption(f"Peak hour: {_peak_hour:02d}:00 with {_peak_pickups:,} pickups")

with st.expander("How it works"):
    st.markdown(
        """
**Model.** MiniBatchKMeans over pickup coordinates only. The timestamp is
never a feature: mixing degrees with hours would distort the geometry.
Instead, the timestamp builds the hourly demand profile of each zone after
clustering.

**K selection.** Silhouette score over a grid of candidate values.

**Maintenance.** Every query is logged, and a scheduled workflow feeds one
new month of pickups into the drift monitor. When cluster quality degrades
or too many pickups fall far from every centroid, the model is retrained and
promoted only if it beats the current one.
        """
    )
