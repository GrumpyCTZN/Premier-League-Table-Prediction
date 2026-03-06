"""
Page 5 — Team Deep Dive
========================
Strength gauges, points histogram, position probability heatmap row,
form badges, and a full head-to-head record table for any selected team.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator import simulate_season
from src.form      import seed_form_window, compute_form_multiplier
from src.downloader import PL_SEASONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running 1 000 simulations…")
def _get_sim_df(_fixtures: pd.DataFrame) -> pd.DataFrame:
    return simulate_season(_fixtures, n_simulations=1000)


def _gauge(value: float, title: str, ref: float = 1.0) -> go.Figure:
    """Single Plotly gauge chart for an attack/defense strength ratio."""
    color = "#3a86ff" if value >= ref else "#ff4444"
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = value,
        title  = dict(text=title, font=dict(color="white", size=13)),
        number = dict(suffix="×", font=dict(color="white", size=22)),
        gauge  = dict(
            axis      = dict(range=[0, 2.5], tickcolor="white",
                             tickfont=dict(color="white")),
            bar       = dict(color=color),
            bgcolor   = "#1e1e1e",
            threshold = dict(
                line  = dict(color="white", width=2),
                thickness = 0.75,
                value     = ref,
            ),
            steps = [
                dict(range=[0,   0.8], color="#3d0a0a"),
                dict(range=[0.8, 1.2], color="#1e1e1e"),
                dict(range=[1.2, 2.5], color="#0a3d1a"),
            ],
        ),
    ))
    fig.update_layout(
        height        = 220,
        margin        = dict(l=10, r=10, t=40, b=10),
        paper_bgcolor = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


def _points_histogram(sim_df: pd.DataFrame, team: str) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x           = sim_df[team],
        nbinsx      = 35,
        histnorm    = "probability density",
        marker_color= "#3a86ff",
        opacity     = 0.85,
        name        = team,
    ))
    mean_pts = sim_df[team].mean()
    fig.add_vline(
        x=mean_pts, line_dash="dash", line_color="#ffbe0b",
        annotation_text=f"Mean: {mean_pts:.1f}",
        annotation_font_color="#ffbe0b",
    )
    fig.update_layout(
        title         = f"{team} — Points Distribution (1 000 sims)",
        xaxis_title   = "Final Points",
        yaxis_title   = "Density",
        height        = 320,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


def _position_heatmap_row(sim_df: pd.DataFrame, team: str) -> go.Figure:
    """Single-row heatmap: probability of finishing in each position 1–20."""
    n_teams = len(sim_df.columns)
    ranks   = sim_df.rank(axis=1, ascending=False, method="min")
    pos_probs = [(ranks[team] == p).mean() for p in range(1, n_teams + 1)]

    fig = go.Figure(go.Heatmap(
        z           = [pos_probs],
        x           = [str(p) for p in range(1, n_teams + 1)],
        y           = [team],
        colorscale  = "Blues",
        showscale   = True,
        text        = [[f"{v:.1%}" for v in pos_probs]],
        texttemplate= "%{text}",
        hovertemplate="Position %{x}: %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title         = "Finish Position Probability (1st → 20th)",
        xaxis_title   = "Position",
        height        = 180,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


def _form_badges(results: list[str]) -> str:
    """Render last-5 form as coloured HTML badges."""
    badge_css = {
        "W": "background:#28a745;color:white;padding:3px 9px;"
             "border-radius:4px;font-weight:bold;margin:2px",
        "D": "background:#6c757d;color:white;padding:3px 9px;"
             "border-radius:4px;font-weight:bold;margin:2px",
        "L": "background:#dc3545;color:white;padding:3px 9px;"
             "border-radius:4px;font-weight:bold;margin:2px",
    }
    if not results:
        return "<span style='color:grey'>No data</span>"
    spans = "".join(
        f'<span style="{badge_css.get(r, "")}">{r}</span>'
        for r in results
    )
    return spans


def _h2h_full_table(pl_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """H2H record vs every other team: Home W/D/L · Away W/D/L · Overall W%."""
    opponents = sorted(
        (set(pl_df["HomeTeam"]) | set(pl_df["AwayTeam"])) - {team}
    )
    rows = []
    for opp in opponents:
        home_m = pl_df[(pl_df["HomeTeam"] == team) & (pl_df["AwayTeam"] == opp)]
        away_m = pl_df[(pl_df["HomeTeam"] == opp)  & (pl_df["AwayTeam"] == team)]

        hw = int((home_m["FTHG"] > home_m["FTAG"]).sum())
        hd = int((home_m["FTHG"] == home_m["FTAG"]).sum())
        hl = int((home_m["FTHG"] < home_m["FTAG"]).sum())

        aw = int((away_m["FTAG"] > away_m["FTHG"]).sum())
        ad = int((away_m["FTAG"] == away_m["FTHG"]).sum())
        al = int((away_m["FTAG"] < away_m["FTHG"]).sum())

        total = hw + hd + hl + aw + ad + al
        wins  = hw + aw
        w_pct = f"{100 * wins / total:.0f}%" if total > 0 else "—"

        rows.append({
            "Opponent":      opp,
            "Home W/D/L":    f"{hw}/{hd}/{hl}",
            "Away W/D/L":    f"{aw}/{ad}/{al}",
            "Total Played":  total,
            "Overall W%":    w_pct,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Team Deep Dive", page_icon="🔍", layout="wide")
st.title("🔍 Team Deep Dive")

if "model_a" not in st.session_state or "data" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

data          = st.session_state["data"]
model_a_state = st.session_state["model_a"]
fixtures      = model_a_state["fixtures"]
pl_df         = data["pl_df"]
strengths_df  = data["team_strengths"]

all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
team = st.selectbox("Select a team", all_teams)

sim_df = _get_sim_df(fixtures)

st.markdown("---")

# ── Strength Gauges ──────────────────────────────────────────────────────
st.subheader("Attack & Defense Strength Ratios")
st.caption("Ratio vs league average (1.0 = average). Higher attack = more goals scored. Lower defense = fewer conceded.")

team_row = strengths_df[strengths_df["Team"] == team]
if team_row.empty:
    st.info("No strength data available for this team.")
    ha_str = da_str = aa_str = dd_str = 1.0
else:
    r      = team_row.iloc[0]
    ha_str = float(r["home_attack_str"])
    da_str = float(r["home_defense_str"])
    aa_str = float(r["away_attack_str"])
    dd_str = float(r["away_defense_str"])

g1, g2, g3, g4 = st.columns(4)
with g1:
    st.plotly_chart(_gauge(ha_str, "Home Attack"), use_container_width=True)
with g2:
    st.plotly_chart(_gauge(da_str, "Home Defense (lower=better)", ref=1.0),
                    use_container_width=True)
with g3:
    st.plotly_chart(_gauge(aa_str, "Away Attack"), use_container_width=True)
with g4:
    st.plotly_chart(_gauge(dd_str, "Away Defense (lower=better)", ref=1.0),
                    use_container_width=True)

# ── Points Histogram ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Points Distribution (1 000 Simulations)")
st.plotly_chart(_points_histogram(sim_df, team), use_container_width=True)

# ── Position Probability Heatmap ──────────────────────────────────────────
st.markdown("---")
st.subheader("Finish Position Probability")
st.plotly_chart(_position_heatmap_row(sim_df, team), use_container_width=True)

# ── Form Badges ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Last 5 Form")
season_before = PL_SEASONS[-2]
form_results  = seed_form_window(pl_df, team, season_before)
form_mult     = compute_form_multiplier(form_results)

st.markdown(
    _form_badges(form_results) +
    f"&nbsp;&nbsp;<span style='color:#adb5bd'>Multiplier: ×{form_mult:.3f}"
    f"&nbsp;({season_before} season)</span>",
    unsafe_allow_html=True,
)

# ── Full H2H Table ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Head-to-Head Record vs Every Team")
h2h_df = _h2h_full_table(pl_df, team)
h2h_df = h2h_df[h2h_df["Total Played"] > 0].sort_values(
    "Total Played", ascending=False
)
st.dataframe(h2h_df, use_container_width=True, hide_index=True)