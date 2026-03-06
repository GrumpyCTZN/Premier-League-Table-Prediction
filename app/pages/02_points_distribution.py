"""
Page 2 — Points Distribution
==============================
Overlapping density histograms for 2–6 selected teams across 1 000
simulations, plus a pairwise probability matrix.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator import simulate_season

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running 1 000 simulations…")
def _get_sim_df(_fixtures: pd.DataFrame) -> pd.DataFrame:
    """Run and cache the raw simulation results (n_teams × 1000)."""
    return simulate_season(_fixtures, n_simulations=1000)


TEAM_COLORS = [
    "#3a86ff", "#ff006e", "#ffbe0b", "#8338ec",
    "#fb5607", "#06d6a0", "#ef233c", "#4cc9f0",
]


def _histogram_fig(sim_df: pd.DataFrame, teams: list[str]) -> go.Figure:
    """Overlapping semi-transparent density histograms for selected teams."""
    fig = go.Figure()
    for i, team in enumerate(teams):
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        fig.add_trace(go.Histogram(
            x           = sim_df[team],
            name        = team,
            opacity     = 0.65,
            nbinsx      = 35,
            histnorm    = "probability density",
            marker_color= color,
            hovertemplate=f"<b>{team}</b><br>Points: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode       = "overlay",
        title         = "Points Distribution across 1 000 Simulations",
        xaxis_title   = "Final Points",
        yaxis_title   = "Density",
        legend_title  = "Team",
        height        = 480,
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


def _probability_matrix(sim_df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """
    Build a matrix where cell [i, j] = P(team_i finishes above team_j).
    Diagonal is "—".
    """
    n = len(teams)
    matrix = pd.DataFrame(index=teams, columns=teams, dtype=object)

    for t in teams:
        matrix.loc[t, t] = "—"

    for t1, t2 in combinations(teams, 2):
        n_sims    = len(sim_df)
        p_t1_wins = (sim_df[t1] > sim_df[t2]).sum() / n_sims
        p_t2_wins = (sim_df[t2] > sim_df[t1]).sum() / n_sims
        matrix.loc[t1, t2] = f"{p_t1_wins:.1%}"
        matrix.loc[t2, t1] = f"{p_t2_wins:.1%}"

    return matrix


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Points Distribution", page_icon="📊", layout="wide")
st.title("📊 Points Distribution")
st.caption("Select 2–6 teams to compare their simulated points distributions.")

if "model_a" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

fixtures  = st.session_state["model_a"]["fixtures"]
all_teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))

# Team selector
default_teams = all_teams[:6]
selected_teams = st.multiselect(
    "Choose teams (2–6)",
    options  = all_teams,
    default  = default_teams[:4],
    max_selections = 6,
)

if len(selected_teams) < 2:
    st.info("Please select at least 2 teams.")
    st.stop()

# Run simulations
sim_df = _get_sim_df(fixtures)

# Histogram
fig = _histogram_fig(sim_df, selected_teams)
st.plotly_chart(fig, use_container_width=True)

# Summary stats
st.markdown("---")
st.subheader("Summary Statistics")
stats = pd.DataFrame({
    "Team":       selected_teams,
    "Mean Pts":   [round(sim_df[t].mean(), 1) for t in selected_teams],
    "Std Dev":    [round(sim_df[t].std(),  1) for t in selected_teams],
    "Min":        [int(sim_df[t].min())       for t in selected_teams],
    "Max":        [int(sim_df[t].max())       for t in selected_teams],
    "P(Top 4)":   [f"{(sim_df.rank(axis=1, ascending=False, method='min')[t] <= 4).mean():.1%}"
                   for t in selected_teams],
}).set_index("Team")
st.dataframe(stats, use_container_width=True)

# Probability matrix
st.markdown("---")
st.subheader("Head-to-Head Finish Probability")
st.caption("Cell [Row, Col] = probability that Row team finishes ABOVE Col team across 1 000 simulations.")
prob_matrix = _probability_matrix(sim_df, selected_teams)
st.dataframe(prob_matrix, use_container_width=True)