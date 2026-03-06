"""
Page 1 — Predicted League Table
================================
Runs 1 000 Monte Carlo simulations using the selected model and displays
the predicted final standings with colour-coded rows and a Plotly bar chart.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator import simulate_season, build_predicted_table
from src.model_b   import predict_base_goals_b, extract_implied_probs
from src.model     import precompute_expected_goals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running 1 000 simulations (Model A)…")
def run_simulations_a(_fixtures: pd.DataFrame) -> pd.DataFrame:
    """Run and cache 1 000 Model A season simulations."""
    sim_df = simulate_season(_fixtures, n_simulations=1000)
    return build_predicted_table(sim_df)


@st.cache_data(show_spinner="Running 1 000 simulations (Model B)…")
def run_simulations_b(
    _model_b,
    _fixtures_a: pd.DataFrame,
    _train_odds: pd.DataFrame,
    _h2h_table: pd.DataFrame,
    _champ_fallback: dict,
    _promoted_teams: set,
    target_season: str,
    league_avg_home: float,
    league_avg_away: float,
) -> pd.DataFrame:
    """
    Pre-compute Model B expected goals per fixture (using per-fixture
    implied_prob_home from training-set median), then simulate.
    """
    if _model_b is None:
        return None

    # Use median implied_prob_home from training data as a proxy for
    # "current market" when per-fixture odds are unavailable in test set
    median_imp = float(_train_odds["implied_prob_home"].median())

    fixtures_b = _fixtures_a.copy()
    mu_home_b, mu_away_b = [], []
    for _, row in fixtures_b.iterrows():
        mh, ma = predict_base_goals_b(
            _model_b, row["HomeTeam"], row["AwayTeam"], median_imp
        )
        mu_home_b.append(mh)
        mu_away_b.append(ma)

    fixtures_b["mu_home"] = mu_home_b
    fixtures_b["mu_away"] = mu_away_b

    sim_df = simulate_season(fixtures_b, n_simulations=1000)
    return build_predicted_table(sim_df)


def _row_color(pos: int, n_teams: int) -> str:
    """Return a hex background colour based on league position."""
    if pos <= 4:
        return "#1a3a5c"    # Champions League blue
    if pos <= 6:
        return "#5c3d00"    # Europa League amber
    if pos > n_teams - 3:
        return "#5c1a1a"    # Relegation red
    return "#1e1e1e"        # default dark


def _style_table(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply row-level background colours to the standings DataFrame."""
    n = len(table)

    def row_bg(row):
        pos = row.name  # 1-based index
        color = _row_color(pos, n)
        return [f"background-color: {color}; color: white"] * len(row)

    return (
        table.style
        .apply(row_bg, axis=1)
        .format({
            "Avg Points":    "{:.1f}",
            "Std Dev":       "{:.1f}",
            "Top-4 %":       "{:.1f}%",
            "Title %":       "{:.1f}%",
            "Relegation %":  "{:.1f}%",
        })
    )


def _points_bar_chart(table: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal bar chart of avg points with std-dev error bars."""
    n = len(table)
    colors = []
    for pos in range(1, n + 1):
        if pos <= 4:
            colors.append("#3a86ff")
        elif pos <= 6:
            colors.append("#ffbe0b")
        elif pos > n - 3:
            colors.append("#ff4444")
        else:
            colors.append("#6c757d")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = table["Avg Points"],
        y            = table["Team"],
        orientation  = "h",
        error_x      = dict(
            type      = "data",
            array     = table["Std Dev"],
            visible   = True,
            color     = "rgba(255,255,255,0.5)",
            thickness = 1.5,
            width     = 4,
        ),
        marker_color = colors,
        text         = table["Avg Points"].map("{:.1f}".format),
        textposition = "outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Points: %{x:.1f}<br>"
            "Std Dev: %{error_x.array:.1f}<extra></extra>"
        ),
    ))

    # Safety line
    fig.add_vline(
        x           = 40,
        line_dash   = "dash",
        line_color  = "tomato",
        opacity     = 0.6,
        annotation_text     = "Safety (~40 pts)",
        annotation_position = "top right",
        annotation_font_color = "tomato",
    )

    fig.update_layout(
        title       = title,
        xaxis_title = "Predicted Points",
        yaxis       = dict(autorange="reversed", tickfont=dict(size=12)),
        height      = max(500, n * 28),
        margin      = dict(l=20, r=60, t=60, b=40),
        paper_bgcolor = "#0e1117",
        plot_bgcolor  = "#0e1117",
        font          = dict(color="white"),
    )
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Predicted League Table", page_icon="⚽", layout="wide")
st.title("⚽ Predicted League Table")
st.caption("1 000 Monte Carlo simulations · Poisson GLM")

# Guard: ensure session state is populated
if "data" not in st.session_state or "model_a" not in st.session_state:
    st.warning("Please launch the app from `app/app.py` so data loads correctly.")
    st.stop()

data     = st.session_state["data"]
model_a  = st.session_state["model_a"]
model_b  = st.session_state["model_b"]

# Model toggle
model_choice = st.radio(
    "Select model",
    options=["Model A — Historical only", "Model B — Historical + Market signal"],
    horizontal=True,
)
use_model_b = model_choice.startswith("Model B")

# Run simulations
if use_model_b and model_b.get("model_b") is not None:
    table = run_simulations_b(
        model_b["model_b"],
        model_a["fixtures"],
        model_b["train_odds"],
        data["h2h_table"],
        data["champ_fallback"],
        model_a["promoted_teams"],
        data["target_season"],
        data["league_avg_home"],
        data["league_avg_away"],
    )
    chart_title = "Model B — Predicted League Table (Historical + Market Signal)"
else:
    if use_model_b:
        st.warning("Model B unavailable (odds data missing). Showing Model A.")
    table = run_simulations_a(model_a["fixtures"])
    chart_title = "Model A — Predicted League Table (Historical Only)"

# Legend
col1, col2, col3, col4 = st.columns(4)
col1.markdown("🔵 **Top 4** — Champions League")
col2.markdown("🟡 **5th–6th** — Europa League")
col3.markdown("⚪ **7th–17th** — Mid-table")
col4.markdown("🔴 **Bottom 3** — Relegation")

st.markdown("---")

# Styled table
st.subheader("Standings")
st.dataframe(
    _style_table(table),
    use_container_width=True,
    height=600,
)

st.markdown("---")

# Bar chart
st.subheader("Average Points with Uncertainty")
fig = _points_bar_chart(table, chart_title)
st.plotly_chart(fig, use_container_width=True)

# Key stats
st.markdown("---")
st.subheader("Key Probabilities")
top_team = table.iloc[0]
rel_teams = table.tail(3)

c1, c2, c3 = st.columns(3)
c1.metric("🏆 Title Favourites",    top_team["Team"],      f"{top_team['Title %']:.1f}% chance")
c2.metric("📈 Avg Points Leader",   top_team["Team"],      f"{top_team['Avg Points']:.1f} pts")
c3.metric("📉 Most at Risk",
          rel_teams.iloc[-1]["Team"],
          f"{rel_teams.iloc[-1]['Relegation %']:.1f}% relegation")